"""
Script-style reproducibility diagnostics for HELM vs KWDG runs.

This is intentionally notebook-like and hard-coded for local iteration.
It writes a case-by-case JSONL stream as it runs, plus a summary JSON report.
"""

from __future__ import annotations

import datetime as datetime_mod
import json
from collections import Counter, defaultdict
from typing import Any

import kwutil
import ubelt as ub

from magnet.backends.helm.helm_outputs import HelmOutputs
from magnet.backends.helm.helm_outputs import HelmRun
from magnet.backends.helm.helm_run_diff import HelmRunDiff
from magnet.utils import sankey_builder


HELM_DETAILS_FPATH = ub.Path('run_details.yaml')
# Use repository-local results by default to avoid HOME-dependent ambiguity.
KWDG_RESULTS_DPATH = (ub.Path.cwd() / 'results/helm').resolve()
REPORT_DPATH = ub.Path('dev/oneoff/repro_reports').ensuredir()


def parse_helm_version(version_text: str) -> tuple[int, ...]:
    """
    Parse version strings like "v0.3.0" into sortable tuples.
    """
    text = str(version_text).strip()
    if text.startswith('v'):
        text = text[1:]
    parts = []
    for tok in text.split('.'):
        if tok.isdigit():
            parts.append(int(tok))
        else:
            parts.append(0)
    return tuple(parts)


def parse_helm_run_dir(run_dir: str) -> dict[str, str]:
    """Parse HELM public run_dir path components."""
    p = ub.Path(run_dir)
    parts = list(p.parts)
    out = {
        'helm_suite_name': 'unknown',
        'helm_version': 'unknown',
        'run_leaf': p.name,
    }
    try:
        idx = parts.index('benchmark_output')
    except ValueError:
        idx = -1
    if idx >= 1:
        out['helm_suite_name'] = str(parts[idx - 1])
    if idx >= 0 and (idx + 2) < len(parts):
        # .../benchmark_output/runs/<version>/<run_spec_dir>
        out['helm_version'] = str(parts[idx + 2])
    else:
        out['helm_version'] = str(p.parent.name)
    return out


def infer_benchmark_group(
    run_spec_name: str | None,
    scenario_class: str | None = None,
) -> str:
    """Infer benchmark/scenario family key (e.g. babi_qa, mmlu, raft)."""
    text = (run_spec_name or '').strip()
    if text:
        idxs = [i for i in [text.find(':'), text.find(',')] if i >= 0]
        if idxs:
            head = text[: min(idxs)].strip()
        else:
            head = text
        if head:
            return head
    if scenario_class:
        base = str(scenario_class).split('.')[-1]
        if base.endswith('Scenario'):
            base = base[:-8]
        if base:
            return base
    return 'unknown'


def select_latest_helm_rows(helm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = defaultdict(list)
    for row in helm_rows:
        run_dir = ub.Path(row['run_dir'])
        parsed = parse_helm_run_dir(str(run_dir))
        version = parsed['helm_version']
        row = dict(row)
        row['helm_version'] = version
        row['helm_version_tuple'] = parse_helm_version(version)
        row['suite_name'] = parsed['helm_suite_name']
        row['benchmark_name'] = parsed['helm_suite_name']
        row['benchmark_group'] = infer_benchmark_group(
            row.get('run_spec_name', None),
            row.get('scenario_class', None),
        )
        by_name[row['run_spec_name']].append(row)

    latest_rows = []
    for _, items in by_name.items():
        best = max(
            items,
            key=lambda r: (
                r['helm_version_tuple'],
                str(r['run_dir']),
            ),
        )
        latest_rows.append(best)

    latest_rows = sorted(
        latest_rows,
        key=lambda r: (r.get('benchmark_name', ''), r.get('run_spec_name', '')),
    )
    return latest_rows


def load_kwdg_rows(results_dpath: ub.Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    finished_jobs = sorted(results_dpath.glob('*/DONE'))
    rows = []
    for fpath in ub.ProgIter(finished_jobs, desc='load kwdg runs'):
        dpath = fpath.parent
        try:
            config = kwutil.Json.coerce(dpath / 'job_config.json')
            run_spec_name = config['helm.run_entry']
            suites = HelmOutputs.coerce(dpath / 'benchmark_output').suites()
            if not suites:
                continue
            runs = suites[0].runs()
            if len(runs) != 1:
                continue
            run = runs[0]
            rows.append(
                {
                    'dpath': str(dpath),
                    'run_spec_name': run_spec_name,
                    'run': run,
                }
            )
        except Exception:
            continue

    lut = {}
    dups = defaultdict(list)
    for row in rows:
        name = row['run_spec_name']
        if name in lut:
            dups[name].append(row['dpath'])
        lut[name] = row
    if dups:
        print(f'WARNING: found {len(dups)} duplicate KWDG run_spec_name entries')
    return rows, lut


def aggregate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counter = Counter()
    diagnosis_counter = Counter()
    reason_counter = Counter()
    reason_by_priority = defaultdict(Counter)

    for row in rows:
        status = row.get('status', 'unknown')
        status_counter[status] += 1
        if status != 'compared':
            continue

        diag = row.get('diagnosis', {}) or {}
        diagnosis_counter[diag.get('label', 'unknown')] += 1

        for reason in diag.get('reasons', []) or []:
            name = reason.get('name', 'unknown')
            priority = reason.get('priority', None)
            reason_counter[name] += 1
            reason_by_priority[str(priority)][name] += 1

    out = {
        'n_rows': len(rows),
        'status_counts': dict(status_counter),
        'diagnosis_label_counts': dict(diagnosis_counter),
        'reason_counts': dict(reason_counter),
        'reason_counts_by_priority': {
            p: dict(c) for p, c in reason_by_priority.items()
        },
    }
    return out


def _normalize_primary_reasons(diag: dict[str, Any]) -> list[str]:
    raw = diag.get('primary_reason_names', []) or []
    if isinstance(raw, str):
        return [raw]
    out = [str(x) for x in raw if x is not None]
    return sorted(out)


def _all_reason_names(diag: dict[str, Any]) -> set[str]:
    reasons = diag.get('reasons', []) or []
    out = set()
    for r in reasons:
        if not isinstance(r, dict):
            continue
        name = r.get('name', None)
        if name is not None:
            out.add(str(name))
    # Also include primary reasons for compatibility with abbreviated records.
    for name in _normalize_primary_reasons(diag):
        out.add(str(name))
    return out


def _bucket_execution_state(reason_names: set[str]) -> str:
    has_dep = 'deployment_drift' in reason_names
    has_exec = 'execution_spec_drift' in reason_names
    if has_dep and has_exec:
        return 'deployment+execution'
    if has_dep:
        return 'deployment_only'
    if has_exec:
        return 'execution_only'
    return 'none'


def _bucket_dataset_state(reason_names: set[str]) -> str:
    has_error = 'dataset_overlap_error' in reason_names
    has_membership = bool(
        {'dataset_instance_drift', 'dataset_variant_drift'} & reason_names
    )
    has_input_prompt = bool(
        {'dataset_input_drift', 'request_prompt_drift'} & reason_names
    )
    if has_error:
        return 'error'
    if has_membership and has_input_prompt:
        return 'membership+input_prompt'
    if has_membership:
        return 'membership_only'
    if has_input_prompt:
        return 'input_prompt_only'
    return 'none'


def _bucket_eval_state(reason_names: set[str]) -> str:
    has_eval = 'evaluation_spec_drift' in reason_names
    has_completion = 'completion_content_drift' in reason_names
    if has_eval and has_completion:
        return 'eval_spec+completion'
    if has_eval:
        return 'eval_spec_only'
    if has_completion:
        return 'completion_only'
    return 'none'


def _bucket_core_state(reason_names: set[str]) -> str:
    if 'core_metric_drift' in reason_names:
        return 'core_drift'
    if 'no_comparable_core_metrics' in reason_names:
        return 'no_comparable_core'
    if 'bookkeeping_metric_drift' in reason_names:
        return 'bookkeeping_only'
    return 'core_match'


def _benchmark_or_suite(row: dict[str, Any]) -> str:
    group = row.get('benchmark_group', None)
    bench = row.get('benchmark_name', None)
    suite = row.get('suite_name', None)
    for cand in [group, bench, suite]:
        if cand is None:
            continue
        text = str(cand).strip()
        if text and text.lower() != 'unknown':
            return text
    return 'unknown'


def _performance_bucket(row: dict[str, Any]) -> str:
    """Bucket core metric agreement for plot readability."""
    va = row.get('value_agreement', {}) or {}
    core = ((va.get('by_class') or {}).get('core') or {})
    agree = core.get('agree_ratio', None)
    if agree is None:
        return 'n/a'
    try:
        x = float(agree)
    except Exception:
        return 'n/a'
    if x <= 0.0:
        return '0%'
    if x < 0.50:
        return '0-50%'
    if x < 0.75:
        return '50-75%'
    if x < 1.0:
        # Not requested explicitly, but needed to avoid hiding these cases.
        return '75-100%'
    return '100%'


def _varying_keys(rows: list[dict[str, Any]], keys: list[str]) -> list[str]:
    """Return keys that vary across rows (ignoring stages with 1 unique value)."""
    out = []
    for key in keys:
        vals = {r.get(key, None) for r in rows}
        if len(vals) > 1:
            out.append(key)
    return out


def build_sankey_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build normalized row records for diagnosis Sankey construction.

    Example:
        >>> rows = [
        ...     {'status': 'compared', 'diagnosis': {'label': 'reproduced', 'primary_priority': 0, 'primary_reason_names': ['no_detected_drift']}},
        ...     {'status': 'compared', 'diagnosis': {'label': 'multiple_primary_reasons', 'primary_priority': 0, 'primary_reason_names': ['deployment_drift', 'execution_spec_drift']}},
        ...     {'status': 'missing_kwdg_match', 'diagnosis': {'label': 'missing_kwdg_match'}},
        ... ]
        >>> sk = build_sankey_rows(rows)
        >>> assert len(sk) == 3
        >>> assert sk[0]['repro_outcome'] == 'reproduced_core'
        >>> assert sk[1]['execution_state'] == 'deployment+execution'
        >>> assert sk[2]['repro_outcome'] == 'not_compared'
        >>> assert sk[0]['benchmark_or_suite'] == 'unknown'
        >>> assert sk[0]['performance_bucket'] == 'n/a'
    """
    sink_rows = []
    for row in case_rows:
        status = str(row.get('status', 'unknown'))
        diag = row.get('diagnosis', {}) or {}
        label = str(diag.get('label', 'unknown'))
        reason_names = _all_reason_names(diag)
        reasons = _normalize_primary_reasons(diag)
        if reasons:
            reason_bucket = ' + '.join(reasons)
        else:
            reason_bucket = label

        p = diag.get('primary_priority', None)
        if isinstance(p, int):
            priority_bucket = f'P{p}'
        else:
            priority_bucket = 'P?'

        core_state = _bucket_core_state(reason_names)
        execution_state = _bucket_execution_state(reason_names)
        dataset_state = _bucket_dataset_state(reason_names)
        eval_state = _bucket_eval_state(reason_names)
        perf_bucket = _performance_bucket(row)
        benchmark_or_suite = _benchmark_or_suite(row)

        if status != 'compared':
            repro_outcome = 'not_compared'
            # keep non-compared flow compact in the sankey.
            priority_bucket = 'n/a'
            reason_bucket = label
            core_state = 'n/a'
            execution_state = 'n/a'
            dataset_state = 'n/a'
            eval_state = 'n/a'
            perf_bucket = 'n/a'
        else:
            if core_state in {'core_match', 'bookkeeping_only'}:
                repro_outcome = 'reproduced_core'
            elif core_state == 'core_drift':
                repro_outcome = 'non_reproduced_core'
            else:
                repro_outcome = 'unknown_core'

        sink_rows.append(
            {
                'status': status,
                'benchmark_or_suite': benchmark_or_suite,
                'repro_outcome': repro_outcome,
                'performance_bucket': perf_bucket,
                'core_state': core_state,
                'execution_state': execution_state,
                'dataset_state': dataset_state,
                'eval_state': eval_state,
                'diagnosis_label': label,
                'primary_priority': priority_bucket,
                'primary_reasons': reason_bucket,
            }
        )
    return sink_rows


def write_sankey_report(
    case_rows: list[dict[str, Any]],
    *,
    report_dpath: ub.Path,
    stamp: str,
) -> dict[str, Any]:
    """Build and write diagnosis Sankey artifacts."""
    sankey_rows = build_sankey_rows(case_rows)
    plotly_errors: list[str] = []
    stage_defs: dict[str, list[str]] = {
        'status': [
            'compared: HELM and KWDG pair was compared.',
            'missing_kwdg_match: no matching KWDG run for this HELM run_spec.',
            'error: comparison failed.',
        ],
        'bench': [
            'inferred scenario family from HELM run_spec_name (e.g. math, mmlu, raft).',
        ],
        'core%': [
            'core metric agreement bucket (value_agreement.by_class.core.agree_ratio):',
            '0% -> agree_ratio == 0.0',
            '0-50% -> 0.0 < agree_ratio < 0.5',
            '50-75% -> 0.5 <= agree_ratio < 0.75',
            '75-100% -> 0.75 <= agree_ratio < 1.0',
            '100% -> agree_ratio == 1.0',
            'n/a -> no comparable core metrics',
        ],
        'exec': [
            'execution-level run-spec drift summary.',
            'deployment+execution: deployment_drift + execution_spec_drift',
            'deployment_only: deployment_drift only',
            'execution_only: execution_spec_drift only',
            'none: no execution/deployment drift detected',
        ],
        'data': [
            'dataset/request drift summary.',
            'none: no detected dataset membership or input/prompt drift',
            'input_prompt_only: dataset_input_drift and/or request_prompt_drift',
            'membership_only: dataset_instance_drift and/or dataset_variant_drift',
            'membership+input_prompt: both membership and input/prompt drift',
            'error: dataset overlap computation failed',
        ],
        'eval': [
            'evaluation/content drift summary.',
            'none: no evaluation schema/content drift detected',
            'eval_spec_only: evaluation_spec_drift only',
            'completion_only: completion_content_drift only',
            'eval_spec+completion: both evaluation_spec_drift and completion_content_drift',
        ],
        'primary': [
            'concatenated primary_reason_names from diagnosis.',
            'Primary means lowest priority value (most upstream stage).',
        ],
        'core_outcome': [
            'high-level core reproducibility outcome.',
            'reproduced_core / non_reproduced_core / unknown_core',
        ],
        'core_state': [
            'core-metric reason bucket from diagnosis reasons.',
            'core_match / core_drift / bookkeeping_only / no_comparable_core',
        ],
        'diag': [
            'diagnosis.label (top-level diagnosis label).',
        ],
        'prio': [
            'diagnosis.primary_priority (0 is most upstream/significant).',
        ],
    }

    def _graph_key_text(title: str, stage_names: list[str]) -> str:
        lines: list[str] = []
        lines.append('Sankey Key')
        lines.append('----------')
        lines.append(f'Graph: {title}')
        lines.append('Stage order: ' + ' -> '.join(stage_names))
        lines.append('')
        for stage in stage_names:
            lines.append(f'{stage}:')
            defs = stage_defs.get(stage, ['(no definition available)'])
            for d in defs:
                lines.append(f'  {d}')
            lines.append('')
        return '\n'.join(lines).rstrip() + '\n'

    def _emit_graph(
        *,
        kind: str,
        title: str,
        rows: list[dict[str, Any]],
        root,
        stage_names: list[str],
    ) -> dict[str, Any]:
        graph = root.build_sankey(rows, label_fmt='{name}: {value}')
        graph_summary = graph.summarize(max_edges=300)
        plan_text = root.to_text()

        stem = report_dpath / f'diagnose_repro_sankey_{stamp}_{kind}'
        json_fpath = stem.augment(ext='.json')
        txt_fpath = stem.augment(ext='.txt')
        key_fpath = stem.augment(stemsuffix='_key', ext='.txt')
        html_fpath = stem.augment(ext='.html')
        png_fpath = stem.augment(ext='.png')
        jpg_fpath = stem.augment(ext='.jpg')

        node_labels, source, target, value = graph._to_sankey_data()
        payload = kwutil.Json.ensure_serializable(
            {
                'kind': kind,
                'title': title,
                'n_rows': len(rows),
                'rows': rows,
                'node_labels': node_labels,
                'source': source,
                'target': target,
                'value': value,
            }
        )
        json_fpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        txt_fpath.write_text(plan_text + '\n\n' + graph_summary + '\n')
        key_fpath.write_text(_graph_key_text(title, stage_names))

        out = {
            'json': str(json_fpath),
            'txt': str(txt_fpath),
            'key_txt': str(key_fpath),
            'html': None,
            'png': None,
            'jpg': None,
        }
        try:
            fig = graph.to_plotly(title=title)
            fig.write_html(str(html_fpath), include_plotlyjs='cdn')
            out['html'] = str(html_fpath)
            try:
                # Keep interactive HTML defaults; apply readability tuning only
                # to static JPG exports.
                import plotly.graph_objects as go

                fig_static = go.Figure(fig.to_dict())
                node_labels = payload.get('node_labels', [])
                max_label_len = max((len(str(x)) for x in node_labels), default=20)
                export_width = min(5200, max(2800, 1700 + max_label_len * 18))
                export_height = min(5200, max(2200, 1000 + len(node_labels) * 50))
                export_scale = 4.5
                fig_static.update_traces(
                    node=dict(pad=20, thickness=20),
                )
                fig_static.update_layout(
                    font_size=13,
                    width=export_width,
                    height=export_height,
                    margin=dict(l=30, r=30, t=70, b=30),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                )
                fig_static.write_image(str(jpg_fpath), scale=export_scale)
                out['jpg'] = str(jpg_fpath)
            except Exception as ex:
                plotly_errors.append(
                    f'[{kind}] unable to write sankey JPG: {ex!r}'
                )
        except Exception as ex:
            plotly_errors.append(
                f'[{kind}] unable to write sankey HTML/images: {ex!r}'
            )
        return out

    # Level 1: all attempts
    root_all = sankey_builder.Root(label=f'Run Pairs n={len(sankey_rows)}')
    status_group = root_all.group(by='status', name='status')
    compared = status_group['compared']
    compared.label = 'status: compared'
    status_group['missing_kwdg_match'].label = 'status: missing_kwdg_match'
    status_group['error'].label = 'status: error'
    status_group['unknown'].label = 'status: unknown'

    compared_rows = [r for r in sankey_rows if r.get('status') == 'compared']
    node = compared
    key_to_name = {
        'benchmark_or_suite': 'bench',
        'performance_bucket': 'core%',
        'execution_state': 'exec',
        'dataset_state': 'data',
        'eval_state': 'eval',
        'primary_reasons': 'primary',
    }
    stage_order = [
        'benchmark_or_suite',
        'performance_bucket',
        'execution_state',
        'dataset_state',
        'eval_state',
        'primary_reasons',
    ]
    selected_stage_keys = _varying_keys(compared_rows, stage_order)
    for key in selected_stage_keys:
        node = node.group(by=key, name=key_to_name[key])
    all_stage_names = ['status'] + [key_to_name[k] for k in selected_stage_keys]

    all_art = _emit_graph(
        kind='all_attempts',
        title='HELM/KWDG Reproducibility Diagnosis (All Attempts)',
        rows=sankey_rows,
        root=root_all,
        stage_names=all_stage_names,
    )

    # Level 2: compared-only detailed
    if compared_rows:
        root_comp = sankey_builder.Root(
            label=f'Compared Pairs n={len(compared_rows)}'
        )
        node2 = root_comp
        stage_order2 = [
            'benchmark_or_suite',
            'performance_bucket',
            'execution_state',
            'dataset_state',
            'eval_state',
            'primary_reasons',
        ]
        selected_stage_keys2 = _varying_keys(compared_rows, stage_order2)
        for key in selected_stage_keys2:
            node2 = node2.group(by=key, name=key_to_name[key])
        compared_stage_names = [key_to_name[k] for k in selected_stage_keys2]
        compared_art = _emit_graph(
            kind='compared_detail',
            title='HELM/KWDG Reproducibility Diagnosis (Compared Only)',
            rows=compared_rows,
            root=root_comp,
            stage_names=compared_stage_names,
        )
    else:
        compared_art = {
            'json': None,
            'txt': None,
            'key_txt': None,
            'html': None,
            'png': None,
            'jpg': None,
        }

    artifacts: dict[str, Any] = {
        # Backward-compatible keys point to the all-attempts sankey.
        'sankey_json': all_art['json'],
        'sankey_txt': all_art['txt'],
        'sankey_key_txt': all_art['key_txt'],
        'sankey_html': all_art['html'],
        'sankey_png': all_art['png'],
        'sankey_jpg': all_art['jpg'],
        # Additional detailed level.
        'sankey_compared_json': compared_art['json'],
        'sankey_compared_txt': compared_art['txt'],
        'sankey_compared_key_txt': compared_art['key_txt'],
        'sankey_compared_html': compared_art['html'],
        'sankey_compared_png': compared_art['png'],
        'sankey_compared_jpg': compared_art['jpg'],
        'sankey_compared_full_json': None,
        'sankey_compared_full_txt': None,
        'sankey_compared_full_key_txt': None,
        'sankey_compared_full_html': None,
        'sankey_compared_full_png': None,
        'sankey_compared_full_jpg': None,
        'plotly_error': (' | '.join(plotly_errors) if plotly_errors else None),
    }

    # Level 3: compared-only, full (unpruned) diagnostic pipeline.
    if compared_rows:
        root_comp_full = sankey_builder.Root(
            label=f'Compared Pairs (Full) n={len(compared_rows)}'
        )
        node3 = root_comp_full
        key_to_name_full = {
            'benchmark_or_suite': 'bench',
            'repro_outcome': 'core_outcome',
            'performance_bucket': 'core%',
            'core_state': 'core_state',
            'execution_state': 'exec',
            'dataset_state': 'data',
            'diagnosis_label': 'diag',
            'eval_state': 'eval',
            'primary_priority': 'prio',
            'primary_reasons': 'primary',
        }
        stage_order3 = [
            'benchmark_or_suite',
            'repro_outcome',
            'performance_bucket',
            'core_state',
            'execution_state',
            'dataset_state',
            'diagnosis_label',
            'eval_state',
            'primary_priority',
            'primary_reasons',
        ]
        for key in stage_order3:
            node3 = node3.group(by=key, name=key_to_name_full[key])
        compared_full_stage_names = [key_to_name_full[k] for k in stage_order3]
        compared_full_art = _emit_graph(
            kind='compared_full',
            title='HELM/KWDG Reproducibility Diagnosis (Compared Full Pipeline)',
            rows=compared_rows,
            root=root_comp_full,
            stage_names=compared_full_stage_names,
        )
        artifacts['sankey_compared_full_json'] = compared_full_art['json']
        artifacts['sankey_compared_full_txt'] = compared_full_art['txt']
        artifacts['sankey_compared_full_key_txt'] = compared_full_art['key_txt']
        artifacts['sankey_compared_full_html'] = compared_full_art['html']
        artifacts['sankey_compared_full_png'] = compared_full_art['png']
        artifacts['sankey_compared_full_jpg'] = compared_full_art['jpg']

    return kwutil.Json.ensure_serializable(artifacts)


def main():
    if not HELM_DETAILS_FPATH.exists():
        raise FileNotFoundError(
            f'Expected HELM detail file at {HELM_DETAILS_FPATH}'
        )

    helm_rows = kwutil.Yaml.load(HELM_DETAILS_FPATH)
    latest_helm_rows = select_latest_helm_rows(helm_rows)

    if not latest_helm_rows:
        raise RuntimeError('No HELM rows found')

    kwdg_rows, kwdg_lut = load_kwdg_rows(KWDG_RESULTS_DPATH)

    print(f'Loaded HELM rows: all={len(helm_rows)} latest_only={len(latest_helm_rows)}')
    print(f'Loaded KWDG rows: {len(kwdg_rows)}')

    stamp = datetime_mod.datetime.now(datetime_mod.UTC).strftime('%Y%m%dT%H%M%SZ')
    case_jsonl_fpath = REPORT_DPATH / f'diagnose_repro_cases_{stamp}.jsonl'
    summary_json_fpath = REPORT_DPATH / f'diagnose_repro_summary_{stamp}.json'

    all_case_rows = []
    with case_jsonl_fpath.open('w', encoding='utf8') as file:
        for idx, helm_row in enumerate(
            ub.ProgIter(latest_helm_rows, desc='compare latest helm vs kwdg'), start=1
        ):
            run_spec_name = helm_row['run_spec_name']
            kwrow = kwdg_lut.get(run_spec_name, None)
            case_row = {
                'index': idx,
                'run_spec_name': run_spec_name,
                'benchmark_name': helm_row.get('benchmark_name', 'unknown'),
                'benchmark_group': helm_row.get('benchmark_group', 'unknown'),
                'suite_name': helm_row.get('suite_name', 'unknown'),
                'model_name': helm_row.get('model', None),
                'helm_version': helm_row.get('helm_version', None),
                'helm_run_dir': str(helm_row['run_dir']),
                'kwdg_run_dir': None if kwrow is None else kwrow['dpath'],
            }

            if kwrow is None:
                case_row.update(
                    {
                        'status': 'missing_kwdg_match',
                        'diagnosis': {
                            'label': 'missing_kwdg_match',
                            'primary_priority': 0,
                            'primary_reason_names': ['missing_kwdg_match'],
                            'reasons': [
                                {
                                    'name': 'missing_kwdg_match',
                                    'priority': 0,
                                    'details': {},
                                }
                            ],
                        },
                    }
                )
                print(
                    f'[{idx:03d}] {run_spec_name} -> missing_kwdg_match'
                )
            else:
                try:
                    helm_run = HelmRun.coerce(helm_row['run_dir'])
                    kwdg_run = kwrow['run']
                    rd = HelmRunDiff(
                        run_a=helm_run,
                        run_b=kwdg_run,
                        a_name='HELM',
                        b_name='KWDG',
                    )
                    summary = rd.summary_dict(level=20)
                    diag = summary.get('diagnosis', {}) or {}

                    case_row.update(
                        {
                            'status': 'compared',
                            'diagnosis': diag,
                            'run_spec_semantic': summary.get(
                                'run_spec_semantic', None
                            ),
                            'scenario_semantic': summary.get(
                                'scenario_semantic', None
                            ),
                            'dataset_overlap': summary.get(
                                'dataset_overlap', None
                            ),
                            'stats_coverage_by_name': summary.get(
                                'stats_coverage_by_name', None
                            ),
                            'stats_coverage_by_name_count': summary.get(
                                'stats_coverage_by_name_count', None
                            ),
                            'value_agreement': summary.get(
                                'value_agreement', None
                            ),
                            'instance_value_agreement': summary.get(
                                'instance_value_agreement', None
                            ),
                        }
                    )
                    primary = diag.get('label', 'unknown')
                    p = diag.get('primary_priority', None)
                    primary_names = diag.get('primary_reason_names', []) or []
                    print(
                        f'[{idx:03d}] {run_spec_name} -> {primary} '
                        f'(p={p}, primary_reasons={primary_names})'
                    )
                    reasons = diag.get('reasons', []) or []
                    for reason in reasons:
                        if reason.get('name') == 'deployment_drift':
                            det = reason.get('details', {}) or {}
                            print(
                                '      deployment: '
                                f'{det.get("a_value", None)!r} -> {det.get("b_value", None)!r}'
                            )
                except Exception as ex:
                    case_row.update(
                        {
                            'status': 'error',
                            'error': repr(ex),
                            'diagnosis': {
                                'label': 'comparison_error',
                                'primary_priority': 0,
                                'primary_reason_names': ['comparison_error'],
                                'reasons': [
                                    {
                                        'name': 'comparison_error',
                                        'priority': 0,
                                        'details': {'error': repr(ex)},
                                    }
                                ],
                            },
                        }
                    )
                    print(f'[{idx:03d}] {run_spec_name} -> ERROR: {ex!r}')

            case_row = kwutil.Json.ensure_serializable(case_row)
            file.write(json.dumps(case_row, ensure_ascii=False) + '\n')
            file.flush()
            all_case_rows.append(case_row)

    summary_report = {
        'report_case_jsonl': str(case_jsonl_fpath),
        'report_summary_json': str(summary_json_fpath),
        'generated_utc': stamp,
        'inputs': {
            'helm_detail_fpath': str(HELM_DETAILS_FPATH),
            'kwdg_results_dpath': str(KWDG_RESULTS_DPATH),
            'n_helm_rows_all': len(helm_rows),
            'n_helm_rows_latest': len(latest_helm_rows),
            'n_kwdg_rows': len(kwdg_rows),
        },
        'aggregate': aggregate_report(all_case_rows),
    }
    try:
        sankey_artifacts = write_sankey_report(
            all_case_rows, report_dpath=REPORT_DPATH, stamp=stamp
        )
    except Exception as ex:
        sankey_artifacts = {
            'sankey_json': None,
            'sankey_txt': None,
            'sankey_key_txt': None,
            'sankey_html': None,
            'sankey_png': None,
            'sankey_jpg': None,
            'sankey_compared_json': None,
            'sankey_compared_txt': None,
            'sankey_compared_key_txt': None,
            'sankey_compared_html': None,
            'sankey_compared_png': None,
            'sankey_compared_jpg': None,
            'sankey_compared_full_json': None,
            'sankey_compared_full_txt': None,
            'sankey_compared_full_key_txt': None,
            'sankey_compared_full_html': None,
            'sankey_compared_full_png': None,
            'sankey_compared_full_jpg': None,
            'plotly_error': f'failed to build sankey report: {ex!r}',
        }
    summary_report['artifacts'] = sankey_artifacts

    summary_report = kwutil.Json.ensure_serializable(summary_report)
    summary_json_fpath.write_text(
        json.dumps(summary_report, indent=2, ensure_ascii=False)
    )

    print('---')
    print(f'Wrote case report: {case_jsonl_fpath}')
    print(f'Wrote summary report: {summary_json_fpath}')
    if sankey_artifacts.get('sankey_json', None):
        print(f'Wrote sankey JSON: {sankey_artifacts["sankey_json"]}')
    if sankey_artifacts.get('sankey_txt', None):
        print(f'Wrote sankey TXT: {sankey_artifacts["sankey_txt"]}')
    if sankey_artifacts.get('sankey_html', None):
        print(f'Wrote sankey HTML: {sankey_artifacts["sankey_html"]}')
    if sankey_artifacts.get('sankey_png', None):
        print(f'Wrote sankey PNG: {sankey_artifacts["sankey_png"]}')
    if sankey_artifacts.get('sankey_jpg', None):
        print(f'Wrote sankey JPG: {sankey_artifacts["sankey_jpg"]}')
    if sankey_artifacts.get('sankey_key_txt', None):
        print(f'Wrote sankey key: {sankey_artifacts["sankey_key_txt"]}')
    if sankey_artifacts.get('sankey_compared_json', None):
        print(
            f'Wrote compared-detail sankey JSON: '
            f'{sankey_artifacts["sankey_compared_json"]}'
        )
    if sankey_artifacts.get('sankey_compared_txt', None):
        print(
            f'Wrote compared-detail sankey TXT: '
            f'{sankey_artifacts["sankey_compared_txt"]}'
        )
    if sankey_artifacts.get('sankey_compared_html', None):
        print(
            f'Wrote compared-detail sankey HTML: '
            f'{sankey_artifacts["sankey_compared_html"]}'
        )
    if sankey_artifacts.get('sankey_compared_png', None):
        print(
            f'Wrote compared-detail sankey PNG: '
            f'{sankey_artifacts["sankey_compared_png"]}'
        )
    if sankey_artifacts.get('sankey_compared_jpg', None):
        print(
            f'Wrote compared-detail sankey JPG: '
            f'{sankey_artifacts["sankey_compared_jpg"]}'
        )
    if sankey_artifacts.get('sankey_compared_key_txt', None):
        print(
            f'Wrote compared-detail sankey key: '
            f'{sankey_artifacts["sankey_compared_key_txt"]}'
        )
    if sankey_artifacts.get('sankey_compared_full_json', None):
        print(
            f'Wrote compared-full sankey JSON: '
            f'{sankey_artifacts["sankey_compared_full_json"]}'
        )
    if sankey_artifacts.get('sankey_compared_full_txt', None):
        print(
            f'Wrote compared-full sankey TXT: '
            f'{sankey_artifacts["sankey_compared_full_txt"]}'
        )
    if sankey_artifacts.get('sankey_compared_full_html', None):
        print(
            f'Wrote compared-full sankey HTML: '
            f'{sankey_artifacts["sankey_compared_full_html"]}'
        )
    if sankey_artifacts.get('sankey_compared_full_png', None):
        print(
            f'Wrote compared-full sankey PNG: '
            f'{sankey_artifacts["sankey_compared_full_png"]}'
        )
    if sankey_artifacts.get('sankey_compared_full_jpg', None):
        print(
            f'Wrote compared-full sankey JPG: '
            f'{sankey_artifacts["sankey_compared_full_jpg"]}'
        )
    if sankey_artifacts.get('sankey_compared_full_key_txt', None):
        print(
            f'Wrote compared-full sankey key: '
            f'{sankey_artifacts["sankey_compared_full_key_txt"]}'
        )
    if sankey_artifacts.get('plotly_error', None):
        print(f'Sankey note: {sankey_artifacts["plotly_error"]}')
    print(
        'Diagnosis label counts: '
        + ub.urepr(summary_report['aggregate']['diagnosis_label_counts'], nl=0)
    )


if __name__ == '__main__':
    main()
