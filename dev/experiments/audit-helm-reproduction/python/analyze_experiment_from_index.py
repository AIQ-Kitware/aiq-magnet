from __future__ import annotations

import argparse
import csv
import datetime as datetime_mod
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from aggregate_core_reports import _find_curve_value, _find_pair, _write_latest_alias
from common import audit_root, default_report_root, env_defaults
from paper_labels import load_paper_label_manager
from rebuild_core_report_from_index import latest_index_csv, load_rows, slugify


def _load_json(fpath: Path) -> dict[str, Any]:
    return json.loads(fpath.read_text())


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("-inf")


def _is_truthy_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _latest_matching_aiq_gpu_row(
    rows: list[dict[str, Any]],
    *,
    run_entry: str,
    exclude_experiment_name: str,
) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        if row.get("run_entry") != run_entry:
            continue
        if row.get("experiment_name") == exclude_experiment_name:
            continue
        if row.get("machine_host") != "aiq-gpu":
            continue
        if row.get("status") not in {"computed", "reused", "unknown", ""}:
            continue
        if not _is_truthy_text(row.get("has_run_spec")):
            continue
        run_dir = row.get("run_dir")
        if not run_dir:
            continue
        candidates.append(row)
    if not candidates:
        return None
    candidates.sort(
        key=lambda r: (
            _coerce_float(r.get("manifest_timestamp")),
            str(r.get("experiment_name") or ""),
            str(r.get("job_id") or ""),
        ),
        reverse=True,
    )
    return candidates[0]


def _latest_pair_report(report_dpath: Path) -> tuple[Path | None, Path | None]:
    json_cands = sorted(report_dpath.glob("pair_report_*.json"), reverse=True)
    txt_cands = sorted(report_dpath.glob("pair_report_*.txt"), reverse=True)
    return (
        json_cands[0] if json_cands else None,
        txt_cands[0] if txt_cands else None,
    )


def _write_latest_pair_aliases(report_dpath: Path) -> dict[str, str]:
    json_fpath, txt_fpath = _latest_pair_report(report_dpath)
    created: dict[str, str] = {}
    if json_fpath is not None:
        _write_latest_alias(json_fpath, report_dpath, "pair_report.latest.json")
        created["pair_report.latest.json"] = str(report_dpath / "pair_report.latest.json")
    if txt_fpath is not None:
        _write_latest_alias(txt_fpath, report_dpath, "pair_report.latest.txt")
        created["pair_report.latest.txt"] = str(report_dpath / "pair_report.latest.txt")
    return created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--index-fpath', default=None)
    parser.add_argument('--index-dpath', default=str(default_report_root() / 'indexes'))
    parser.add_argument('--allow-single-repeat', action='store_true')
    args = parser.parse_args()

    index_fpath = (
        Path(args.index_fpath).expanduser().resolve()
        if args.index_fpath else
        latest_index_csv(Path(args.index_dpath).expanduser().resolve())
    )
    rows = load_rows(index_fpath)
    experiment_rows = [r for r in rows if r.get('experiment_name') == args.experiment_name]
    if not experiment_rows:
        raise SystemExit(f'No rows found for experiment_name={args.experiment_name!r}')
    run_entries = sorted({r.get('run_entry') for r in experiment_rows if r.get('run_entry')})

    out_dpath = default_report_root() / f'experiment-analysis-{slugify(args.experiment_name)}'
    out_dpath.mkdir(parents=True, exist_ok=True)
    reports_dpath = out_dpath / 'core-reports'
    reports_dpath.mkdir(parents=True, exist_ok=True)

    rebuild_script = audit_root() / 'python' / 'rebuild_core_report_from_index.py'
    built_report_paths = []
    skipped_run_entries: list[dict[str, Any]] = []
    for run_entry in run_entries:
        report_dpath = reports_dpath / f'core-metrics-{slugify(run_entry)}'
        cmd = [
            env_defaults()['AIQ_PYTHON'],
            str(rebuild_script),
            '--run-entry', str(run_entry),
            '--index-fpath', str(index_fpath),
            '--experiment-name', str(args.experiment_name),
            '--report-dpath', str(report_dpath),
        ]
        if args.allow_single_repeat:
            cmd.append('--allow-single-repeat')
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as ex:
            skipped_run_entries.append({
                'run_entry': run_entry,
                'reason': 'rebuild_failed',
                'returncode': ex.returncode,
            })
            continue
        built_report_paths.append(report_dpath / 'core_metric_report.latest.json')

    summary_rows = []
    for report_json in built_report_paths:
        if not report_json.exists():
            continue
        report = _load_json(report_json)
        report_dir = report_json.parent
        selection_fpath = report_dir / 'report_selection.latest.json'
        selection = _load_json(selection_fpath) if selection_fpath.exists() else {}
        repeat = _find_pair(report, 'kwdagger_repeat')
        official = _find_pair(report, 'official_vs_kwdagger')
        summary_rows.append({
            'experiment_name': args.experiment_name,
            'run_spec_name': report.get('run_spec_name'),
            'run_entry': selection.get('run_entry'),
            'report_dir': str(report_dir),
            'generated_utc': report.get('generated_utc'),
            'diagnostic_flags': report.get('diagnostic_flags', []),
            'kwdagger_a_empty_completion_rate': (((report.get('run_diagnostics') or {}).get('kwdagger_a') or {}).get('empty_completion_rate')),
            'kwdagger_a_mean_output_tokens': ((((report.get('run_diagnostics') or {}).get('kwdagger_a') or {}).get('output_token_count') or {}).get('mean')),
            'official_empty_completion_rate': (((report.get('run_diagnostics') or {}).get('official') or {}).get('empty_completion_rate')),
            'official_mean_output_tokens': ((((report.get('run_diagnostics') or {}).get('official') or {}).get('output_token_count') or {}).get('mean')),
            'repeat_instance_agree_0': _find_curve_value(repeat.get('instance_level', {}).get('agreement_vs_abs_tol', []), 0.0),
            'official_instance_agree_0': _find_curve_value(official.get('instance_level', {}).get('agreement_vs_abs_tol', []), 0.0),
            'official_instance_agree_01': _find_curve_value(official.get('instance_level', {}).get('agreement_vs_abs_tol', []), 0.1),
            'official_instance_agree_025': _find_curve_value(official.get('instance_level', {}).get('agreement_vs_abs_tol', []), 0.25),
            'official_instance_agree_05': _find_curve_value(official.get('instance_level', {}).get('agreement_vs_abs_tol', []), 0.5),
            'official_runlevel_p90': (((official.get('run_level') or {}).get('overall_quantiles') or {}).get('abs_delta') or {}).get('p90'),
            'official_runlevel_max': (((official.get('run_level') or {}).get('overall_quantiles') or {}).get('abs_delta') or {}).get('max'),
        })

    compare_pair_script = audit_root() / 'python' / 'compare_pair.py'
    paper_labels = load_paper_label_manager(style='paper_short')
    summary_by_run_spec = {
        row['run_spec_name']: row
        for row in summary_rows
        if row.get('run_spec_name')
    }
    cross_machine_rows: list[dict[str, Any]] = []
    for run_spec_name, summary_row in summary_by_run_spec.items():
        run_entry = summary_row.get('run_entry')
        if not run_entry:
            continue
        experiment_match = next(
            (
                row for row in experiment_rows
                if row.get('run_entry') == run_entry and _is_truthy_text(row.get('has_run_spec')) and row.get('run_dir')
            ),
            None,
        )
        if experiment_match is None:
            continue
        aiq_gpu_match = _latest_matching_aiq_gpu_row(
            rows,
            run_entry=run_entry,
            exclude_experiment_name=args.experiment_name,
        )
        if aiq_gpu_match is None:
            continue
        machine_a = 'aiq-gpu'
        machine_b = str(experiment_match.get('machine_host') or args.experiment_name)
        display_label_a = paper_labels.machine_label(machine_a)
        display_label_b = paper_labels.machine_label(machine_b)
        cross_report_dpath = Path(summary_row['report_dir']) / 'cross-machine-aiq-gpu'
        cross_report_dpath.mkdir(parents=True, exist_ok=True)
        compare_cmd = [
            env_defaults()['AIQ_PYTHON'],
            str(compare_pair_script),
            '--run-a', str(aiq_gpu_match['run_dir']),
            '--run-b', str(experiment_match['run_dir']),
            '--label-a', machine_a,
            '--label-b', machine_b,
            '--display-label-a', display_label_a,
            '--display-label-b', display_label_b,
            '--report-dpath', str(cross_report_dpath),
        ]
        subprocess.run(compare_cmd, check=True)
        latest_links = _write_latest_pair_aliases(cross_report_dpath)
        cross_json_fpath = cross_report_dpath / 'pair_report.latest.json'
        cross_txt_fpath = cross_report_dpath / 'pair_report.latest.txt'
        cross_payload = _load_json(cross_json_fpath) if cross_json_fpath.exists() else {}
        strict = cross_payload.get('strict_summary', {}) or {}
        cross_diag = (strict.get('diagnosis', {}) or {})
        cross_overall = ((strict.get('value_agreement', {}) or {}).get('overall', {}) or {})
        cross_means = ((strict.get('instance_value_agreement', {}) or {}).get('means', {}) or {})
        row = {
            'run_spec_name': run_spec_name,
            'run_entry': run_entry,
            'machine_a': machine_a,
            'machine_b': machine_b,
            'machine_a_display': display_label_a,
            'machine_b_display': display_label_b,
            'report_dir': str(cross_report_dpath),
            'report_json': str(cross_json_fpath) if cross_json_fpath.exists() else None,
            'report_txt': str(cross_txt_fpath) if cross_txt_fpath.exists() else None,
            'diagnosis_label': cross_diag.get('label'),
            'primary_reason_names': cross_diag.get('primary_reason_names'),
            'run_level_agree_ratio': cross_overall.get('agree_ratio'),
            'instance_level_agree_ratio': cross_means.get('agree_ratio'),
            'run_level_abs_p90': ((((cross_payload.get('distance_summary') or {}).get('run_level') or {}).get('overall') or {}).get('abs_delta') or {}).get('p90'),
            'run_level_abs_max': ((((cross_payload.get('distance_summary') or {}).get('run_level') or {}).get('overall') or {}).get('abs_delta') or {}).get('max'),
            'instance_level_abs_p90': ((((cross_payload.get('distance_summary') or {}).get('instance_level') or {}).get('overall') or {}).get('abs_delta') or {}).get('p90'),
            'instance_level_abs_max': ((((cross_payload.get('distance_summary') or {}).get('instance_level') or {}).get('overall') or {}).get('abs_delta') or {}).get('max'),
            'latest_links': latest_links,
        }
        summary_row['cross_machine_aiq_gpu'] = row
        cross_machine_rows.append(row)

    stamp = datetime_mod.datetime.now(datetime_mod.UTC).strftime('%Y%m%dT%H%M%SZ')
    history_dpath = out_dpath / '.history' / stamp[:8]
    history_dpath.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(summary_rows).sort_values('run_spec_name')
    json_fpath = history_dpath / f'experiment_summary_{stamp}.json'
    csv_fpath = history_dpath / f'experiment_summary_{stamp}.csv'
    txt_fpath = history_dpath / f'experiment_summary_{stamp}.txt'

    payload = {
        'generated_utc': stamp,
        'experiment_name': args.experiment_name,
        'index_fpath': str(index_fpath),
        'n_run_entries': len(run_entries),
        'n_built_reports': len(summary_rows),
        'n_skipped_run_entries': len(skipped_run_entries),
        'run_entries': run_entries,
        'skipped_run_entries': skipped_run_entries,
        'cross_machine_rows': cross_machine_rows,
        'rows': summary_rows,
    }
    json_fpath.write_text(json.dumps(payload, indent=2))
    table.to_csv(csv_fpath, index=False)

    lines = []
    lines.append('Experiment Analysis Summary')
    lines.append('')
    lines.append(f'generated_utc: {stamp}')
    lines.append(f'experiment_name: {args.experiment_name}')
    lines.append(f'index_fpath: {index_fpath}')
    lines.append(f'n_run_entries: {len(run_entries)}')
    lines.append(f'n_built_reports: {len(summary_rows)}')
    lines.append(f'n_skipped_run_entries: {len(skipped_run_entries)}')
    lines.append('')
    lines.append('run_entries:')
    for run_entry in run_entries:
        lines.append(f'  - {run_entry}')
    if skipped_run_entries:
        lines.append('')
        lines.append('skipped_run_entries:')
        for item in skipped_run_entries:
            lines.append(f"  - run_entry: {item['run_entry']}")
            lines.append(f"    reason: {item['reason']}")
            lines.append(f"    returncode: {item['returncode']}")
    lines.append('')
    if cross_machine_rows:
        lines.append('cross_machine_aiq_gpu:')
        for row in cross_machine_rows:
            lines.append(f"  - run_spec_name: {row['run_spec_name']}")
            lines.append(f"    machine_a: {row['machine_a']}")
            lines.append(f"    machine_a_display: {row['machine_a_display']}")
            lines.append(f"    machine_b: {row['machine_b']}")
            lines.append(f"    machine_b_display: {row['machine_b_display']}")
            lines.append(f"    report_dir: {row['report_dir']}")
            lines.append(f"    report_txt: {row['report_txt']}")
            lines.append(f"    diagnosis_label: {row['diagnosis_label']}")
            lines.append(f"    primary_reason_names: {row['primary_reason_names']}")
            lines.append(f"    run_level_agree_ratio: {row['run_level_agree_ratio']}")
            lines.append(f"    instance_level_agree_ratio: {row['instance_level_agree_ratio']}")
            lines.append(f"    run_level_abs_p90: {row['run_level_abs_p90']}")
            lines.append(f"    run_level_abs_max: {row['run_level_abs_max']}")
            lines.append(f"    instance_level_abs_p90: {row['instance_level_abs_p90']}")
            lines.append(f"    instance_level_abs_max: {row['instance_level_abs_max']}")
        lines.append('')
    lines.append('per_run_spec:')
    for row in summary_rows:
        lines.append(f"  - run_spec_name: {row['run_spec_name']}")
        lines.append(f"    report_dir: {row['report_dir']}")
        lines.append(f"    diagnostic_flags: {row['diagnostic_flags']}")
        lines.append(f"    kwdagger_a_empty_completion_rate: {row['kwdagger_a_empty_completion_rate']}")
        lines.append(f"    kwdagger_a_mean_output_tokens: {row['kwdagger_a_mean_output_tokens']}")
        lines.append(f"    official_empty_completion_rate: {row['official_empty_completion_rate']}")
        lines.append(f"    official_mean_output_tokens: {row['official_mean_output_tokens']}")
        lines.append(f"    repeat_instance_agree_0: {row['repeat_instance_agree_0']}")
        lines.append(f"    official_instance_agree_0: {row['official_instance_agree_0']}")
        lines.append(f"    official_instance_agree_01: {row['official_instance_agree_01']}")
        lines.append(f"    official_instance_agree_025: {row['official_instance_agree_025']}")
        lines.append(f"    official_instance_agree_05: {row['official_instance_agree_05']}")
        lines.append(f"    official_runlevel_p90: {row['official_runlevel_p90']}")
        lines.append(f"    official_runlevel_max: {row['official_runlevel_max']}")
    txt_fpath.write_text('\n'.join(lines) + '\n')

    _write_latest_alias(json_fpath, out_dpath, 'experiment_summary.latest.json')
    _write_latest_alias(csv_fpath, out_dpath, 'experiment_summary.latest.csv')
    _write_latest_alias(txt_fpath, out_dpath, 'experiment_summary.latest.txt')

    print(f'Wrote experiment summary json: {json_fpath}')
    print(f'Wrote experiment summary csv: {csv_fpath}')
    print(f'Wrote experiment summary txt: {txt_fpath}')


if __name__ == '__main__':
    main()
