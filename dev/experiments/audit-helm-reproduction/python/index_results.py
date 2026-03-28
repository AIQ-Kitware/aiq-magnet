from __future__ import annotations

import argparse
import datetime as datetime_mod
import json
from collections import Counter
from pathlib import Path
from typing import Any

import kwutil
import pandas as pd

from common import default_report_root, env_defaults
from magnet.backends.helm.cli.materialize_helm_run import parse_run_entry_description
from magnet.backends.helm.helm_outputs import HelmOutputs


def _safe_json_load(fpath: Path) -> dict[str, Any]:
    if not fpath.exists():
        return {}
    try:
        return json.loads(fpath.read_text())
    except Exception:
        return {}


def _first_run_dir(job_dpath: Path) -> Path | None:
    bo = job_dpath / 'benchmark_output'
    if not bo.exists():
        return None
    try:
        outputs = HelmOutputs.coerce(bo)
    except Exception:
        return None
    runs = []
    for suite in outputs.suites(pattern='*'):
        runs.extend(list(suite.runs()))
    if not runs:
        return None
    return Path(runs[0].path)


def _process_context_info(process_context: dict[str, Any], fallback_host: str | None) -> dict[str, Any]:
    props = process_context.get('properties', {}) if isinstance(process_context, dict) else {}
    machine = props.get('machine', {}) if isinstance(props.get('machine', {}), dict) else {}
    extra = props.get('extra', {}) if isinstance(props.get('extra', {}), dict) else {}
    env = extra.get('env', {}) if isinstance(extra.get('env', {}), dict) else {}
    nvidia_smi = extra.get('nvidia_smi', {}) if isinstance(extra.get('nvidia_smi', {}), dict) else {}
    gpus = nvidia_smi.get('gpus', []) if isinstance(nvidia_smi.get('gpus', []), list) else []

    host = machine.get('host')
    provenance = 'recorded'
    if not host:
        host = fallback_host
        provenance = 'fallback' if fallback_host else 'unknown'

    return {
        'machine_host': host,
        'machine_user': machine.get('user'),
        'machine_os': machine.get('os_name'),
        'machine_arch': machine.get('arch'),
        'python_version': machine.get('py_version'),
        'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES'),
        'gpu_count': len(gpus),
        'gpu_names': [g.get('name') for g in gpus if isinstance(g, dict)],
        'gpu_memory_total_mb': [g.get('memory_total_mb') for g in gpus if isinstance(g, dict)],
        'provenance_source': provenance,
    }


def _row_for_job(job_config_fpath: Path, fallback_host: str | None) -> dict[str, Any]:
    job_dpath = job_config_fpath.parent
    adapter_manifest = _safe_json_load(job_dpath / 'adapter_manifest.json')
    process_context = _safe_json_load(job_dpath / 'process_context.json')
    if not process_context:
        process_context = adapter_manifest.get('process_context', {}) if isinstance(adapter_manifest, dict) else {}
    run_dir = _first_run_dir(job_dpath)
    run_spec = _safe_json_load(run_dir / 'run_spec.json') if run_dir else {}

    job_config = _safe_json_load(job_config_fpath)
    run_entry = job_config.get('helm.run_entry')
    benchmark = None
    model = None
    method = None
    if run_entry:
        try:
            benchmark, tokens = parse_run_entry_description(run_entry)
            model = tokens.get('model')
            method = tokens.get('method')
        except Exception:
            benchmark = None

    context_info = _process_context_info(process_context, fallback_host)
    adapter_spec = run_spec.get('adapter_spec', {}) if isinstance(run_spec, dict) else {}
    metric_specs = run_spec.get('metric_specs', []) if isinstance(run_spec, dict) else []

    row = {
        'experiment_name': job_dpath.parent.parent.name if job_dpath.parent.name == 'helm' else job_dpath.parent.name,
        'job_id': job_dpath.name,
        'job_dpath': str(job_dpath),
        'status': adapter_manifest.get('status'),
        'manifest_timestamp': adapter_manifest.get('timestamp'),
        'run_entry': run_entry,
        'benchmark': benchmark,
        'model': model,
        'method': method,
        'suite': job_config.get('helm.suite'),
        'max_eval_instances': job_config.get('helm.max_eval_instances'),
        'run_dir': str(run_dir) if run_dir else None,
        'has_run_dir': bool(run_dir and run_dir.exists()),
        'has_run_spec': bool(run_dir and (run_dir / 'run_spec.json').exists()),
        'has_stats': bool(run_dir and (run_dir / 'stats.json').exists()),
        'has_per_instance_stats': bool(run_dir and (run_dir / 'per_instance_stats.json').exists()),
        'model_deployment': adapter_spec.get('model_deployment'),
        'metric_class_names': [m.get('class_name') for m in metric_specs if isinstance(m, dict)],
    }
    row.update(context_info)
    return row


def _write_summary(rows: list[dict[str, Any]], out_fpath: Path) -> None:
    benchmark_counts = Counter(row.get('benchmark') or 'unknown' for row in rows)
    model_counts = Counter(row.get('model') or 'unknown' for row in rows)
    host_counts = Counter(row.get('machine_host') or 'unknown' for row in rows)
    status_counts = Counter(row.get('status') or 'unknown' for row in rows)

    lines = []
    lines.append('Audit Results Index Summary')
    lines.append('')
    lines.append(f'n_rows: {len(rows)}')
    lines.append('')
    lines.append('status_counts:')
    for key, val in sorted(status_counts.items()):
        lines.append(f'  {key}: {val}')
    lines.append('')
    lines.append('machine_host_counts:')
    for key, val in sorted(host_counts.items()):
        lines.append(f'  {key}: {val}')
    lines.append('')
    lines.append('benchmark_counts:')
    for key, val in sorted(benchmark_counts.items()):
        lines.append(f'  {key}: {val}')
    lines.append('')
    lines.append('model_counts:')
    for key, val in sorted(model_counts.items()):
        lines.append(f'  {key}: {val}')
    out_fpath.write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', default=env_defaults()['AUDIT_RESULTS_ROOT'])
    parser.add_argument('--report-dpath', default=str(default_report_root() / 'indexes'))
    parser.add_argument('--fallback-host', default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    report_dpath = Path(args.report_dpath).expanduser().resolve()
    report_dpath.mkdir(parents=True, exist_ok=True)
    stamp = datetime_mod.datetime.now(datetime_mod.UTC).strftime('%Y%m%dT%H%M%SZ')

    rows = []
    for job_config_fpath in sorted(results_root.rglob('job_config.json')):
        try:
            rows.append(_row_for_job(job_config_fpath, args.fallback_host))
        except Exception as ex:
            rows.append({
                'job_dpath': str(job_config_fpath.parent),
                'status': 'index_error',
                'error': repr(ex),
                'machine_host': args.fallback_host,
                'provenance_source': 'fallback' if args.fallback_host else 'unknown',
            })

    jsonl_fpath = report_dpath / f'audit_results_index_{stamp}.jsonl'
    csv_fpath = report_dpath / f'audit_results_index_{stamp}.csv'
    summary_fpath = report_dpath / f'audit_results_index_{stamp}.txt'

    with jsonl_fpath.open('w') as file:
        for row in rows:
            file.write(json.dumps(kwutil.Json.ensure_serializable(row)) + '\n')

    table = pd.DataFrame(rows)
    if not table.empty:
        preferred = [
            'experiment_name', 'job_id', 'status', 'benchmark', 'model', 'method',
            'max_eval_instances', 'machine_host', 'gpu_count', 'gpu_names',
            'cuda_visible_devices', 'provenance_source', 'run_dir',
        ]
        cols = [c for c in preferred if c in table.columns] + [c for c in table.columns if c not in preferred]
        table = table[cols]
    table.to_csv(csv_fpath, index=False)
    _write_summary(rows, summary_fpath)

    print(f'Wrote jsonl index: {jsonl_fpath}')
    print(f'Wrote csv index: {csv_fpath}')
    print(f'Wrote summary: {summary_fpath}')


if __name__ == '__main__':
    main()
