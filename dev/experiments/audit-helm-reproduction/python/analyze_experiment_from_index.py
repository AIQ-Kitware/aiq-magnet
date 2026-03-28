from __future__ import annotations

import argparse
import datetime as datetime_mod
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from aggregate_core_reports import _find_curve_value, _find_pair, _write_latest_alias
from common import audit_root, default_report_root, env_defaults
from rebuild_core_report_from_index import latest_index_csv, load_rows, slugify


def _report_dir_for_run_entry(run_entry: str) -> Path:
    return default_report_root() / f'core-metrics-{slugify(run_entry)}'


def _load_json(fpath: Path) -> dict[str, Any]:
    return json.loads(fpath.read_text())


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

    rebuild_script = audit_root() / 'python' / 'rebuild_core_report_from_index.py'
    built_report_paths = []
    for run_entry in run_entries:
        cmd = [
            env_defaults()['AIQ_PYTHON'],
            str(rebuild_script),
            '--run-entry', str(run_entry),
            '--index-fpath', str(index_fpath),
        ]
        if args.allow_single_repeat:
            cmd.append('--allow-single-repeat')
        subprocess.run(cmd, check=True)
        built_report_paths.append(_report_dir_for_run_entry(run_entry) / 'core_metric_report.latest.json')

    summary_rows = []
    for report_json in built_report_paths:
        if not report_json.exists():
            continue
        report = _load_json(report_json)
        repeat = _find_pair(report, 'kwdagger_repeat')
        official = _find_pair(report, 'official_vs_kwdagger')
        summary_rows.append({
            'experiment_name': args.experiment_name,
            'run_spec_name': report.get('run_spec_name'),
            'report_dir': str(report_json.parent),
            'generated_utc': report.get('generated_utc'),
            'repeat_instance_agree_0': _find_curve_value(repeat.get('instance_level', {}).get('agreement_vs_abs_tol', []), 0.0),
            'official_instance_agree_0': _find_curve_value(official.get('instance_level', {}).get('agreement_vs_abs_tol', []), 0.0),
            'official_instance_agree_01': _find_curve_value(official.get('instance_level', {}).get('agreement_vs_abs_tol', []), 0.1),
            'official_instance_agree_025': _find_curve_value(official.get('instance_level', {}).get('agreement_vs_abs_tol', []), 0.25),
            'official_instance_agree_05': _find_curve_value(official.get('instance_level', {}).get('agreement_vs_abs_tol', []), 0.5),
            'official_runlevel_p90': (((official.get('run_level') or {}).get('overall_quantiles') or {}).get('abs_delta') or {}).get('p90'),
            'official_runlevel_max': (((official.get('run_level') or {}).get('overall_quantiles') or {}).get('abs_delta') or {}).get('max'),
        })

    out_dpath = default_report_root() / f'experiment-analysis-{slugify(args.experiment_name)}'
    out_dpath.mkdir(parents=True, exist_ok=True)
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
        'run_entries': run_entries,
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
    lines.append('')
    lines.append('run_entries:')
    for run_entry in run_entries:
        lines.append(f'  - {run_entry}')
    lines.append('')
    lines.append('per_run_spec:')
    for row in summary_rows:
        lines.append(f"  - run_spec_name: {row['run_spec_name']}")
        lines.append(f"    report_dir: {row['report_dir']}")
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
