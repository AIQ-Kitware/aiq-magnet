from __future__ import annotations

import argparse
import csv
import datetime as datetime_mod
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from common import audit_root, default_report_root, env_defaults
from compare_batch import collect_historic_candidates, choose_historic_candidate


def latest_index_csv(index_dpath: Path) -> Path:
    cands = sorted(index_dpath.glob('audit_results_index_*.csv'), reverse=True)
    if not cands:
        raise FileNotFoundError(f'No index csv files found in {index_dpath}')
    return cands[0]


def load_rows(index_fpath: Path) -> list[dict[str, Any]]:
    with index_fpath.open() as file:
        return list(csv.DictReader(file))


def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return float('-inf')


def matching_rows(
    rows: list[dict[str, Any]],
    run_entry: str,
    experiment_name: str | None = None,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get('run_entry') != run_entry:
            continue
        if experiment_name is not None and row.get('experiment_name') != experiment_name:
            continue
        if row.get('status') not in {'computed', 'reused', 'unknown', ''}:
            continue
        if row.get('has_run_spec', '').lower() not in {'true', '1'}:
            continue
        run_dir = row.get('run_dir')
        if not run_dir:
            continue
        out.append(row)
    out.sort(key=lambda r: (_coerce_float(r.get('manifest_timestamp')), r.get('experiment_name', '')), reverse=True)
    return out


def slugify(text: str) -> str:
    return (
        text.replace('/', '-')
        .replace(':', '-')
        .replace(',', '-')
        .replace('=', '-')
        .replace('@', '-')
    )


def _write_latest_selection(report_dpath: Path, selection: dict[str, Any]) -> Path:
    stamp = datetime_mod.datetime.now(datetime_mod.UTC).strftime('%Y%m%dT%H%M%SZ')
    history_dpath = report_dpath / '.history' / stamp[:8]
    history_dpath.mkdir(parents=True, exist_ok=True)
    history_fpath = history_dpath / f'report_selection_{stamp}.json'
    history_fpath.write_text(json.dumps(selection, indent=2))
    latest_fpath = report_dpath / 'report_selection.latest.json'
    if latest_fpath.exists() or latest_fpath.is_symlink():
        latest_fpath.unlink()
    rel_src = os.path.relpath(history_fpath, start=report_dpath)
    os.symlink(rel_src, latest_fpath)
    return latest_fpath


def _safe_unlink(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def _symlink(target: str | os.PathLike[str], link_path: Path) -> Path:
    target = Path(target).expanduser().resolve()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(link_path)
    rel_src = os.path.relpath(target, start=link_path.parent)
    os.symlink(rel_src, link_path)
    return link_path


def _find_kwdagger_job_dpath(run_dpath: str | os.PathLike[str]) -> Path | None:
    current = Path(run_dpath).expanduser().resolve()
    for cand in [current, *current.parents]:
        if (cand / 'job_config.json').exists():
            return cand
    return None


def _write_selected_run_symlinks(report_dpath: Path, selection: dict[str, Any]) -> dict[str, str]:
    created = {}
    run_targets = {
        'kwdagger_a.run': selection['left_run_a'],
        'kwdagger_b.run': selection['left_run_b'],
        'official.run': selection['right_run_a'],
    }
    for name, target in run_targets.items():
        created[name] = str(_symlink(target, report_dpath / name))

    job_targets = {
        'kwdagger_a.job': _find_kwdagger_job_dpath(selection['left_run_a']),
        'kwdagger_b.job': _find_kwdagger_job_dpath(selection['left_run_b']),
    }
    for name, target in job_targets.items():
        if target is not None:
            created[name] = str(_symlink(target, report_dpath / name))
    return created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-entry', required=True)
    parser.add_argument('--index-fpath', default=None)
    parser.add_argument('--index-dpath', default=str(default_report_root() / 'indexes'))
    parser.add_argument('--precomputed-root', default=env_defaults()['HELM_PRECOMPUTED_ROOT'])
    parser.add_argument('--report-dpath', default=None)
    parser.add_argument('--left-label', default='kwdagger_repeat')
    parser.add_argument('--right-label', default='official_vs_kwdagger')
    parser.add_argument('--allow-single-repeat', action='store_true')
    parser.add_argument('--experiment-name', default=None)
    args = parser.parse_args()

    index_fpath = (
        Path(args.index_fpath).expanduser().resolve()
        if args.index_fpath else
        latest_index_csv(Path(args.index_dpath).expanduser().resolve())
    )
    rows = load_rows(index_fpath)
    matches = matching_rows(rows, args.run_entry, experiment_name=args.experiment_name)
    if not matches:
        if args.experiment_name is not None:
            raise SystemExit(
                f'No indexed kwdagger runs found for run_entry={args.run_entry!r} '
                f'within experiment_name={args.experiment_name!r}'
            )
        raise SystemExit(f'No indexed kwdagger runs found for run_entry={args.run_entry!r}')

    if len(matches) >= 2:
        left_a = matches[0]['run_dir']
        left_b = matches[1]['run_dir']
    elif args.allow_single_repeat:
        left_a = matches[0]['run_dir']
        left_b = matches[0]['run_dir']
    else:
        raise SystemExit(
            f'Need at least 2 matching kwdagger runs for run_entry={args.run_entry!r}; '
            f'found {len(matches)}. Use --allow-single-repeat to duplicate the latest run.'
        )

    desired_max = None
    try:
        desired_max = int(matches[0].get('max_eval_instances')) if matches[0].get('max_eval_instances') else None
    except Exception:
        desired_max = None

    historic_candidates = collect_historic_candidates(args.precomputed_root, args.run_entry)
    chosen_historic, info = choose_historic_candidate(historic_candidates, desired_max)
    if chosen_historic is None:
        raise SystemExit(f'No historic HELM candidate found for run_entry={args.run_entry!r}')

    report_dpath = Path(args.report_dpath) if args.report_dpath else (
        default_report_root() / f'core-metrics-{slugify(args.run_entry)}'
    )
    report_dpath = report_dpath.expanduser().resolve()
    report_dpath.mkdir(parents=True, exist_ok=True)

    script = audit_root() / 'python' / 'core_metric_report.py'
    cmd = [
        env_defaults()['AIQ_PYTHON'],
        str(script),
        '--left-run-a', str(left_a),
        '--left-run-b', str(left_b),
        '--left-label', args.left_label,
        '--right-run-a', str(chosen_historic['run_dir']),
        '--right-run-b', str(left_a),
        '--right-label', args.right_label,
        '--report-dpath', str(report_dpath),
    ]
    print(f'index_fpath={index_fpath}')
    print(f'left_run_a={left_a}')
    print(f'left_run_b={left_b}')
    print(f'right_run_a={chosen_historic["run_dir"]}')
    print(f'report_dpath={report_dpath}')
    print(f'historic_info={info}')
    selection = {
        'index_fpath': str(index_fpath),
        'run_entry': args.run_entry,
        'left_run_a': str(left_a),
        'left_run_b': str(left_b),
        'right_run_a': str(chosen_historic['run_dir']),
        'left_label': args.left_label,
        'right_label': args.right_label,
        'report_dpath': str(report_dpath),
        'experiment_name': args.experiment_name,
        'historic_info': info,
    }
    selection_fpath = _write_latest_selection(report_dpath, selection)
    link_info = _write_selected_run_symlinks(report_dpath, selection)
    selection['selected_run_links'] = link_info
    selection_fpath = _write_latest_selection(report_dpath, selection)
    print(f'selection_fpath={selection_fpath}')
    subprocess.run(cmd, check=True)

    inspect_script = audit_root() / 'python' / 'inspect_pair_samples.py'
    sample_jobs = [
        [
            env_defaults()['AIQ_PYTHON'],
            str(inspect_script),
            '--run-a', str(left_a),
            '--run-b', str(left_b),
            '--label', args.left_label,
            '--report-dpath', str(report_dpath),
        ],
        [
            env_defaults()['AIQ_PYTHON'],
            str(inspect_script),
            '--run-a', str(chosen_historic['run_dir']),
            '--run-b', str(left_a),
            '--label', args.right_label,
            '--report-dpath', str(report_dpath),
        ],
    ]
    for inspect_cmd in sample_jobs:
        subprocess.run(inspect_cmd, check=True)


if __name__ == '__main__':
    main()
