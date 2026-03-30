from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from pathlib import Path

from common import audit_root, default_report_root, env_defaults
from rebuild_core_report_from_index import latest_index_csv, load_rows, matching_rows
from compare_batch import collect_historic_candidates, choose_historic_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-fpath', default=None)
    parser.add_argument('--index-dpath', default=str(default_report_root() / 'indexes'))
    parser.add_argument('--report-root', default=str(default_report_root()))
    parser.add_argument('--allow-single-repeat', action='store_true')
    args = parser.parse_args()

    index_fpath = Path(args.index_fpath) if args.index_fpath else latest_index_csv(Path(args.index_dpath))
    rows = load_rows(index_fpath)
    run_entries = sorted({row.get('run_entry') for row in rows if row.get('run_entry')})
    counts = Counter(row.get('run_entry') for row in rows if row.get('run_entry'))

    script = audit_root() / 'python' / 'rebuild_core_report_from_index.py'
    built = 0
    skipped = []
    for run_entry in run_entries:
        matches = matching_rows(rows, run_entry)
        n = len(matches)
        if n == 0:
            skipped.append((run_entry, n, 'no_indexed_runs'))
            continue
        if n < 2 and not args.allow_single_repeat:
            skipped.append((run_entry, n, 'not_enough_matching_runs'))
            continue
        desired_max = None
        try:
            desired_max = int(matches[0].get('max_eval_instances')) if matches and matches[0].get('max_eval_instances') else None
        except Exception:
            desired_max = None
        historic_candidates = collect_historic_candidates(env_defaults()['HELM_PRECOMPUTED_ROOT'], run_entry)
        chosen_historic, _info = choose_historic_candidate(historic_candidates, desired_max)
        if chosen_historic is None:
            skipped.append((run_entry, n, 'no_historic_match'))
            continue
        cmd = [
            env_defaults()['AIQ_PYTHON'],
            str(script),
            '--run-entry', str(run_entry),
            '--index-fpath', str(index_fpath),
        ]
        if args.allow_single_repeat:
            cmd.append('--allow-single-repeat')
        subprocess.run(cmd, check=True)
        built += 1

    print(f'index_fpath={index_fpath}')
    print(f'n_unique_run_entries={len(run_entries)}')
    print(f'n_reports_built={built}')
    print(f'n_skipped={len(skipped)}')
    for run_entry, n, reason in skipped:
        print(f'skipped run_entry={run_entry!r} matching_runs={n} reason={reason}')


if __name__ == '__main__':
    main()
