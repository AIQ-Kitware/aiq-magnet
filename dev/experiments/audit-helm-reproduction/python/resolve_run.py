from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import kwutil

from magnet.backends.helm.cli.materialize_helm_run import (
    discover_benchmark_output_dirs,
    run_dir_matches_requested,
)
from magnet.backends.helm.helm_outputs import HelmOutputs


def load_json(fpath: Path) -> dict[str, Any]:
    return json.loads(fpath.read_text())


def summarize_run_artifacts(run_dir: Path | None) -> dict[str, Any]:
    required_files = [
        'run_spec.json',
        'scenario_state.json',
        'stats.json',
        'per_instance_stats.json',
    ]
    if run_dir is None:
        return {
            'artifact_status': 'missing_run_dir',
            'missing_files': required_files,
        }
    missing_files = [name for name in required_files if not (run_dir / name).exists()]
    return {
        'artifact_status': 'ready' if not missing_files else 'incomplete_run_dir',
        'missing_files': missing_files,
    }


def resolve_kwdagger_run(results_dpath: Path, run_entry: str) -> dict[str, Any] | None:
    helm_root = results_dpath / 'helm'
    if not helm_root.exists():
        return None
    matches = []
    for job_cfg in helm_root.glob('*/job_config.json'):
        data = load_json(job_cfg)
        if data.get('helm.run_entry') != run_entry:
            continue
        job_dpath = job_cfg.parent
        run_dirs = []
        try:
            suites = HelmOutputs.coerce(job_dpath / 'benchmark_output').suites()
            for suite in suites:
                for run in suite.runs():
                    run_dirs.append(Path(run.path))
        except Exception:
            run_dirs = []
        if len(run_dirs) == 1:
            run_dir = run_dirs[0]
        else:
            run_dir = None
            for candidate in run_dirs:
                if run_dir_matches_requested(candidate.name, run_entry):
                    run_dir = candidate
                    break
        match = {
            'job_id': job_dpath.name,
            'job_dpath': str(job_dpath),
            'run_dir': None if run_dir is None else str(run_dir),
            'run_entry': run_entry,
        }
        match.update(summarize_run_artifacts(run_dir))
        matches.append(match)
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f'Found multiple kwdagger matches for {run_entry}: {kwutil.Json.dumps(matches, indent=2)}'
        )
    return matches[0]


def resolve_historic_run(precomputed_root: Path, run_entry: str) -> dict[str, Any] | None:
    matches = []
    for bo in discover_benchmark_output_dirs([precomputed_root]):
        try:
            outputs = HelmOutputs.coerce(bo)
        except Exception:
            continue
        for suite in outputs.suites():
            for run in suite.runs():
                run_dir = Path(run.path)
                if not run_dir_matches_requested(run.name, run_entry):
                    continue
                match = {
                    'run_dir': str(run_dir),
                    'suite': suite.path.name if hasattr(suite, 'path') else None,
                    'helm_version': run_dir.parent.name,
                    'run_entry': run_entry,
                }
                match.update(summarize_run_artifacts(run_dir))
                matches.append(match)
    if not matches:
        return None
    return {
        'matches': matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['kwdg', 'historic'])
    parser.add_argument('--run-entry', required=True)
    parser.add_argument('--results-dpath', default=None)
    parser.add_argument('--precomputed-root', default='/data/crfm-helm-public')
    args = parser.parse_args()

    if args.mode == 'kwdg':
        if args.results_dpath is None:
            raise SystemExit('--results-dpath is required in kwdg mode')
        info = resolve_kwdagger_run(
            results_dpath=Path(args.results_dpath).expanduser().resolve(),
            run_entry=args.run_entry,
        )
    else:
        info = resolve_historic_run(
            precomputed_root=Path(args.precomputed_root).expanduser().resolve(),
            run_entry=args.run_entry,
        )
    print(kwutil.Json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
