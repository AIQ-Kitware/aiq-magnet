"""kwdagger node command: obtain one evaluation through ``aiq_evals.ensure``.

Usage (rendered by :class:`magnet.backends.aiq_evals.EvaluationNode`)::

    python -m magnet.backends.aiq_evals.cli.run_node \\
        --request '<EvaluationRequest JSON>' --store_dpath DIR --out_dpath . \\
        [--select '<selector JSON>'] [--coverage_policy complete|any] \\
        [--worker_python PY] [--timeout_seconds S] [--import_source PATH] \\
        [--measurement_identity DIGEST]

Writes ``evaluation.json`` (the node's primary output) only when the native run
succeeded, so kwdagger never mistakes a failed attempt for a finished node and
reruns it next time. Failed attempts are still recorded in the aiq-evals store
and summarized in ``attempt_summary.json``. The exit status is 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVALUATION_SCHEMA = 'magnet-aiq-evals-node/1'


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m magnet.backends.aiq_evals.cli.run_node')
    parser.add_argument('--request', required=True, help='EvaluationRequest as JSON')
    parser.add_argument('--store_dpath', required=True)
    parser.add_argument('--out_dpath', default='.')
    parser.add_argument('--evaluation_fname', default='evaluation.json')
    parser.add_argument('--select', default=None, help='evidence selector JSON')
    parser.add_argument('--coverage_policy', default='complete')
    parser.add_argument('--worker_python', default=None)
    parser.add_argument('--timeout_seconds', type=float, default=None)
    parser.add_argument('--import_source', default=None)
    parser.add_argument('--allow_external_symlinks', default='False')
    parser.add_argument('--measurement_identity', default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from aiq_evals import EvaluationRequest, ensure_evaluation

    from magnet.backends.aiq_evals.projection import evidence_view

    request = EvaluationRequest.from_dict(json.loads(args.request))
    out_dpath = Path(args.out_dpath)
    out_dpath.mkdir(parents=True, exist_ok=True)
    outcome = ensure_evaluation(
        request,
        args.store_dpath,
        worker_python=args.worker_python,
        timeout_seconds=args.timeout_seconds,
        import_source=args.import_source,
        allow_external_symlinks=str(args.allow_external_symlinks).lower() in {'1', 'true', 'yes'},
    )
    identity = outcome.resolved.identity
    summary = {
        'schema': EVALUATION_SCHEMA,
        'action': outcome.action,
        'reuse_reason': outcome.reuse_reason,
        'run_path': str(outcome.run.path),
        'attempt_path': None if outcome.attempt is None else str(outcome.attempt.path),
        'status': outcome.run.result.status,
        'measurement_identity': identity.to_dict(),
        'preflight_identity': args.measurement_identity,
        'request': request.to_dict(),
    }
    preflight = args.measurement_identity
    if preflight and not preflight.startswith('unresolved') and preflight != identity.digest:
        # The node's kwdagger identity was computed from a resolution that no
        # longer holds (task code, engine, or adapter changed in between).
        summary['error'] = (
            f'measurement identity changed since scheduling: preflight {preflight}, '
            f'now {identity.digest}; reschedule'
        )
        (out_dpath / 'attempt_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(summary['error'], file=sys.stderr)
        return 3
    if outcome.run.result.status != 'succeeded':
        (out_dpath / 'attempt_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'aiq-evals run {outcome.run.result.status}: {outcome.run.path}', file=sys.stderr)
        return 2
    select = json.loads(args.select) if args.select else None
    summary['evidence'] = evidence_view(outcome.run, select, args.coverage_policy).to_dict()
    fpath = out_dpath / args.evaluation_fname
    tmp = fpath.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    tmp.replace(fpath)  # the primary output appears atomically
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
