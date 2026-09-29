"""Preflight command: resolve an EvaluationNode's identities (plan M3).

Rendered and run by :class:`magnet.backends.aiq_evals.EvaluationNode` while a
schedule compiles, through the same container/host wrapper as the node's real
command. Resolution therefore sees the same engine, task code, worker
interpreter, and import paths that execution will see: a ``worker_python`` or
engine that exists only inside the node's container resolves there too.

Usage::

    python -m magnet.backends.aiq_evals.cli.resolve_node \\
        --request '<EvaluationRequest JSON>' [--worker_python PY] \\
        [--import_source PATH] [--allow_external_symlinks True]

Prints one JSON object on the last line of stdout: ``measurement_identity``
(``digest``, ``reusable``, ``unknown_reasons``) and ``import_identity`` (the
content identity of ``import_source``, or null). Exits non-zero with the
error on stderr when resolution fails.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

RESOLUTION_SCHEMA = 'magnet-aiq-evals-resolution/1'


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m magnet.backends.aiq_evals.cli.resolve_node')
    parser.add_argument('--request', required=True, help='EvaluationRequest as JSON')
    parser.add_argument('--worker_python', default=None)
    parser.add_argument('--import_source', default=None)
    parser.add_argument('--allow_external_symlinks', default='False')
    return parser


def resolve(request_dict: dict, worker_python: str | None, import_source: str | None,
            allow_external_symlinks: bool) -> dict:
    from magnet_evals import (
        EvaluationRequest,
        ExecutionContext,
        native_source_identity,
        resolve_evaluation_async,
    )

    request = EvaluationRequest.from_dict(request_dict)
    with tempfile.TemporaryDirectory(prefix='magnet-preflight-') as scratch:
        context = ExecutionContext(output_dir=Path(scratch), worker_python=worker_python)
        resolved = asyncio.run(resolve_evaluation_async(request, context))
    identity = resolved.identity
    return {
        'schema': RESOLUTION_SCHEMA,
        'measurement_identity': {
            'digest': identity.digest,
            'reusable': bool(identity.reusable),
            'unknown_reasons': list(identity.unknown_reasons),
        },
        'import_identity': (
            None if import_source is None
            else native_source_identity(import_source, allow_external_symlinks=allow_external_symlinks)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = resolve(
            json.loads(args.request),
            args.worker_python or None,
            args.import_source or None,
            str(args.allow_external_symlinks).lower() in {'1', 'true', 'yes'},
        )
    except Exception as ex:  # reported to the scheduling process, which fails loudly
        print(f'{type(ex).__name__}: {ex}', file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
