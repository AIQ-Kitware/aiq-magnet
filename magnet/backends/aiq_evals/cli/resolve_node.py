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

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import kwconf

RESOLUTION_SCHEMA = 'magnet-aiq-evals-resolution/1'


class ResolveNodeCLI(kwconf.Config):
    """Resolve an EvaluationNode's identities where the node runs."""

    __prog__ = 'python -m magnet.backends.aiq_evals.cli.resolve_node'

    request = kwconf.Value(None, required=True, parser=str, help='EvaluationRequest as JSON')
    worker_python = kwconf.Value(None, parser=str, help='engine worker interpreter')
    import_source = kwconf.Value(None, parser=str, help='native artifacts the node imports')
    allow_external_symlinks = kwconf.Value(False, isflag=True, help='follow links out of the import source')


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
        # Identity never needs a secret's value, and a key may only exist inside
        # the node's later endpoint lease; execution still checks it.
        resolved = asyncio.run(resolve_evaluation_async(request, context, require_secrets=False))
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
    args = ResolveNodeCLI.cli(argv=True if argv is None else argv, strict=True, special_options=False)
    try:
        payload = resolve(
            json.loads(args.request),
            args.worker_python or None,
            args.import_source or None,
            bool(args.allow_external_symlinks),
        )
    except Exception as ex:  # reported to the scheduling process, which fails loudly
        print(f'{type(ex).__name__}: {ex}', file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
