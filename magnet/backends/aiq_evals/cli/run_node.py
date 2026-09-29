"""kwdagger node command: obtain one evaluation through ``magnet_evals.ensure``.

Usage (rendered by :class:`magnet.backends.aiq_evals.EvaluationNode`)::

    python -m magnet.backends.aiq_evals.cli.run_node \\
        --request '<EvaluationRequest JSON>' --store_dpath DIR --out_dpath . \\
        [--select '<selector JSON>'] [--coverage_policy complete|any] \\
        [--worker_python PY] [--timeout_seconds S] [--import_source PATH] \\
        [--measurement_identity DIGEST] [--import_identity DIGEST]

Writes ``evaluation.json`` (the node's primary output) only when the native run
succeeded, so kwdagger never mistakes a failed attempt for a finished node and
reruns it next time. Failed attempts are still recorded in the aiq-magnet-evals
store and summarized in ``attempt_summary.json``; the exit status is 2.

``evaluation.json`` references the run (path and identities) and records how
to project it (selector, coverage policy). It stores no claim-facing values:
loaders recompute the evidence from the validated run (plan M4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
    parser.add_argument('--import_identity', default=None)
    parser.add_argument('--endpoint', default=None, help='leased infer-stack alias (primary model)')
    parser.add_argument('--endpoints', default=None, help='leased aliases by model role, JSON {role: alias}')
    parser.add_argument('--leased_command', default=None,
                        help='gate mode: run this leased child only if the store lacks the run')
    parser.add_argument('--lock_held', default='False',
                        help='the scheduling gate holds the acquisition lock for this node')
    return parser


def lease_runtime(endpoints: dict | str | None, request: dict, environ=None) -> tuple[dict, dict]:
    """Operational bindings for a node running inside ``infer-stack run`` (M8).

    ``endpoints`` maps model roles to leased infer-stack aliases (a bare string
    means the primary role). Returns ``(model_endpoints, env)``: the lease's
    base URL for each leased role and the lease's API key as a runtime secret.
    Neither is hashed or persisted. One lease serves every alias behind one
    OpenAI-compatible base URL; infer-stack exports each alias's served model
    name as ``INFER_STACK_ENDPOINT_<SLUG>``. The request must bind each leased
    role to that name, because silently changing the model would change what
    is measured.
    """
    import os

    from magnet.backends.aiq_evals.pipeline import lease_endpoint_var

    if isinstance(endpoints, str):
        endpoints = {'primary': endpoints}
    endpoints = {str(role): str(alias) for role, alias in dict(endpoints or {}).items() if alias}
    if not endpoints:
        return {}, {}
    environ = os.environ if environ is None else environ
    base_url = environ.get('OPENAI_BASE_URL')
    if not base_url:
        raise SystemExit(f'endpoints {endpoints} are set but no lease is active (OPENAI_BASE_URL unset)')
    bindings = {m.get('role', 'primary'): m for m in request['models']}
    for role, alias in endpoints.items():
        if role not in bindings:
            raise SystemExit(f'leased endpoint {alias!r} is for role {role!r}, which the request does not bind')
        served = environ.get(lease_endpoint_var(alias))
        if served is None:
            # A lease always exports it; missing means it was not forwarded
            # (e.g. into a container), and then nothing can be verified.
            raise SystemExit(
                f'the lease exports no {lease_endpoint_var(alias)} for endpoint {alias!r}; '
                'cannot verify which model it serves'
            )
        if bindings[role]['model'] != served:
            raise SystemExit(
                f'leased endpoint {alias!r} serves {served!r} but the request binds role '
                f'{role!r} to {bindings[role]["model"]!r}; set the model binding to the served name'
            )
    env = {'OPENAI_API_KEY': environ['OPENAI_API_KEY']} if environ.get('OPENAI_API_KEY') else {}
    return {role: base_url for role in endpoints}, env


def _truthy(value) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def _write_evaluation(args, out_dpath: Path, summary: dict) -> None:
    fpath = Path(args.evaluation_fname)
    if fpath.parent == Path('.'):
        # A bare name lives in the node directory; kwdagger passes a full path.
        fpath = out_dpath / fpath
    tmp = fpath.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    tmp.replace(fpath)  # the primary output appears atomically


def gate(args, out_dpath: Path) -> int:
    """Decide under the store's acquisition lock whether a leased run is needed.

    Runs on the scheduling host. With a reusable scheduled identity it takes
    the acquisition lock, and if a valid canonical run of the scheduled request
    exists it records reuse without leasing anything. Otherwise it runs the
    leased child (``--leased_command``; the child's ``ensure`` knows the lock
    is held) and keeps the lock until the child exits, so concurrent nodes for
    one measurement start one lease. A non-reusable identity always runs the
    child. SIGTERM reaches the child's process group (SIGTERM, then SIGKILL).
    """
    import asyncio
    import os
    import signal
    import subprocess

    from magnet_evals import EvaluationRequest, load_run
    from magnet_evals.store import ResultStore

    from magnet.backends.aiq_evals.pipeline import EVALUATION_SCHEMA
    from magnet.backends.aiq_evals.projection import normalize_selector

    request = EvaluationRequest.from_dict(json.loads(args.request))
    digest = args.measurement_identity or ''
    store = ResultStore(args.store_dpath)

    def stored_run():
        try:
            run = load_run(store.run_path(digest))
        except Exception:
            return None
        ok = (run.complete and run.result.status == 'succeeded' and run.manifest.get('reusable')
              and run.resolved.identity.digest == digest
              and run.resolved.request.to_dict() == request.to_dict())
        return run if ok else None

    def run_child() -> int:
        proc = subprocess.Popen(['bash', '-c', args.leased_command], start_new_session=(os.name == 'posix'))
        try:
            return proc.wait()
        except BaseException:
            for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, None)):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
            raise

    if len(digest) != 64:
        return run_child()

    async def acquire() -> int:
        async with store.acquisition_lock(digest) as lock:
            run = stored_run()
            if run is None:
                return await asyncio.to_thread(run_child)
            _write_evaluation(args, out_dpath, {
                'schema': EVALUATION_SCHEMA,
                'action': 'reused',
                'reuse_reason': 'validated canonical run (lease gate)',
                'waited': lock.waited,
                'run_path': str(run.path),
                'attempt_path': None,
                'status': run.result.status,
                'measurement_identity': run.resolved.identity.to_dict(),
                'normalized_artifact_identity': run.manifest.get('normalized_artifact_identity'),
                'import_identity': None,
                'preflight_identity': args.measurement_identity,
                'preflight_import_identity': None,
                'select': normalize_selector(json.loads(args.select) if args.select else None),
                'coverage_policy': args.coverage_policy,
                'request': request.to_dict(),
            })
            return 0

    return asyncio.run(acquire())


def _stale_identities(args, digest: str, import_identity: str | None) -> list[str]:
    """Differences between the scheduled (preflight) and current identities."""
    changed = []
    preflight = args.measurement_identity
    if preflight and not preflight.startswith('unresolved') and preflight != digest:
        changed.append(f'measurement identity: preflight {preflight}, now {digest}')
    if args.import_identity and args.import_identity != import_identity:
        changed.append(f'imported content: preflight {args.import_identity}, now {import_identity}')
    return changed


def _reschedule(out_dpath: Path, summary: dict, stale: list[str]) -> int:
    """The node's kwdagger identity no longer describes this computation."""
    summary = dict(summary, error='identity changed since scheduling; reschedule: ' + '; '.join(stale))
    (out_dpath / 'attempt_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(summary['error'], file=sys.stderr)
    return 3


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import asyncio
    import signal

    from magnet_evals import (
        EvaluationRequest,
        ExecutionContext,
        ensure_evaluation,
        native_source_identity,
        resolve_evaluation_async,
    )
    from magnet_evals.errors import ImportIdentityMismatch

    from magnet.backends.aiq_evals.pipeline import EVALUATION_SCHEMA
    from magnet.backends.aiq_evals.projection import normalize_selector

    # kwdagger/tmux stop jobs with SIGTERM; turn it into cancellation so
    # aiq-magnet-evals interrupts and reaps its engine worker group (M8).
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    if args.leased_command:
        out_dpath = Path(args.out_dpath)
        out_dpath.mkdir(parents=True, exist_ok=True)
        return gate(args, out_dpath)
    request_dict = json.loads(args.request)
    select = normalize_selector(json.loads(args.select) if args.select else None)
    model_endpoints, lease_env = lease_runtime(
        json.loads(args.endpoints) if args.endpoints else args.endpoint, request_dict,
    )
    request = EvaluationRequest.from_dict(request_dict)
    out_dpath = Path(args.out_dpath)
    out_dpath.mkdir(parents=True, exist_ok=True)
    allow_external = _truthy(args.allow_external_symlinks)

    # Before any engine work: does the scheduled identity still hold? Task code,
    # the engine, the adapter, or imported files may have changed since
    # preflight, and then this node's kwdagger identity is stale.
    context = ExecutionContext(
        output_dir=Path(args.store_dpath), env=lease_env, worker_python=args.worker_python,
        timeout_seconds=args.timeout_seconds, model_endpoints=model_endpoints,
    )
    resolved = asyncio.run(resolve_evaluation_async(request, context))
    current_import = None if args.import_source is None else native_source_identity(
        args.import_source, allow_external_symlinks=allow_external,
    )
    stale = _stale_identities(args, resolved.identity.digest, current_import)
    if stale:
        return _reschedule(out_dpath, {
            'schema': EVALUATION_SCHEMA, 'status': 'not-run', 'request': request.to_dict(),
            'measurement_identity': resolved.identity.to_dict(), 'import_identity': current_import,
            'preflight_identity': args.measurement_identity, 'preflight_import_identity': args.import_identity,
        }, stale)
    try:
        outcome = ensure_evaluation(
            resolved,
            args.store_dpath,
            worker_python=args.worker_python,
            timeout_seconds=args.timeout_seconds,
            import_source=args.import_source,
            allow_external_symlinks=allow_external,
            model_endpoints=model_endpoints,
            env=lease_env,
            lock_held=_truthy(args.lock_held),
            # Import exactly the content this node was scheduled for, or nothing.
            expected_import_identity=args.import_identity or None,
        )
    except ImportIdentityMismatch as ex:
        return _reschedule(out_dpath, {
            'schema': EVALUATION_SCHEMA, 'status': 'not-run', 'request': request.to_dict(),
            'measurement_identity': resolved.identity.to_dict(), 'import_identity': current_import,
            'preflight_identity': args.measurement_identity, 'preflight_import_identity': args.import_identity,
        }, [str(ex)])
    identity = outcome.resolved.identity
    summary = {
        'schema': EVALUATION_SCHEMA,
        'action': outcome.action,
        'reuse_reason': outcome.reuse_reason,
        'waited': outcome.waited,
        'run_path': str(outcome.run.path),
        'attempt_path': None if outcome.attempt is None else str(outcome.attempt.path),
        'status': outcome.run.result.status,
        'measurement_identity': identity.to_dict(),
        'normalized_artifact_identity': outcome.run.manifest.get('normalized_artifact_identity'),
        'import_identity': outcome.import_identity,
        'preflight_identity': args.measurement_identity,
        'preflight_import_identity': args.import_identity,
        # How MAGNET projects the run; the projected values are never stored.
        'select': select,
        'coverage_policy': args.coverage_policy,
        'request': request.to_dict(),
    }
    # The imported files could still change between hashing and copying.
    stale = _stale_identities(args, identity.digest, outcome.import_identity)
    if stale:
        return _reschedule(out_dpath, summary, stale)
    if outcome.run.result.status != 'succeeded':
        (out_dpath / 'attempt_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'aiq-magnet-evals run {outcome.run.result.status}: {outcome.run.path}', file=sys.stderr)
        return 2
    _write_evaluation(args, out_dpath, summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
