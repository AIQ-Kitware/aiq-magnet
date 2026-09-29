"""kwdagger node that obtains one evaluation through ``aiq_evals`` (plan M2/M3).

``EvaluationNode`` converts recipe parameters into an ``aiq_evals``
``EvaluationRequest`` and runs :mod:`magnet.backends.aiq_evals.cli.run_node`.
It projects the result into exactly one flat kwdagger row. It owns no
engine-specific execution logic: ``engine`` names the evaluation engine
(``helm``, ``olmo_eval``, ``inspect_ai``), while the recipe's ``backend`` stays
the kwdagger/cmd_queue scheduler (``serial``, ``tmux``, ...).

Identity (M3):

* For a real schedule, ``configure`` preflight-resolves the request in the
  engine worker and puts the resolved measurement digest into the node's
  ``algo_params``. kwdagger's node identity therefore changes with task code,
  engine, or adapter changes, not only with the recipe text. A non-reusable
  identity gets a unique nonce, so it is never skipped as done.
* Dry runs never resolve (M9); they only compile the request shape.
* ``does_exist`` accepts the primary output only while the aiq-evals run it
  points at still validates as a successful bundle. A stale marker cannot hide
  a failed, missing, or tampered run.
* Native reuse is ``aiq_evals.ensure`` against a shared store keyed by that
  identity. A changed evidence ``select`` makes a new, cheap node run that
  reuses the unchanged native evaluation.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import shlex
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from magnet.process_node import MagnetProcessNode

REQUEST_KEYS = (
    'engine', 'task', 'task_revision', 'data_revision', 'models',
    'task_options', 'generation', 'engine_options',
)


def missing_request_keys(config: dict[str, Any]) -> list[str]:
    return [key for key in ('engine', 'task', 'models') if config.get(key) in (None, '', [])]


def build_request_dict(config: dict[str, Any]) -> dict[str, Any]:
    """The ``EvaluationRequest`` a node config describes (no engine import)."""
    missing = missing_request_keys(config)
    if missing:
        raise ValueError(f'EvaluationNode needs {missing}')
    models = config['models']
    if isinstance(models, str):
        models = json.loads(models)
    return {
        'schema_version': 1,
        'engine': config['engine'],
        'task': config['task'],
        'task_revision': config.get('task_revision'),
        'data_revision': config.get('data_revision'),
        'models': [
            {
                'role': model.get('role', 'primary'),
                'model': model['model'],
                'provider': model.get('provider'),
                'revision': model.get('revision'),
                'cache_token': model.get('cache_token'),
                'provider_options': dict(model.get('provider_options') or {}),
            }
            for model in models
        ],
        'task_options': dict(config.get('task_options') or {}),
        'generation': dict(config.get('generation') or {}),
        'engine_options': dict(config.get('engine_options') or {}),
    }


_PREFLIGHT: contextvars.ContextVar[bool] = contextvars.ContextVar('magnet_aiq_evals_preflight', default=False)


@contextlib.contextmanager
def preflight_scope(enabled: bool):
    """Resolve EvaluationNode identities while a real schedule compiles (M3/M9).

    Resolutions are memoized only within one scope, so every schedule
    re-resolves (task code may have changed) and draws fresh nonces for
    non-reusable identities. Dry runs pass ``enabled=False``.
    """
    token = _PREFLIGHT.set(enabled)
    cache_clear = getattr(_preflight_digest, 'cache_clear', None)
    if cache_clear is not None:
        cache_clear()
    try:
        yield
    finally:
        _PREFLIGHT.reset(token)


@lru_cache(maxsize=None)
def _preflight_digest(request_json: str, worker_python: str | None, nonce_key: str) -> str:
    # Cached per process: kwdagger may configure the same node more than once
    # while compiling one schedule, and every call must see the same identity.
    from aiq_evals import EvaluationRequest, ExecutionContext, resolve_evaluation_async

    request = EvaluationRequest.from_dict(json.loads(request_json))
    with tempfile.TemporaryDirectory(prefix='magnet-preflight-') as scratch:
        context = ExecutionContext(output_dir=Path(scratch), worker_python=worker_python)
        resolved = asyncio.run(resolve_evaluation_async(request, context))
    if not resolved.identity.reusable:
        return f'unresolved-{uuid.uuid4().hex}'
    return resolved.identity.digest


class EvaluationNode(MagnetProcessNode):
    """Invoke one ``aiq_evals`` evaluation request as a kwdagger node."""

    name = 'evaluate'
    executable = 'python -m magnet.backends.aiq_evals.cli.run_node'

    algo_params = {
        'engine': None,
        'task': None,
        'task_revision': None,
        'data_revision': None,
        'models': None,
        'task_options': None,
        'generation': None,
        'engine_options': None,
        # Native artifacts to import instead of executing (identity-bearing:
        # different content yields a different normalized artifact).
        'import_source': None,
        # MAGNET evidence projection. Changing these reruns this cheap node but
        # reuses the native evaluation from the aiq-evals store.
        'select': None,
        'coverage_policy': 'complete',
        # Filled by preflight resolution; kwdagger hashes it into the node id.
        'measurement_identity': 'unresolved-dry-run',
    }
    perf_params = {
        # Shared content-addressed aiq-evals store; default under the kwdagger root.
        'store_dpath': None,
        # Engine worker interpreter (engines stay out of MAGNET's environment).
        'worker_python': None,
        'timeout_seconds': None,
        'allow_external_symlinks': False,
        # infer-stack catalog alias to lease for the primary model (M8). Operational:
        # the model binding's revision/cache_token is the identity, not the lease.
        'endpoint': None,
    }
    endpoint_params = ('endpoint',)
    in_paths: set[str] = set()
    out_paths = {'out_dpath': '.', 'evaluation_fname': 'evaluation.json'}
    primary_out_key = 'evaluation_fname'

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # A recipe's algo_params/perf_params supply values; they must extend,
        # not replace, the parameters this class declares (e.g. the preflight
        # ``measurement_identity``).
        for key in ('algo_params', 'perf_params'):
            if kwargs.get(key) is not None:
                kwargs[key] = {**getattr(type(self), key), **dict(kwargs[key])}
        super().__init__(*args, **kwargs)

    @property
    def final_algo_config(self) -> Any:
        """Algo params with the preflight-resolved measurement identity (M3).

        kwdagger hashes this mapping into the node id. Inside a real
        ``preflight_scope`` the request is resolved in the engine worker once
        per schedule; outside it (dry runs, result loading) nothing resolves.
        """
        config = super().final_algo_config
        if _PREFLIGHT.get() and not missing_request_keys(dict(config)):
            current = str(config.get('measurement_identity') or '')
            if current.startswith('unresolved') or not current:
                config = type(config)(config)
                request = build_request_dict(dict(config))
                worker = self.config.get('worker_python') or self.perf_params.get('worker_python')
                config['measurement_identity'] = _preflight_digest(
                    json.dumps(request, sort_keys=True), worker, self.name
                )
        return config

    def _store_dpath(self) -> str:
        store = self.final_config.get('store_dpath')
        if store:
            return str(store)
        root = getattr(self, 'root_dpath', None) or '.'
        return str(Path(root) / '_aiq_evals_store')

    @property
    def command(self) -> str:
        config = self.final_config
        if missing_request_keys(config):
            # kwdagger renders unconfigured nodes (e.g. when printing the graph).
            request_text = '<unconfigured: needs engine/task/models>'
        else:
            request_text = json.dumps(build_request_dict(config), sort_keys=True)
        args = {
            'request': request_text,
            'store_dpath': self._store_dpath(),
            'out_dpath': config.get('out_dpath', '.'),
            'evaluation_fname': config.get('evaluation_fname', 'evaluation.json'),
            'coverage_policy': config.get('coverage_policy') or 'complete',
        }
        if config.get('measurement_identity') not in (None, ''):
            args['measurement_identity'] = config['measurement_identity']
        if config.get('select'):
            select = config['select']
            args['select'] = select if isinstance(select, str) else json.dumps(select, sort_keys=True)
        for key in ('worker_python', 'timeout_seconds', 'import_source'):
            if config.get(key) not in (None, ''):
                args[key] = config[key]
        if config.get('allow_external_symlinks'):
            args['allow_external_symlinks'] = 'True'
        if config.get('endpoint'):
            args['endpoint'] = config['endpoint']
        argstr = ' \\\n    '.join(f'--{key}={shlex.quote(str(value))}' for key, value in args.items())
        command = f'{self.executable} \\\n    {argstr}'
        command = self._wrap_interpreter(command)
        if self._native_result_available():
            # The node will only reuse the stored run; leasing would start a
            # model for nothing (M8: avoid duplicate model startup).
            return command
        return self.wrap_with_lease(command)

    def _native_result_available(self) -> bool:
        digest = str(self.final_config.get('measurement_identity') or '')
        if len(digest) != 64:
            return False
        try:
            from aiq_evals import load_run
            from aiq_evals.store import ResultStore

            run = load_run(ResultStore(self._store_dpath()).run_path(digest))
        except Exception:
            return False
        return run.complete and run.result.status == 'succeeded'

    @property
    def does_exist(self) -> bool:
        """Done only if the referenced aiq-evals run still validates as succeeded."""
        paths = self.final_out_paths
        fpath = paths.get(self.primary_out_key) if paths else None
        return fpath is not None and evaluation_is_valid(Path(fpath))

    def test_is_computed_command(self) -> str | None:
        """The job-level "done" guard validates the run, not just the marker file."""
        paths = self.final_out_paths
        fpath = paths.get(self.primary_out_key) if paths else None
        if fpath is None:
            return None
        return self._wrap_interpreter(
            f'python -m magnet.backends.aiq_evals.cli.check_done {shlex.quote(str(fpath))}'
        )

    def _wrap_interpreter(self, command: str) -> str:
        if self.containerization_is_enabled():
            return self.wrap_with_container(command)
        from magnet.containers import host_interpreter

        return host_interpreter(command)

    def load_result(self, node_dpath: Any) -> Any:
        return load_evaluation_row(self, node_dpath)


def evaluation_is_valid(fpath: Path) -> bool:
    if not fpath.is_file():
        return False
    try:
        from aiq_evals import load_run

        summary = json.loads(fpath.read_text())
        run = load_run(summary['run_path'])
    except Exception:
        return False
    return (
        run.complete
        and run.result.status == 'succeeded'
        and run.resolved.identity.digest == summary['measurement_identity']['digest']
    )


def load_evaluation_row(node: Any, node_dpath: Any) -> Any:
    """One flat row: ``metrics.<node>.*`` from the stored evidence view (M5)."""
    from kwdagger.utils import util_dotdict

    from magnet.backends.aiq_evals.projection import flat_metrics

    node_dpath = Path(node_dpath)
    summary = json.loads((node_dpath / node.out_paths[node.primary_out_key]).read_text())
    metrics = flat_metrics(summary['evidence'])
    metrics['action'] = summary['action']
    flat = util_dotdict.DotDict({f'metrics.{key}': value for key, value in metrics.items()})
    return flat.insert_prefix(node.name, index=1)
