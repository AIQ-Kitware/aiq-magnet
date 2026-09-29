"""kwdagger node that obtains one evaluation through ``magnet_evals`` (plan M2/M3).

``EvaluationNode`` converts recipe parameters into an ``magnet_evals``
``EvaluationRequest`` and runs :mod:`magnet.backends.aiq_evals.cli.run_node`.
It projects the result into exactly one flat kwdagger row. It owns no
engine-specific execution logic: ``engine`` names the evaluation engine
(``helm``, ``olmo_eval``, ``inspect_ai``), while the recipe's ``backend`` stays
the kwdagger/cmd_queue scheduler (``serial``, ``tmux``, ...).

Identity (M3):

* For a real schedule, the node preflight-resolves its request before kwdagger
  hashes it. Preflight runs :mod:`~magnet.backends.aiq_evals.cli.resolve_node`
  through the *same* container/host wrapper as the node's command, so an
  engine or ``worker_python`` that exists only in the node's container
  resolves exactly as execution will. The resolved measurement digest, and
  for an import the native content identity of ``import_source``, become the
  computed ``measurement_identity`` / ``import_identity`` parameters. kwdagger's
  node identity therefore changes with task code, engine, adapter, or
  imported-file changes, not only with the recipe text.
* Those two parameters are computed, never configured: a recipe that sets them
  is rejected, and every real preflight overwrites whatever value is present.
* A non-reusable identity gets a unique nonce, so it is never skipped as done.
* Dry runs never resolve (M9); they only compile and statically check the
  request shape, evidence selector, and coverage policy.
* ``does_exist`` accepts the primary output only while the run it points at
  still validates as the same successful bundle (measurement, normalized
  artifact, and import identities). A stale marker cannot hide a failed,
  missing, replaced, or tampered run.

Evidence (M4/M5): ``evaluation.json`` records *which* run and *how* to project
it (selector, coverage policy), never the claim-facing values themselves.
Rows are recomputed from the validated run every time they are loaded, so
editing ``evaluation.json`` cannot change what a claim sees.

Native reuse is ``magnet_evals.ensure`` against a shared store. Acquisition is
single-flight in that store, so nodes that differ only in ``select`` (for
example a selector matrix) and run concurrently execute the native evaluation
once; the others wait and reuse it.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Any

from magnet.process_node import MagnetProcessNode

REQUEST_KEYS = (
    'engine', 'task', 'task_revision', 'data_revision', 'models',
    'task_options', 'generation', 'engine_options',
)
#: Parameters filled by preflight. They take part in kwdagger's node identity
#: but may not be configured by a recipe.
COMPUTED_KEYS = ('measurement_identity', 'import_identity')
DRY_RUN_IDENTITY = 'unresolved-dry-run'
EVALUATION_SCHEMA = 'magnet-aiq-evals-node/2'


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


def selector_value(select: Any) -> dict[str, Any] | None:
    """A recipe/matrix ``select`` value (mapping or JSON text) as a mapping."""
    if select in (None, ''):
        return None
    if isinstance(select, str):
        select = json.loads(select)
    if not isinstance(select, dict):
        raise ValueError(f'EvaluationNode select must be a mapping; got {select!r}')
    return select


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes'}


class _PreflightState:
    """Resolutions and nonces for one schedule compilation."""

    def __init__(self) -> None:
        self.resolutions: dict[str, dict[str, Any]] = {}
        self.nonces: dict[str, str] = {}


_PREFLIGHT: contextvars.ContextVar[_PreflightState | None] = contextvars.ContextVar(
    'magnet_aiq_evals_preflight', default=None,
)


@contextlib.contextmanager
def preflight_scope(enabled: bool):
    """Resolve EvaluationNode identities while a real schedule compiles (M3/M9).

    Resolutions are memoized only within one scope: kwdagger may configure the
    same node more than once while compiling a schedule, and every call must
    see the same identity, but each new schedule re-resolves (task code or
    imported files may have changed) and draws fresh nonces for non-reusable
    identities. Dry runs pass ``enabled=False``.
    """
    token = _PREFLIGHT.set(_PreflightState() if enabled else None)
    try:
        yield
    finally:
        _PREFLIGHT.reset(token)


class PreflightError(RuntimeError):
    """Preflight resolution of an EvaluationNode failed."""


def run_preflight(command: str) -> dict[str, Any]:
    """Run a rendered ``resolve_node`` command and parse its resolution."""
    proc = subprocess.run(['bash', '-c', command], capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
        detail = (proc.stderr or proc.stdout).strip()[-4000:]
        raise PreflightError(
            f'EvaluationNode preflight resolution failed (exit {proc.returncode}): {detail}\n'
            f'command: {command}'
        )
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as ex:
        raise PreflightError(f'preflight printed no resolution: {lines[-1]!r}') from ex
    return payload


class EvaluationNode(MagnetProcessNode):
    """Invoke one ``magnet_evals`` evaluation request as a kwdagger node."""

    name = 'evaluate'
    executable = 'python -m magnet.backends.aiq_evals.cli.run_node'
    resolve_executable = 'python -m magnet.backends.aiq_evals.cli.resolve_node'

    algo_params = {
        'engine': None,
        'task': None,
        'task_revision': None,
        'data_revision': None,
        'models': None,
        'task_options': None,
        'generation': None,
        'engine_options': None,
        # Native artifacts to import instead of executing. Their content
        # identity (``import_identity``) is part of the node identity.
        'import_source': None,
        # MAGNET evidence projection. Changing these reruns this cheap node but
        # reuses the native evaluation from the aiq-magnet-evals store.
        'select': None,
        'coverage_policy': 'complete',
        # Computed by preflight (see COMPUTED_KEYS); never configured.
        'measurement_identity': DRY_RUN_IDENTITY,
        'import_identity': None,
    }
    perf_params = {
        # Shared content-addressed aiq-magnet-evals store; default under the kwdagger root.
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
        # not replace, the parameters this class declares.
        supplied = dict(kwargs.get('algo_params') or {})
        declared = type(self).algo_params
        configured = sorted(
            key for key in COMPUTED_KEYS if key in supplied and supplied[key] != declared[key]
        )
        if configured:
            raise ValueError(
                f'EvaluationNode parameters {configured} are computed by preflight '
                'resolution and cannot be set by a recipe'
            )
        for key in ('algo_params', 'perf_params'):
            if kwargs.get(key) is not None:
                kwargs[key] = {**getattr(type(self), key), **dict(kwargs[key])}
        super().__init__(*args, **kwargs)

    def _final(self) -> dict[str, Any]:
        """``final_config`` as a plain mapping (the lease mixin types it optional)."""
        return dict(self.final_config or {})

    def _setting(self, key: str) -> Any:
        # Read perf settings without touching final_config (which would recurse
        # through final_algo_config).
        value = self.config.get(key)
        return self.perf_params.get(key) if value is None else value

    def _import_source(self, config: dict[str, Any]) -> str | None:
        source = config.get('import_source')
        # Relative to where the schedule is compiled, not the node's directory.
        return None if source in (None, '') else os.path.abspath(str(source))

    @property
    def final_algo_config(self) -> Any:
        """Algo params with the preflight-resolved identities (M3).

        kwdagger hashes this mapping into the node id. Inside a real
        ``preflight_scope`` the request is resolved once per schedule in the
        node's own execution environment; outside it (dry runs, result
        loading) nothing resolves. Either way, the computed identities replace
        any value a configuration supplied.
        """
        config = type(super().final_algo_config)(super().final_algo_config)
        self._check_static(config)
        state = _PREFLIGHT.get()
        if state is None or missing_request_keys(dict(config)):
            config['measurement_identity'] = DRY_RUN_IDENTITY
            config['import_identity'] = None
            return config
        command = self.preflight_command(config)
        resolution = state.resolutions.get(command)
        if resolution is None:
            resolution = state.resolutions[command] = run_preflight(command)
        identity = resolution['measurement_identity']
        if identity['reusable']:
            config['measurement_identity'] = identity['digest']
        else:
            nonce = state.nonces.setdefault(command, uuid.uuid4().hex)
            config['measurement_identity'] = f'unresolved-{nonce}'
        config['import_identity'] = resolution.get('import_identity')
        return config

    def _check_static(self, config: Any) -> None:
        """Selector and policy errors surface while compiling, even in a dry run."""
        from magnet.backends.aiq_evals.projection import (
            COVERAGE_POLICIES,
            normalize_selector,
        )

        normalize_selector(selector_value(config.get('select')))
        policy = config.get('coverage_policy') or 'complete'
        if policy not in COVERAGE_POLICIES:
            raise ValueError(f'EvaluationNode coverage_policy must be one of {COVERAGE_POLICIES}; got {policy!r}')

    def preflight_command(self, config: Any) -> str:
        """The resolution command, wrapped exactly like the node's own command."""
        args = {'request': json.dumps(build_request_dict(dict(config)), sort_keys=True)}
        worker = self._setting('worker_python')
        if worker not in (None, ''):
            args['worker_python'] = worker
        source = self._import_source(dict(config))
        if source is not None:
            args['import_source'] = source
            if _truthy(self._setting('allow_external_symlinks')):
                args['allow_external_symlinks'] = 'True'
        argstr = ' '.join(f'--{key}={shlex.quote(str(value))}' for key, value in args.items())
        return self._wrap_interpreter(f'{self.resolve_executable} {argstr}')

    def _store_dpath(self) -> str:
        store = self._final().get('store_dpath')
        if store:
            return str(store)
        root = getattr(self, 'root_dpath', None) or '.'
        return str(Path(root) / '_aiq_evals_store')

    @property
    def command(self) -> str:
        config = self._final()
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
        for key in COMPUTED_KEYS:
            if config.get(key) not in (None, ''):
                args[key] = config[key]
        select = selector_value(config.get('select'))
        if select is not None:
            args['select'] = json.dumps(select, sort_keys=True)
        for key in ('worker_python', 'timeout_seconds'):
            if config.get(key) not in (None, ''):
                args[key] = config[key]
        source = self._import_source(config)
        if source is not None:
            args['import_source'] = source
        if _truthy(config.get('allow_external_symlinks')):
            args['allow_external_symlinks'] = 'True'
        if config.get('endpoint'):
            args['endpoint'] = config['endpoint']
        argstr = ' \\\n    '.join(f'--{key}={shlex.quote(str(value))}' for key, value in args.items())
        command = f'{self.executable} \\\n    {argstr}'
        command = self._wrap_interpreter(command)
        if source is not None or self._native_result_available():
            # An import runs no model, and a stored run is only reused: leasing
            # would start a model for nothing (M8: avoid duplicate startup).
            return command
        return self.wrap_with_lease(command)

    def _native_result_available(self) -> bool:
        digest = str(self._final().get('measurement_identity') or '')
        if len(digest) != 64:
            return False
        try:
            from magnet_evals import load_run
            from magnet_evals.store import ResultStore

            run = load_run(ResultStore(self._store_dpath()).run_path(digest))
        except Exception:
            return False
        return run.complete and run.result.status == 'succeeded'

    def expected_evaluation(self) -> dict[str, Any]:
        """What this node's ``evaluation.json`` must record to count as done."""
        from magnet.backends.aiq_evals.projection import normalize_selector

        config = self._final()

        expected: dict[str, Any] = {
            'select': normalize_selector(selector_value(config.get('select'))),
            'coverage_policy': config.get('coverage_policy') or 'complete',
        }
        digest = str(config.get('measurement_identity') or '')
        if len(digest) == 64:
            expected['measurement_identity'] = digest
        if config.get('import_identity'):
            expected['import_identity'] = config['import_identity']
        return expected

    def _evaluation_fpath(self) -> Any:
        paths = self.final_out_paths
        return paths.get(str(self.primary_out_key)) if paths else None

    @property
    def does_exist(self) -> bool:
        """Done only if the recorded run still validates and matches this node."""
        fpath = self._evaluation_fpath()
        return fpath is not None and evaluation_is_valid(Path(fpath), self.expected_evaluation())

    def test_is_computed_command(self) -> str | None:
        """The job-level "done" guard validates the run, not just the marker file."""
        fpath = self._evaluation_fpath()
        if fpath is None:
            return None
        expected = json.dumps(self.expected_evaluation(), sort_keys=True)
        return self._wrap_interpreter(
            'python -m magnet.backends.aiq_evals.cli.check_done '
            f'{shlex.quote(str(fpath))} --expected={shlex.quote(expected)}'
        )

    def _wrap_interpreter(self, command: str) -> str:
        if self.containerization_is_enabled():
            return self.wrap_with_container(command)
        from magnet.containers import host_interpreter

        return host_interpreter(command)

    def load_result(self, node_dpath: Any) -> Any:
        return load_evaluation_row(self, node_dpath)


class InvalidEvaluation(ValueError):
    """An ``evaluation.json`` no longer matches the run it references."""


def validated_run(fpath: str | os.PathLike[str]) -> tuple[dict[str, Any], Any]:
    """Load ``evaluation.json`` and the run it references, or raise.

    The run must load with full checksum verification, be complete and
    succeeded, and match every identity the node recorded: measurement,
    normalized artifact, and (for imports) native content.
    """
    from magnet_evals import load_run

    fpath = Path(fpath)
    try:
        summary = json.loads(fpath.read_text())
    except (OSError, json.JSONDecodeError) as ex:
        raise InvalidEvaluation(f'unreadable {fpath}: {ex}') from ex
    if summary.get('schema') != EVALUATION_SCHEMA:
        raise InvalidEvaluation(f'{fpath} has schema {summary.get("schema")!r}; expected {EVALUATION_SCHEMA}')
    try:
        run = load_run(summary['run_path'])
    except Exception as ex:
        raise InvalidEvaluation(f'referenced run does not validate: {ex}') from ex
    if not (run.complete and run.result.status == 'succeeded'):
        raise InvalidEvaluation('referenced run is not a complete, succeeded run')
    if run.resolved.identity.digest != summary['measurement_identity']['digest']:
        raise InvalidEvaluation('referenced run has a different measurement identity')
    if run.manifest.get('normalized_artifact_identity') != summary.get('normalized_artifact_identity'):
        raise InvalidEvaluation('referenced run has a different normalized artifact identity')
    if summary.get('import_identity') and run.manifest.get('native_artifact_identity') != summary['import_identity']:
        raise InvalidEvaluation('referenced run holds different imported native content')
    return summary, run


def evaluation_is_valid(fpath: Path, expected: dict[str, Any] | None = None) -> bool:
    """Whether ``evaluation.json`` references a valid run, as ``expected``.

    ``expected`` (from :meth:`EvaluationNode.expected_evaluation`) pins the
    projection (selector, coverage policy) and identities the node was
    scheduled with, so an edited ``evaluation.json`` makes the node not done.
    """
    if not Path(fpath).is_file():
        return False
    try:
        summary, _ = validated_run(fpath)
    except (InvalidEvaluation, KeyError, TypeError):
        return False
    for key, value in dict(expected or {}).items():
        recorded = summary.get(key)
        if key == 'measurement_identity':
            recorded = (recorded or {}).get('digest')
        if recorded != value:
            return False
    return True


def load_evidence(fpath: str | os.PathLike[str]) -> tuple[dict[str, Any], Any]:
    """``(summary, evidence view)`` of a node, recomputed from its validated run (M4)."""
    from magnet.backends.aiq_evals.projection import evidence_view

    summary, run = validated_run(fpath)
    view = evidence_view(run, summary.get('select'), summary.get('coverage_policy') or 'complete')
    return summary, view


def load_evaluation_row(node: Any, node_dpath: Any) -> Any:
    """One flat row: ``metrics.<node>.*`` recomputed from the validated run (M5)."""
    from kwdagger.utils import util_dotdict

    from magnet.backends.aiq_evals.projection import flat_metrics

    summary, view = load_evidence(Path(node_dpath) / node.out_paths[node.primary_out_key])
    metrics = flat_metrics(view.to_dict())
    metrics['action'] = summary['action']
    metrics['import_identity'] = summary.get('import_identity')
    flat = util_dotdict.DotDict({f'metrics.{key}': value for key, value in metrics.items()})
    return flat.insert_prefix(node.name, index=1)
