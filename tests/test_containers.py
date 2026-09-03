"""
Containerized node execution.

The boundary under test is orchestration-outside / work-inside: MAGNET
compiles the DAG on the host, each node's command runs in an image.
"""

import kwdagger
import pytest

from magnet import containers
from magnet.containers import (
    ContainerProcessNode,
    containerization_is_enabled,
)
from magnet import leasing
from magnet.leasing import INSIDE_LEASE_ENVVAR, LeasedProcessNode
from magnet.execution import MagnetProcessNode

import shlex as _shlex
import sys as _sys
#: On the host route a bare ``python`` renders as this interpreter (magnet.containers.host_interpreter).
HOST_PY = _shlex.quote(_sys.executable)

IMAGE = 'aiq-eval-node:latest'


class Work(ContainerProcessNode):
    name = 'work'
    executable = 'python -m pkg.work'
    algo_params = {'task': None}


class Infer(MagnetProcessNode):
    name = 'infer'
    executable = 'python -m pkg.infer'
    endpoint_params = ('model_id',)
    algo_params = {'model_id': None}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Only the ambient variable needs clearing. Execution settings live on the
    # node each test builds, so there is no process state to reset.
    monkeypatch.delenv(INSIDE_LEASE_ENVVAR, raising=False)


def _node(cls, config, settings=None, lease=None):
    """A configured node carrying this test's execution settings."""
    node = cls()
    node.configure(config)
    if containers.is_container_capable(node):
        node.apply_settings(settings or containers.ContainerSettings())
    if lease is not None and leasing.is_lease_capable(node):
        node.apply_lease_settings(lease)
    return node


def _on(*, image=IMAGE, mounts='/repo', **kwargs):
    """The container settings a containerized invocation would supply."""
    return containers.ContainerSettings.coerce(
        image=image, mounts=mounts, **kwargs
    )


def _leased(enabled=True):
    return leasing.LeaseSettings(enabled=enabled)


def test_nodes_run_on_the_host_unless_an_image_is_named():
    node = _node(Work, {'task': 't'})
    assert not containerization_is_enabled(node)
    assert node.command.startswith(HOST_PY + ' -m pkg.work')


def test_invocation_container_env_overrides_captured_host_env(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/host/examples')
    command = _node(
        Work,
        {'task': 't'},
        _on(env={'PYTHONPATH': '/opt/magnet/examples'}),
    ).command
    assert 'PYTHONPATH=/opt/magnet/examples' in command
    assert '/host/examples' not in command


def test_container_env_accepts_json():
    settings = containers.ContainerSettings.coerce(
        env='{"PYTHONPATH": "/opt/magnet/examples", "DEMO": 1}'
    )
    assert settings.env == {
        'PYTHONPATH': '/opt/magnet/examples',
        'DEMO': '1',
    }


def test_the_command_runs_in_the_image():
    command = _node(Work, {'task': 't'}, _on()).command
    assert command.startswith('docker run --rm ')
    assert command.rstrip().endswith('--task=t')
    # The image name immediately precedes the command it runs.
    prefix, rest = command.split(f' {IMAGE} ', 1)
    assert 'python -m pkg.work' in rest


def test_the_repo_is_mounted_at_its_own_path(monkeypatch):
    """kwdagger bakes absolute paths into every command; they have to
    resolve to the same files inside the container."""
    command = _node(Work, {'task': 't'}, _on(mounts='/a/repo:/b/data')).command
    assert '-v /a/repo:/a/repo' in command
    assert '-v /b/data:/b/data' in command


def test_artifacts_are_not_root_owned():
    import os

    command = _node(Work, {'task': 't'}, _on()).command
    assert f'--user {os.getuid()}:{os.getgid()}' in command


def test_the_endpoint_env_is_forwarded_by_name():
    """By name, not by value: the lease sets it at job time, long after
    this command string was rendered."""
    command = _node(Work, {'task': 't'}, _on()).command
    assert '-e OPENAI_BASE_URL' in command
    assert '-e OPENAI_API_KEY' in command
    assert 'OPENAI_BASE_URL=' not in command


def test_a_pipelines_own_variables_are_forwarded_on_request():
    """MAGNET must not need to know what an evaluation calls its settings."""
    settings = _on(forward_env='SOME_BACKEND_FACTORY,SOME_URL')
    command = _node(Work, {'task': 't'}, settings).command
    assert '-e SOME_BACKEND_FACTORY' in command
    assert '-e SOME_URL' in command


def test_the_default_env_policy_is_generic_and_overridable():
    """Nothing evaluation-specific may be baked into the default set.

    A generic framework naming one evaluation's variables is a design smell
    -- and a disclosure risk, since not every evaluation repo is public.
    Keeping the policy on the node class also lets a specialized node replace
    it without editing a module-level passlist.
    """
    allowed = ('OPENAI_', 'HF_', 'PYTHON', 'TRANSFORMERS_')
    names = Work.container_runtime_env + Work.container_capture_env
    for name in names:
        assert name.startswith(allowed), name

    class Minimal(Work):
        container_runtime_env = ()
        container_capture_env = ('PYTHONPATH',)

    command = _node(Minimal, {'task': 't'}, _on()).command
    assert 'OPENAI_BASE_URL' not in command


def test_container_and_lease_nodes_are_independent_siblings():
    assert not issubclass(LeasedProcessNode, ContainerProcessNode)
    assert not issubclass(ContainerProcessNode, LeasedProcessNode)
    assert containers.is_container_capable(ContainerProcessNode())
    assert not leasing.is_lease_capable(ContainerProcessNode())
    assert leasing.is_lease_capable(LeasedProcessNode())
    assert not containers.is_container_capable(LeasedProcessNode())
    both = MagnetProcessNode()
    assert containers.is_container_capable(both)
    assert leasing.is_lease_capable(both)


def test_the_lease_wraps_the_container_not_the_other_way_round(monkeypatch):
    """Acquiring needs the Docker daemon and the ledger, both on the host.

    Inside-out would mean a container reaching for the host's daemon; and
    being inside is what lets the container inherit the endpoint env.
    """
    command = _node(Infer, {'model_id': 'qwen'}, _on(), _leased()).command
    assert command.index('infer-stack run') < command.index('docker run')


def test_either_layer_works_alone():
    leased_only = _node(Infer, {'model_id': 'qwen'}, lease=_leased()).command
    assert leased_only.startswith('infer-stack run')
    assert 'docker run' not in leased_only

    boxed_only = _node(
        Infer, {'model_id': 'qwen'}, _on(), _leased(enabled=False)
    ).command
    assert boxed_only.startswith('docker run')
    assert 'infer-stack run' not in boxed_only


def test_it_is_still_an_ordinary_kwdagger_node():
    node = _node(Work, {'task': 't'}, _on())
    assert isinstance(node, kwdagger.ProcessNode)
    # Where a node runs must not change what it computes.
    assert 'docker' not in str(node.algo_id)
    assert 'docker' not in str(node.process_id)


def test_a_declared_variables_value_is_captured_not_forwarded(monkeypatch):
    """The environment that runs the command is not the one that rendered it.

    A cmd_queue tmux worker created against an already-running server inherits
    that server's environment, so a bare ``-e NAME`` for orchestrator
    configuration forwards nothing and the node silently falls back to a
    default.
    """
    monkeypatch.setenv('SOME_BACKEND_FACTORY', 'pkg.mod:factory')
    settings = _on(forward_env='SOME_BACKEND_FACTORY')
    command = _node(Work, {'task': 't'}, settings).command
    assert '-e SOME_BACKEND_FACTORY=pkg.mod:factory' in command


def test_a_lease_variable_is_never_captured_even_when_set(monkeypatch):
    """A lease value must come from the job, not the orchestrator's shell.

    OPENAI_BASE_URL set in the orchestrator is not the endpoint this job
    leased; baking it in would freeze the wrong URL over the one
    ``infer-stack run`` writes at job time.
    """
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://stale-orchestrator/v1')
    command = _node(Work, {'task': 't'}, _on()).command
    assert '-e OPENAI_BASE_URL' in command
    assert 'stale-orchestrator' not in command


def test_a_node_may_declare_its_own_container_settings():
    """A node's own image, mounts and env override the invocation's."""

    class Declared(Work):
        container_image = 'other:tag'
        container_mounts = ['/a', '/b']
        container_env = {'SOME_BACKEND_FACTORY': 'node.declared:factory'}

    command = _node(Declared, {'task': 't'}, _on()).command
    assert 'other:tag' in command
    assert '-v /a:/a' in command and '-v /b:/b' in command
    assert '-e SOME_BACKEND_FACTORY=node.declared:factory' in command
    # The invocation's image is overridden, not appended to.
    assert 'aiq-eval-node:latest' not in command


def test_pythonpath_is_captured_not_left_bare(monkeypatch):
    """PYTHONPATH is orchestrator configuration, not a lease value.

    Left as a bare name it arrives empty in a cmd_queue tmux worker that did
    not inherit it, and every import inside the node fails.
    """
    monkeypatch.setenv('PYTHONPATH', '/repo:/repo/ta1/thing')
    command = _node(Work, {'task': 't'}, _on()).command
    assert '-e PYTHONPATH=/repo:/repo/ta1/thing' in command


def test_bare_python_means_the_interpreter_that_runs_the_node():
    """On the host the orchestrator's interpreter runs the node; in an image
    the image's does. A pipeline writes ``python`` and never has to guess
    which at construction."""
    import shlex
    import sys
    node = Work()
    assert node.command.startswith(shlex.quote(sys.executable) + ' -m pkg.work')
    node.container_image = IMAGE
    assert node.command.startswith('docker run')
    assert ' \\\n    python -m pkg.work' in node.command
    assert sys.executable not in node.command
