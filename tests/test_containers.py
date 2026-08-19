"""
Containerized node execution.

The boundary under test is orchestration-outside / work-inside: MAGNET
compiles the DAG on the host, each node's command runs in an image.
"""

import kwdagger
import pytest

from magnet.containers import (
    FORWARD_ENV_ENVVAR,
    IMAGE_ENVVAR,
    MOUNTS_ENVVAR,
    ContainerProcessNode,
    containerization_is_enabled,
)
from magnet.leasing import INSIDE_LEASE_ENVVAR, LEASING_ENVVAR, LeasedProcessNode

IMAGE = 'aiq-eval-node:latest'


class Work(ContainerProcessNode):
    name = 'work'
    executable = 'python -m pkg.work'
    algo_params = {'task': None}


class Infer(LeasedProcessNode):
    name = 'infer'
    executable = 'python -m pkg.infer'
    endpoint_params = ('model_id',)
    algo_params = {'model_id': None}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(INSIDE_LEASE_ENVVAR, raising=False)
    monkeypatch.delenv(LEASING_ENVVAR, raising=False)
    monkeypatch.delenv(IMAGE_ENVVAR, raising=False)
    monkeypatch.delenv(MOUNTS_ENVVAR, raising=False)
    monkeypatch.delenv(FORWARD_ENV_ENVVAR, raising=False)


def _node(cls, config):
    node = cls()
    node.configure(config)
    return node


def _on(monkeypatch, *, image=IMAGE, mounts='/repo'):
    monkeypatch.setenv(IMAGE_ENVVAR, image)
    monkeypatch.setenv(MOUNTS_ENVVAR, mounts)


def test_nodes_run_on_the_host_unless_an_image_is_named():
    assert not containerization_is_enabled()
    assert _node(Work, {'task': 't'}).command.startswith('python -m pkg.work')


def test_the_command_runs_in_the_image(monkeypatch):
    _on(monkeypatch)
    command = _node(Work, {'task': 't'}).command
    assert command.startswith('docker run --rm ')
    assert command.rstrip().endswith('--task=t')
    # The image name immediately precedes the command it runs.
    prefix, rest = command.split(f' {IMAGE} ', 1)
    assert 'python -m pkg.work' in rest


def test_the_repo_is_mounted_at_its_own_path(monkeypatch):
    """kwdagger bakes absolute paths into every command; they have to
    resolve to the same files inside the container."""
    _on(monkeypatch, mounts='/a/repo:/b/data')
    command = _node(Work, {'task': 't'}).command
    assert '-v /a/repo:/a/repo' in command
    assert '-v /b/data:/b/data' in command


def test_artifacts_are_not_root_owned(monkeypatch):
    import os

    _on(monkeypatch)
    assert f'--user {os.getuid()}:{os.getgid()}' in _node(Work, {'task': 't'}).command


def test_the_endpoint_env_is_forwarded_by_name(monkeypatch):
    """By name, not by value: the lease sets it at job time, long after
    this command string was rendered."""
    _on(monkeypatch)
    command = _node(Work, {'task': 't'}).command
    assert '-e OPENAI_BASE_URL' in command
    assert '-e OPENAI_API_KEY' in command
    assert 'OPENAI_BASE_URL=' not in command


def test_a_pipelines_own_variables_are_forwarded_on_request(monkeypatch):
    """MAGNET must not need to know what an evaluation calls its settings."""
    _on(monkeypatch)
    monkeypatch.setenv(FORWARD_ENV_ENVVAR, 'SOME_BACKEND_FACTORY,SOME_URL')
    command = _node(Work, {'task': 't'}).command
    assert '-e SOME_BACKEND_FACTORY' in command
    assert '-e SOME_URL' in command


def test_the_defaults_are_generic(monkeypatch):
    """Nothing evaluation-specific may be baked into the default set.

    A generic framework naming one evaluation's variables is a design smell
    -- and a disclosure risk, since not every evaluation repo is public and
    this one is. Whitelisting recognised prefixes means a new default has to
    be a well-known variable or an explicit decision.
    """
    from magnet.containers import DEFAULT_FORWARDED_ENV

    allowed = ('OPENAI_', 'HF_', 'PYTHON', 'TRANSFORMERS_')
    for name in DEFAULT_FORWARDED_ENV:
        assert name.startswith(allowed), name


def test_the_lease_wraps_the_container_not_the_other_way_round(monkeypatch):
    """Acquiring needs the Docker daemon and the ledger, both on the host.

    Inside-out would mean a container reaching for the host's daemon; and
    being inside is what lets the container inherit the endpoint env.
    """
    _on(monkeypatch)
    monkeypatch.setenv(LEASING_ENVVAR, '1')
    command = _node(Infer, {'model_id': 'qwen'}).command
    assert command.index('infer-stack run') < command.index('docker run')


def test_either_layer_works_alone(monkeypatch):
    monkeypatch.setenv(LEASING_ENVVAR, '1')
    leased_only = _node(Infer, {'model_id': 'qwen'}).command
    assert leased_only.startswith('infer-stack run')
    assert 'docker run' not in leased_only

    monkeypatch.delenv(LEASING_ENVVAR)
    _on(monkeypatch)
    boxed_only = _node(Infer, {'model_id': 'qwen'}).command
    assert boxed_only.startswith('docker run')
    assert 'infer-stack run' not in boxed_only


def test_it_is_still_an_ordinary_kwdagger_node(monkeypatch):
    _on(monkeypatch)
    node = _node(Work, {'task': 't'})
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
    _on(monkeypatch)
    monkeypatch.setenv(FORWARD_ENV_ENVVAR, 'SOME_BACKEND_FACTORY')
    monkeypatch.setenv('SOME_BACKEND_FACTORY', 'pkg.mod:factory')
    command = _node(Work, {'task': 't'}).command
    assert '-e SOME_BACKEND_FACTORY=pkg.mod:factory' in command


def test_a_lease_variable_is_never_captured_even_when_set(monkeypatch):
    """A lease value must come from the job, not the orchestrator's shell.

    OPENAI_BASE_URL set in the orchestrator is not the endpoint this job
    leased; baking it in would freeze the wrong URL over the one
    ``infer-stack run`` writes at job time.
    """
    _on(monkeypatch)
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://stale-orchestrator/v1')
    command = _node(Work, {'task': 't'}).command
    assert '-e OPENAI_BASE_URL' in command
    assert 'stale-orchestrator' not in command


def test_a_node_may_declare_its_own_container_settings(monkeypatch):
    """A node's own image, mounts and env override the process-wide ones."""
    _on(monkeypatch)

    class Declared(Work):
        container_image = 'other:tag'
        container_mounts = ['/a', '/b']
        container_env = {'SOME_BACKEND_FACTORY': 'node.declared:factory'}

    command = _node(Declared, {'task': 't'}).command
    assert 'other:tag' in command
    assert '-v /a:/a' in command and '-v /b:/b' in command
    assert '-e SOME_BACKEND_FACTORY=node.declared:factory' in command
    # The process-wide image is overridden, not appended to.
    assert 'aiq-eval-node:latest' not in command


def test_pythonpath_is_captured_not_left_bare(monkeypatch):
    """PYTHONPATH is orchestrator configuration, not a lease value.

    Left as a bare name it arrives empty in a cmd_queue tmux worker that did
    not inherit it, and every import inside the node fails.
    """
    _on(monkeypatch)
    monkeypatch.setenv('PYTHONPATH', '/repo:/repo/ta1/thing')
    command = _node(Work, {'task': 't'}).command
    assert '-e PYTHONPATH=/repo:/repo/ta1/thing' in command
