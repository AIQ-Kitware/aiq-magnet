"""
Containerized node execution.

The boundary under test is orchestration-outside / work-inside: MAGNET
compiles the DAG on the host, each node's command runs in an image.
"""

import kwdagger
import pytest

from magnet.containers import (
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
