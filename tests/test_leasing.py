"""
Per-node endpoint leasing.

The property under test is not "the string looks right" -- it is that the
DAG, not a wrapper script, decides which model is held while which job runs.
"""

import kwdagger
import pytest

from magnet import containers, leasing
from magnet.leasing import (
    INSIDE_LEASE_ENVVAR,
    LeasedProcessNode,
    leasing_is_enabled,
)


class Infer(LeasedProcessNode):
    name = 'infer'
    executable = 'python -m pkg.infer'
    endpoint_params = ('model_id', 'extractor_model_id')
    algo_params = {'model_id': None, 'extractor_model_id': None}


class Analyse(LeasedProcessNode):
    """A node that touches no model: it must never hold one."""

    name = 'analyse'
    executable = 'python -m pkg.analyse'
    algo_params = {'metric': None}


@pytest.fixture(autouse=True)
def _leasing_on(monkeypatch):
    monkeypatch.delenv(INSIDE_LEASE_ENVVAR, raising=False)
    # Both settings are process-wide, so unlike a monkeypatched environment
    # variable they persist across tests unless put back deliberately.
    containers.configure()
    leasing.configure(True)
    yield
    containers.configure()
    leasing.configure(False)


def _node(cls, config):
    node = cls()
    node.configure(config)
    return node


def _prefix(command):
    """The lease wrapper only, without the wrapped command."""
    return command.split(' -- ', 1)[0]


def test_the_node_leases_the_models_it_names():
    node = _node(Infer, {'model_id': 'mock/tiny-1b',
                         'extractor_model_id': 'mock/extractor-70b'})
    command = node.command
    assert command.startswith('infer-stack run ')
    # ONE --endpoint carrying every name. `infer-stack run` takes a single
    # comma-separated string, so a repeated flag would silently keep only the
    # last model and leave the rest unleased.
    assert _prefix(command).count('--endpoint') == 1
    assert '--endpoint mock/tiny-1b,mock/extractor-70b' in command
    # The original command survives intact after the `--`.
    assert 'python -m pkg.infer' in command.split(' -- ', 1)[1]
    assert '--model_id=mock/tiny-1b' in command


def test_a_model_free_node_holds_nothing():
    node = _node(Analyse, {'metric': 'auroc'})
    assert node.command.startswith('python -m pkg.analyse')
    assert 'infer-stack' not in node.command


def test_an_unset_endpoint_param_does_not_become_a_lease():
    node = _node(Infer, {'model_id': 'mock/tiny-1b',
                         'extractor_model_id': None})
    prefix = _prefix(node.command)
    assert '--endpoint mock/tiny-1b ' in prefix
    assert 'None' not in prefix


def test_the_same_model_twice_is_named_once():
    node = _node(Infer, {'model_id': 'm', 'extractor_model_id': 'm'})
    assert '--endpoint m ' in _prefix(node.command)


def test_two_instances_lease_different_models():
    """The whole point: the alias comes from the instance, not the class."""
    a = _node(Infer, {'model_id': 'mock/tiny-1b', 'extractor_model_id': None})
    b = _node(Infer, {'model_id': 'mock/frontier-b', 'extractor_model_id': None})
    assert '--endpoint mock/tiny-1b' in a.command
    assert '--endpoint mock/frontier-b' in b.command
    assert 'frontier' not in a.command


def test_the_lease_waits_rather_than_failing_when_busy():
    # A DAG routinely schedules more jobs than the box has GPUs; treating
    # that as an error would make --jobs > n_gpus unusable.
    command = _node(Infer, {'model_id': 'm', 'extractor_model_id': None}).command
    assert '--queue' in command
    assert '--timeout' in command
    assert '--ttl' in command


def test_leasing_is_off_inside_an_outer_lease(monkeypatch):
    leasing.configure(True)
    monkeypatch.setenv(INSIDE_LEASE_ENVVAR, 'lease-abc123')
    assert not leasing_is_enabled()
    node = _node(Infer, {'model_id': 'm', 'extractor_model_id': None})
    assert node.command.startswith('python -m pkg.infer')


def test_explicit_opt_out(monkeypatch):
    # e.g. a run against OpenRouter, which infer-stack does not manage.
    monkeypatch.delenv(INSIDE_LEASE_ENVVAR, raising=False)
    leasing.configure(False)
    assert not leasing_is_enabled()


def test_leasing_is_off_unless_asked_for(monkeypatch):
    """A card pointed at an unmanaged server must keep working untouched."""
    monkeypatch.delenv(INSIDE_LEASE_ENVVAR, raising=False)
    leasing.configure(False)
    assert not leasing_is_enabled()
    node = _node(Infer, {'model_id': 'm', 'extractor_model_id': None})
    assert node.command.startswith('python -m pkg.infer')


def test_it_is_still_an_ordinary_kwdagger_node():
    node = _node(Infer, {'model_id': 'm', 'extractor_model_id': 'e'})
    assert isinstance(node, kwdagger.ProcessNode)
    # The lease must not leak into identity: two runs of the same work under
    # different lease settings are the same work.
    assert 'infer-stack' not in str(node.algo_id)
    assert 'infer-stack' not in str(node.process_id)
