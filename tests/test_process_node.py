"""
Process-node configuration arrives as CLI arguments and lands on the node.

Containerization and endpoint leasing are independent capabilities. MAGNET's
normal integration surface carries both; the capability mixins remain usable
on their own without creating a public class for every combination.
"""

import os
import shutil
from unittest import mock

import kwdagger
from kwdagger.pipeline import coerce_pipeline

from magnet import containers, leasing
from magnet.containers import ContainerCapability, ContainerSettings
from magnet.leasing import LeaseCapability, LeaseSettings
from magnet.process_node import MagnetProcessNode


class _Work(MagnetProcessNode):
    name = 'work'
    executable = 'python -m pkg.work'


class _Infer(MagnetProcessNode):
    name = 'infer'
    executable = 'python -m pkg.infer'
    algo_params = {'model_id': None}
    endpoint_params = ('model_id',)


class _ContainerOnly(ContainerCapability, kwdagger.ProcessNode):
    """Test-only proof that the container capability stands alone."""

    name = 'container_only'
    executable = 'python -m pkg.work'

    @property
    def command(self):
        return containers.render_container_command(self, super().command)


class _LeaseOnly(LeaseCapability, kwdagger.ProcessNode):
    """Test-only proof that the leasing capability stands alone."""

    name = 'lease_only'
    executable = 'python -m pkg.infer'
    algo_params = {'model_id': None}
    endpoint_params = ('model_id',)

    @property
    def command(self):
        base = containers.host_interpreter(super().command)
        return leasing.render_lease_command(self, base)


def _node(cls=_Work, settings=None, lease=None):
    node = cls()
    if containers.is_container_capable(node):
        node.apply_container_settings(settings or ContainerSettings())
    if lease is not None and leasing.is_lease_capable(node):
        node.apply_lease_settings(lease)
    return node


def test_capability_mixins_can_be_composed_independently(monkeypatch):
    """The architecture is orthogonal even though MAGNET exposes one node."""
    boxed = _ContainerOnly()
    boxed.configure({})
    boxed.apply_container_settings(ContainerSettings.coerce(image='box:latest'))
    assert containers.is_container_capable(boxed)
    assert not leasing.is_lease_capable(boxed)
    assert boxed.command.startswith('docker run --rm ')
    assert 'infer-stack run' not in boxed.command

    # Rendering a lease prefix checks whether infer-stack exists. This test is
    # about composition, so provide a harmless positive discovery result.
    monkeypatch.setattr(shutil, 'which', lambda name: f'/fake/{name}')
    leased = _LeaseOnly()
    leased.configure({'model_id': 'smol-135'})
    leased.apply_lease_settings(LeaseSettings(enabled=True))
    assert leasing.is_lease_capable(leased)
    assert not containers.is_container_capable(leased)
    assert leased.command.startswith('infer-stack run ')
    assert 'docker run' not in leased.command

    integrated = _Work()
    assert containers.is_container_capable(integrated)
    assert leasing.is_lease_capable(integrated)


def test_nothing_is_read_from_the_old_environment_variables():
    stale = {
        'MAGNET_NODE_IMAGE': 'stale:image',
        'MAGNET_NODE_MOUNTS': '/stale',
        'MAGNET_NODE_DOCKER_ARGS': '--stale',
        'MAGNET_NODE_FORWARD_ENV': 'STALE_VAR',
        'MAGNET_PER_NODE_LEASING': '1',
    }
    with mock.patch.dict(os.environ, stale):
        node = _node(_Infer, lease=LeaseSettings())
        assert containers.containerization_is_enabled(node) is False
        assert containers.node_mounts(node) == []
        assert 'STALE_VAR' not in (node.container_forward_env or ())
        assert leasing.leasing_is_enabled(node) is False


def test_the_image_comes_from_configuration():
    node = _node(settings=ContainerSettings.coerce(
        image='magnet:latest', mounts='/repo'))
    assert containers.containerization_is_enabled(node) is True
    prefix = containers.container_prefix(node)
    assert prefix.endswith('magnet:latest')
    assert '-v /repo:/repo' in prefix


def test_a_node_still_wins_over_the_invocation():
    node = _Work()
    node.container_image = 'node:image'
    node.apply_container_settings(
        ContainerSettings.coerce(image='invocation:image')
    )
    assert containers.node_image(node) == 'node:image'


def test_mounts_accept_a_list_or_a_separated_string():
    for spec in (['/a', '/b'], '/a:/b', '/a,/b'):
        node = _node(settings=ContainerSettings.coerce(mounts=spec))
        assert containers.node_mounts(node) == ['/a', '/b']


def test_docker_args_reach_the_prefix():
    node = _node(settings=ContainerSettings.coerce(
        image='i', docker_args='--gpus all'))
    assert '--gpus all' in containers.container_prefix(node)


def test_leasing_is_off_until_asked_for():
    off = _node(_Infer, lease=LeaseSettings())
    off.configure({'model_id': 'm'})
    assert leasing.leasing_is_enabled(off) is False
    asked = _node(_Infer, lease=LeaseSettings(enabled=True))
    asked.configure({'model_id': 'm'})
    assert leasing.leasing_is_enabled(asked) is True


def test_leasing_stays_off_inside_someone_elses_lease():
    """An ambient fact only infer-stack can state, so it stays an env var."""
    node = _node(_Infer, lease=LeaseSettings(enabled=True))
    node.configure({'model_id': 'm'})
    with mock.patch.dict(os.environ, {leasing.INSIDE_LEASE_ENVVAR: 'abc123'}):
        assert leasing.leasing_is_enabled(node) is False


def test_the_endpoint_variables_are_still_forwarded_by_name():
    """infer-stack owns these; magnet must not capture a value for them."""
    node = _node(settings=ContainerSettings.coerce(image='i'))
    with mock.patch.dict(os.environ, {'OPENAI_BASE_URL': 'http://stale'}):
        prefix = containers.container_prefix(node)
    assert '-e OPENAI_BASE_URL' in prefix
    assert 'http://stale' not in prefix


def test_two_runs_in_one_process_do_not_share_settings():
    first = _node(settings=ContainerSettings.coerce(image='first:image'))
    second = _node(settings=ContainerSettings.coerce(image='second:image'))
    assert containers.node_image(first) == 'first:image'
    assert containers.node_image(second) == 'second:image'


def test_applying_settings_twice_changes_nothing():
    settings = ContainerSettings.coerce(
        image='i', mounts='/repo', forward_env='A,B')
    node = _node(settings=settings)
    once = containers.container_prefix(node)
    node.apply_container_settings(settings)
    assert containers.container_prefix(node) == once
    assert tuple(node.container_forward_env or ()).count('A') == 1


def test_settings_reach_every_node_of_a_built_dag():
    """The one MAGNET class works as kwdagger's declarative class too."""
    spec = {'nodes': {
        'work': {
            'class': 'magnet.process_node.MagnetProcessNode',
            'executable': 'python -m pkg.work',
            'out_paths': {'results_fpath': 'results.json'},
        },
        'plain': {
            'executable': 'python -m pkg.plain',
            'out_paths': {'results_fpath': 'plain.json'},
        },
    }}
    pipeline = coerce_pipeline(spec)
    ContainerSettings.coerce(image='i:tag').apply(pipeline)

    assert pipeline.node_dict['work'].container_image == 'i:tag'
    assert not hasattr(pipeline.node_dict['plain'], 'container_image')
