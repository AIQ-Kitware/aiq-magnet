"""
Execution configuration arrives as CLI arguments and lands on the node.

What to run a node in, and whether it leases its own endpoints, is
configuration a caller passes. It used to be read from MAGNET_NODE_* /
MAGNET_PER_NODE_LEASING, which hid where a value came from; then from
process-wide settings, which hid *which* run a value belonged to. It now
travels as a :class:`~magnet.containers.ContainerSettings` /
:class:`~magnet.leasing.LeaseSettings` and is written onto the DAG's nodes
before anything renders a command.

Facts about the surrounding machine -- the GPU count, whether infer-stack
already wrapped us -- are still discovered, because only the machine can
state them.
"""

import os
from unittest import mock

from kwdagger.pipeline import coerce_pipeline

from magnet import containers, leasing
from magnet.containers import ContainerProcessNode, ContainerSettings
from magnet.leasing import LeasedProcessNode, LeaseSettings


class _Work(ContainerProcessNode):
    name = 'work'
    executable = 'python -m pkg.work'


class _Infer(LeasedProcessNode):
    name = 'infer'
    executable = 'python -m pkg.infer'


def _node(cls=_Work, settings=None, lease=None):
    node = cls()
    node.apply_settings(settings or ContainerSettings())
    if lease is not None:
        node.apply_lease_settings(lease)
    return node


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
        assert 'STALE_VAR' not in containers.forwarded_env(node)
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
    node.apply_settings(ContainerSettings.coerce(image='invocation:image'))
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
    assert leasing.leasing_is_enabled(off) is False
    asked = _node(_Infer, lease=LeaseSettings(enabled=True))
    assert leasing.leasing_is_enabled(asked) is True


def test_leasing_stays_off_inside_someone_elses_lease():
    """An ambient fact only infer-stack can state, so it stays an env var."""
    node = _node(_Infer, lease=LeaseSettings(enabled=True))
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
    """The reason none of this is process state.

    With a module-level setting, configuring the second run silently rewrote
    where the first one would say it ran. Nothing here is reachable from
    anywhere but the node, so there is nothing to reset between runs and no
    order in which these two can interfere.
    """
    first = _node(settings=ContainerSettings.coerce(image='first:image'))
    second = _node(settings=ContainerSettings.coerce(image='second:image'))
    assert containers.node_image(first) == 'first:image'
    assert containers.node_image(second) == 'second:image'


def test_applying_settings_twice_changes_nothing():
    """apply_settings runs once per schedule, but must not depend on that."""
    settings = ContainerSettings.coerce(
        image='i', mounts='/repo', forward_env='A,B')
    node = _node(settings=settings)
    once = containers.container_prefix(node)
    node.apply_settings(settings)
    assert containers.container_prefix(node) == once
    assert containers.forwarded_env(node).count('A') == 1


def test_settings_reach_every_node_of_a_built_dag():
    """What apply_settings is for: the DAG is where MAGNET holds the nodes.

    kwdagger constructs them -- for a declarative card, from the class the card
    names -- so there is no constructor to pass anything to and ``command``
    takes no arguments.
    """
    spec = {'nodes': {
        'work': {
            'class': 'magnet.containers.ContainerYamlProcessNode',
            'executable': 'python -m pkg.work',
            'out_paths': {'results_fpath': 'results.json'},
        },
        'plain': {
            'executable': 'python -m pkg.plain',
            'out_paths': {'results_fpath': 'plain.json'},
        },
    }}
    pipeline = coerce_pipeline(spec)
    containers.apply_settings(pipeline, ContainerSettings.coerce(image='i:tag'))

    assert pipeline.node_dict['work'].container_image == 'i:tag'
    # A node that cannot be containerized is skipped, not forced. Naming it is
    # _check_container_settings_apply's job.
    assert not hasattr(pipeline.node_dict['plain'], 'container_image')
