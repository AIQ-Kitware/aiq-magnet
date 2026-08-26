"""
Execution configuration arrives as CLI arguments, not environment variables.

What to run a node in, and whether it leases its own endpoints, is
configuration a caller passes. It used to be read from MAGNET_NODE_* /
MAGNET_PER_NODE_LEASING, which hid where a value came from. Facts about the
surrounding machine -- the GPU count, whether infer-stack already wrapped us --
are still discovered, because only the machine can state them.
"""

import os
from unittest import mock

import pytest

from magnet import containers, leasing

@pytest.fixture(autouse=True)
def _restore():
    before_containers = containers.current_settings()
    before_leasing = leasing.leasing_is_enabled()
    yield
    containers.configure(
        image=before_containers.image,
        mounts=before_containers.mounts,
        docker_args=before_containers.docker_args,
        forward_env=before_containers.forward_env,
    )
    leasing.configure(before_leasing)


class _Node:
    pass


def test_nothing_is_read_from_the_old_environment_variables():
    stale = {
        'MAGNET_NODE_IMAGE': 'stale:image',
        'MAGNET_NODE_MOUNTS': '/stale',
        'MAGNET_NODE_DOCKER_ARGS': '--stale',
        'MAGNET_NODE_FORWARD_ENV': 'STALE_VAR',
        'MAGNET_PER_NODE_LEASING': '1',
    }
    with mock.patch.dict(os.environ, stale):
        containers.configure()
        leasing.configure(False)
        assert containers.containerization_is_enabled() is False
        assert containers.node_mounts() == []
        assert 'STALE_VAR' not in containers.forwarded_env()
        assert leasing.leasing_is_enabled() is False


def test_the_image_comes_from_configuration():
    containers.configure(image='magnet:latest', mounts='/repo')
    assert containers.containerization_is_enabled() is True
    prefix = containers.container_prefix()
    assert prefix.endswith('magnet:latest')
    assert '-v /repo:/repo' in prefix


def test_a_node_still_wins_over_the_process_setting():
    containers.configure(image='process:image')
    node = _Node()
    node.container_image = 'node:image'
    assert containers.node_image(node) == 'node:image'


def test_mounts_accept_a_list_or_a_separated_string():
    containers.configure(mounts=['/a', '/b'])
    assert containers.node_mounts() == ['/a', '/b']
    containers.configure(mounts='/a:/b')
    assert containers.node_mounts() == ['/a', '/b']
    containers.configure(mounts='/a,/b')
    assert containers.node_mounts() == ['/a', '/b']


def test_docker_args_reach_the_prefix():
    containers.configure(image='i', docker_args='--gpus all')
    assert '--gpus all' in containers.container_prefix()


def test_leasing_is_off_until_asked_for():
    leasing.configure(False)
    assert leasing.leasing_is_enabled() is False
    leasing.configure(True)
    assert leasing.leasing_is_enabled() is True


def test_leasing_stays_off_inside_someone_elses_lease():
    """An ambient fact only infer-stack can state, so it stays an env var."""
    leasing.configure(True)
    with mock.patch.dict(os.environ, {leasing.INSIDE_LEASE_ENVVAR: 'abc123'}):
        assert leasing.leasing_is_enabled() is False


def test_the_endpoint_variables_are_still_forwarded_by_name():
    """infer-stack owns these; magnet must not capture a value for them."""
    containers.configure(image='i')
    with mock.patch.dict(os.environ, {'OPENAI_BASE_URL': 'http://stale'}):
        prefix = containers.container_prefix()
    assert '-e OPENAI_BASE_URL' in prefix
    assert 'http://stale' not in prefix
