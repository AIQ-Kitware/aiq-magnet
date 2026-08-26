"""
The queue worker cap is bounded by the GPUs this machine has.

A leased node holds its answerer while waiting for the extractor it also needs.
Start enough shards to claim every GPU and none can get the extractor, so none
release: measured on a 4-GPU host as four answerers placed, eight leases
queued, zero rows in an hour. It does not fail, it converges forever.
"""

from unittest import mock

from magnet.evaluation_new import DEFAULT_TMUX_WORKERS, resolve_tmux_workers


def test_an_explicit_worker_count_is_used_as_given():
    assert resolve_tmux_workers(4) == 4
    assert resolve_tmux_workers('4') == 4


def test_auto_leaves_one_gpu_for_the_shared_extractor():
    with mock.patch('magnet.evaluation_new.detected_gpu_count', return_value=4):
        assert resolve_tmux_workers('auto') == 3


def test_auto_never_resolves_to_zero_workers():
    with mock.patch('magnet.evaluation_new.detected_gpu_count', return_value=1):
        assert resolve_tmux_workers('auto') == 1


def test_auto_on_a_host_without_gpus_keeps_the_plain_default():
    """No GPUs means no GPU contention to protect against."""
    with mock.patch('magnet.evaluation_new.detected_gpu_count', return_value=0):
        assert resolve_tmux_workers('auto') == DEFAULT_TMUX_WORKERS
