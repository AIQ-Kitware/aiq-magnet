"""
A leased node is confined to the GPUs its Slurm job was allocated.

infer-stack takes its GPU inventory from `nvidia-smi -L`. On a host with no
device cgroup that lists every card, not the ones this job was given -- aiq-gpu
sets ConstrainDevices=yes but TaskPlugin=task/none, so no cgroup is created and
a 2-GPU allocation still sees all four. Two nodes then place model servers on
the same card and one dies with CUDA OOM.

The allow-list cannot be a value: the DAG is rendered on the submit host, where
the allocation does not exist yet. It is shell text that expands at job time,
so these tests assert on the argv a shell actually produces rather than on how
the string is spelled.
"""

import os
import shlex
import subprocess
import textwrap
from unittest import mock

import pytest

from magnet import containers, leasing
from magnet.leasing import LeasedProcessNode

pytestmark = pytest.mark.skipif(
    not os.path.exists('/bin/bash'), reason='needs bash'
)


class Infer(LeasedProcessNode):
    name = 'infer'
    executable = 'infer'
    algo_params = {'model_id': None}
    endpoint_params = ('model_id',)


@pytest.fixture(autouse=True)
def _leasing_on(monkeypatch):
    monkeypatch.delenv(leasing.INSIDE_LEASE_ENVVAR, raising=False)
    # Poisoned on purpose: nothing may be interpolated at render time.
    monkeypatch.setenv('SLURM_JOB_GPUS', '7')
    containers.configure()
    leasing.configure(True)
    yield
    containers.configure()
    leasing.configure(False)


def _command():
    node = Infer()
    node.configure({'model_id': 'm'})
    return node.command


def _argv(command, env):
    """Run the rendered command under a stub infer-stack; return its argv."""
    stub_dir = os.environ['_STUB_DIR']
    proc = subprocess.run(
        ['/bin/bash', '-c', command],
        capture_output=True, text=True,
        env={'PATH': f'{stub_dir}:/usr/bin:/bin', 'HOME': '/tmp', **env},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.split('\n')


@pytest.fixture(autouse=True)
def _stub(tmp_path, monkeypatch):
    stub = tmp_path / 'infer-stack'
    stub.write_text(textwrap.dedent('''\
        #!/bin/bash
        for arg in "$@"; do echo "$arg"; done
    '''))
    stub.chmod(0o755)
    monkeypatch.setenv('_STUB_DIR', str(tmp_path))


def test_no_value_is_interpolated_when_the_command_is_rendered():
    """SLURM_JOB_GPUS=7 is set on the render host and must not appear."""
    command = _command()
    assert '--allowed_gpus=7' not in command
    assert '${SLURM_JOB_GPUS' in command


def test_off_slurm_the_flag_disappears_entirely():
    argv = _argv(_command(), {})
    assert not [a for a in argv if 'allowed_gpus' in a]
    # An empty unquoted expansion is dropped by the shell, so infer-stack is
    # never handed an empty --allowed_gpus to interpret.
    assert '' not in argv[:-1]


def test_the_job_allocation_is_used():
    argv = _argv(_command(), {'SLURM_JOB_GPUS': '0,1'})
    assert '--allowed_gpus=0,1' in argv


def test_the_step_allocation_is_the_fallback():
    """Measured under srun: SLURM_STEP_GPUS set, SLURM_JOB_GPUS not."""
    argv = _argv(_command(), {'SLURM_STEP_GPUS': '2,3'})
    assert '--allowed_gpus=2,3' in argv


def test_the_job_allocation_wins_over_the_step():
    argv = _argv(_command(), {'SLURM_JOB_GPUS': '0', 'SLURM_STEP_GPUS': '3'})
    assert '--allowed_gpus=0' in argv
    assert '--allowed_gpus=3' not in argv


def test_the_command_is_valid_shell():
    proc = subprocess.run(
        ['/bin/bash', '-n', '-c', _command()], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_it_survives_set_u():
    """A naive $SLURM_JOB_GPUS would abort the job under `set -u`."""
    command = 'set -u\n' + _command()
    argv = _argv(command, {})
    assert not [a for a in argv if 'allowed_gpus' in a]


def test_it_survives_being_quoted_into_a_job_script(tmp_path):
    """cmd_queue's Slurm backend wraps the whole command; that must defer."""
    script = tmp_path / 'job.sh'
    script.write_text('#!/bin/bash\n' + _command() + '\n')
    script.chmod(0o755)
    proc = subprocess.run(
        ['/bin/bash', '-c', shlex.quote(str(script))],
        capture_output=True, text=True,
        env={'PATH': f"{os.environ['_STUB_DIR']}:/usr/bin:/bin",
             'HOME': '/tmp', 'SLURM_JOB_GPUS': '1,2'},
    )
    assert proc.returncode == 0, proc.stderr
    assert '--allowed_gpus=1,2' in proc.stdout.split('\n')


def test_it_can_be_turned_off():
    leasing.configure(True, allowed_gpus=False)
    argv = _argv(_command(), {'SLURM_JOB_GPUS': '0,1'})
    assert not [a for a in argv if 'allowed_gpus' in a]


def test_cuda_visible_devices_is_not_consulted():
    """It may hold UUIDs, which infer-stack parses with int() and crashes on."""
    with mock.patch.dict(
        os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-4d888104-dead-beef'}
    ):
        command = _command()
    assert 'CUDA_VISIBLE_DEVICES' not in command
    argv = _argv(command, {'CUDA_VISIBLE_DEVICES': 'GPU-4d888104-dead-beef'})
    assert not [a for a in argv if 'allowed_gpus' in a]
