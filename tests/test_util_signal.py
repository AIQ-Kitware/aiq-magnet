"""Signal handling preserves cleanup, exit codes and the caller's handlers."""

import signal
import subprocess
import sys

import pytest

from magnet.utils.util_signal import signals_as_system_exit


@pytest.mark.parametrize('fail', [False, True])
def test_restores_handlers_after_normal_or_exceptional_exit(fail):
    signals = [signal.SIGTERM, signal.SIGINT]
    previous = {signum: signal.getsignal(signum) for signum in signals}
    try:
        with signals_as_system_exit():
            if fail:
                raise RuntimeError('test failure')
    except RuntimeError:
        assert fail
    restored = {signum: signal.getsignal(signum) for signum in signals}
    assert restored == previous


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT])
def test_signal_runs_cleanup_and_exits_with_conventional_status(signum):
    # Deliver real signals in a child, protecting the pytest process.
    program = '''
import signal
import sys
from magnet.utils.util_signal import signals_as_system_exit

signum = int(sys.argv[1])
previous = signal.getsignal(signum)
try:
    with signals_as_system_exit():
        try:
            signal.raise_signal(signum)
        finally:
            print('cleanup ran', flush=True)
finally:
    assert signal.getsignal(signum) == previous
'''
    result = subprocess.run(
        [sys.executable, '-c', program, str(signum)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 128 + signum, result.stderr
    assert result.stdout.strip() == 'cleanup ran'
