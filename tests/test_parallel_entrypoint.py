"""
``python -m magnet.evaluation`` must survive ``--jobs > 1``.

This has to run as a subprocess. The bug only exists when the module is
executed under the name ``__main__``, because that is what makes joblib's
loky backend pickle its classes by value along with their module globals --
including a loguru logger that owns an unpicklable file sink once
``setup_logging`` has run. An in-process test imports the module normally,
so the classes pickle by reference and the bug cannot reproduce.

The symptom is a bare ``PicklingError: Could not pickle the task to send it
to the workers``, with nothing naming the logger, so it is worth a
regression test rather than a comment.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CARD = REPO / 'magnet' / 'cards' / 'simple.yaml'


def _run(entrypoint, jobs, output_path):
    return subprocess.run(
        [sys.executable, '-m', *entrypoint, str(CARD),
         '--output_path', str(output_path), '--jobs', str(jobs)],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )


@pytest.mark.skipif(not CARD.exists(), reason='bundled demo card is absent')
@pytest.mark.parametrize('jobs', [1, 2])
def test_module_entrypoint_runs_at_any_job_count(jobs, tmp_path):
    proc = _run(['magnet.evaluation'], jobs, tmp_path / f'j{jobs}')
    assert 'PicklingError' not in proc.stderr, (
        f'--jobs {jobs} could not ship its work to the workers:\n'
        f'{proc.stderr[-2000:]}'
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


@pytest.mark.skipif(not CARD.exists(), reason='bundled demo card is absent')
def test_package_entrypoint_still_works(tmp_path):
    """The other way in, which was never affected -- kept so a fix to one
    entry point cannot quietly break the other."""
    proc = subprocess.run(
        [sys.executable, '-m', 'magnet', 'evaluate', str(CARD),
         '--output_path', str(tmp_path / 'pkg'), '--jobs', '2'],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
