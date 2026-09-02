"""
Shared setup so the suite runs on a minimal install.

MAGNET's core installs without the `leasing` extra, and `magnet.leasing`
refuses to render an `infer-stack run` prefix when no `infer-stack` executable
is on PATH -- a courtesy check, so the failure is a clear message at render
time rather than `infer-stack: command not found` in a job log after the DAG
has been submitted.

That check is right for a caller and beside the point for these tests: they
assert what *string* gets rendered, and never run it. Without infer-stack
installed it turned forty-two of them red for a reason none of them is about.

So an executable is put on PATH for the session when there is not one already.
A real install is left alone -- `shutil.which` finds the real thing and this
fixture does nothing -- so nothing here can mask a broken lease on a machine
that can actually take one. The tests that need infer-stack to *behave* (see
`test_gpu_allow_list.py`) still build their own stub and run against it.
"""

import shutil
import textwrap

import pytest


@pytest.fixture(scope='session', autouse=True)
def _infer_stack_on_path(tmp_path_factory):
    """Guarantee an `infer-stack` on PATH, without replacing a real one."""
    if shutil.which('infer-stack') is not None:
        yield
        return

    dpath = tmp_path_factory.mktemp('leasing-stub')
    stub = dpath / 'infer-stack'
    stub.write_text(textwrap.dedent('''\
        #!/bin/bash
        # Present so `magnet.leasing` will render a prefix. Rendering is what
        # the suite tests; nothing here is expected to be executed.
        echo "stub infer-stack: not meant to run" >&2
        exit 1
    '''))
    stub.chmod(0o755)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv('PATH', f'{dpath}:{shutil.os.environ["PATH"]}')
        yield
