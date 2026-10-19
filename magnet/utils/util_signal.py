"""Scoped signal handling for command-line entrypoints."""

import signal
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def signals_as_system_exit() -> Generator[None, None, None]:
    """Convert SIGTERM/SIGINT to SystemExit so Python cleanup can run.

    Uses conventional exit codes (128 + signal number) and restores prior
    handlers on exit. Like signal.signal(), this requires the main thread.
    """
    def exit_handler(signum, frame):
        raise SystemExit(128 + signum)

    signals = [signal.SIGTERM, signal.SIGINT]
    previous = {}
    try:
        for signum in signals:
            previous[signum] = signal.signal(signum, exit_handler)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
