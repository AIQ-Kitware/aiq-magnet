import os
import sys

import ubelt as ub
from loguru import logger


def setup_logging(
    verbose: bool = False, log_out_dir: str | None = None
) -> None:
    """Configure loguru logging.

    - Default level is INFO, or DEBUG when --verbose is set.
    - You may override via MAGNET_LOG_LEVEL (e.g. DEBUG, INFO, WARNING).
    """

    level = os.environ.get('MAGNET_LOG_LEVEL')
    if not level:
        level = 'DEBUG' if verbose else 'INFO'

    logger.remove()

    if log_out_dir:
        ub.Path(log_out_dir).ensuredir()
        logger.add(
            f'{log_out_dir}/log',
            level=level,
            enqueue=True,
        )

    try:
        from rich.console import Console
        from rich.logging import RichHandler

        # Create a console specifically for stderr
        error_console = Console(stderr=True)
        logger.add(
            RichHandler(
                console=error_console,  # Force Rich to use stderr
                markup=True,
                rich_tracebacks=True,
            ),
            level=level,
            format='{message}',
            backtrace=False,
            diagnose=False,
        )

    except ImportError:
        # Fallback to standard loguru output if rich is not available
        logger.add(
            sys.stderr,
            level=level,
            colorize=True,
            backtrace=False,
            diagnose=False,
        )
