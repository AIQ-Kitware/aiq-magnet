"""
The top level MAGNET CLI

Example:
    >>> # Test that help works for each subcli
    >>> from magnet.cli.main import *  # NOQA
    >>> MagnetCLI.main(argv=['--help'], _noexit=True)
    >>> MagnetCLI.main(argv=['download', '--help'], _noexit=True)
    >>> MagnetCLI.main(argv=['download', 'helm', '--help'], _noexit=True)
    >>> MagnetCLI.main(argv=['evaluate', '--help'], _noexit=True)
    >>> MagnetCLI.main(argv=['evaluate_legacy', '--help'], _noexit=True)
    >>> MagnetCLI.main(argv=['evaluate_new', '--help'], _noexit=True)
    >>> # Test version works
    >>> MagnetCLI.main(argv=['--version'])
"""

import kwconf

from magnet import __version__
from magnet.cli.download_cli import DownloadModalCLI
from magnet.cli.results_cli import ResultsCLI
from magnet.evaluation import EvaluationConfig
from magnet.evaluation_new import NewEvaluationCLI


class MagnetCLI(kwconf.ModalCLI):
    """
    Top level MAGNET CLI
    """

    __version__ = __version__

    download = DownloadModalCLI
    # Preserve the historical name during migration to the new evaluator.
    evaluate_legacy = kwconf.ModalValue(EvaluationConfig, alias=['evaluate'])
    evaluate_new = NewEvaluationCLI
    results = ResultsCLI


__cli__ = MagnetCLI


if __name__ == '__main__':
    """
    CommandLine:
        python -m magnet.cli.main
    """
    __cli__.main()
