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
from magnet.cli.download_cli import DownloadModalCLI
from magnet.evaluation import EvaluationConfig
from magnet.evaluation_new import NewEvaluationCLI
from magnet import __version__


class MagnetCLI(kwconf.ModalCLI):
    """
    Top level MAGNET CLI
    """
    __version__ = __version__


MagnetCLI.register(DownloadModalCLI, command='download')
# The historical evaluator gets an explicit migration name while `evaluate`
# remains a compatibility alias until the new evaluator takes that name.
MagnetCLI.register(
    EvaluationConfig, command='evaluate_legacy', alias=['evaluate']
)
MagnetCLI.register(NewEvaluationCLI, command='evaluate_new')


__cli__ = MagnetCLI


if __name__ == '__main__':
    """
    CommandLine:
        python -m magnet.cli.main
    """
    __cli__.main()
