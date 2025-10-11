"""
The top level Magnet CLI

Example:
    >>> # Test that help works for each subcli
    >>> from magnet.cli.main import *  # NOQA
    >>> MagnetCLI.main(argv=['--help'], _noexit=True)
    >>> MagnetCLI.main(argv=['download', '--help'], _noexit=True)
    >>> MagnetCLI.main(argv=['download', 'helm', '--help'], _noexit=True)
    >>> # Test version works
    >>> MagnetCLI.main(argv=['--version'])
"""
import scriptconfig as scfg
from magnet.cli.download_cli import DownloadModalCLI
from magnet import __version__


class MagnetCLI(scfg.ModalCLI):
    """
    Top level magnet modal CLI
    """
    __version__ = __version__


MagnetCLI.register(DownloadModalCLI, command='download')


__cli__ = MagnetCLI


if __name__ == '__main__':
    """
    CommandLine:
        python -m magnet.cli.main
    """
    __cli__.main()
