"""
The top level Magnet CLI
"""
import scriptconfig as scfg
from magnet.cli.download_cli import DownloadModalCLI


class MagnetCLI(scfg.ModalCLI):
    ...


# MagnetCLI.register(DownloadModalCLI)
MagnetCLI.register(DownloadModalCLI, command='download')


__cli__ = MagnetCLI
