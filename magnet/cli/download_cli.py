#!/usr/bin/env python3
import scriptconfig as scfg


class DownloadModalCLI(scfg.ModalCLI):
    """
    Download precomputed results for different benchmarking backends.
    """
    __command__ = 'download'

    from magnet.backends.helm.download_helm_results import DownloadHelmConfig as helm

__cli__ = DownloadModalCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/aiq-magnet/magnet/cli/download_cli.py
        python -m magnet.cli.download_cli
    """
    __cli__.main()
