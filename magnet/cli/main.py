"""
The top level Magnet CLI
"""
import scriptconfig as scfg
from magnet.cli.download_cli import DownloadModalCLI
from magnet import __version__

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('foo', default='1', required=False)
# PortedParser = scfg.DataConfig.cls_from_argparse(parser, name='blagh')


# def main(cmdline=None, **kw):
#     args = PortedParser.cli(cmdline=cmdline, data=kw, strict=True, special_options=False)

# PortedParser.main = main
# parser.parse_args()


class MagnetCLI(scfg.ModalCLI):
    """
    Top level magnet modal CLI
    """
    # __version__ = __version__
    ...


# MagnetCLI.register(DownloadModalCLI)
MagnetCLI.register(DownloadModalCLI, command='download')
# MagnetCLI.register(PortedParser, command='blagh')


__cli__ = MagnetCLI


if __name__ == '__main__':
    """
    CommandLine:
        python -m magnet.cli.main
    """
    __cli__.main()
