"""
Forward to the Model CLI defined in :mod:`magnet.cli.main`
"""
from magnet.cli.main import __cli__


main = __cli__.main

if __name__ == '__main__':
    main()
