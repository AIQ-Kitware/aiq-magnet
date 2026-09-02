"""
Keep test collection working on a minimal install.

MAGNET's core installs without HELM: `crfm-helm` and `scikit-learn` come from
the ``helm`` extra. The modules that need them say so at import time, raising
:class:`~magnet.exceptions.MissingOptionalDependency` with the extra named.

That is the right behaviour for a caller and the wrong one for a collector.
``--xdoctest`` imports every module in the package to find its doctests, and it
reports any import failure as an error -- so on a minimal install the whole
`magnet/backends/helm/` tree failed collection, and `pytest <installed magnet>`
went red for a reason that is not a defect. The test *files* were already
guarded with ``pytest.importorskip``; nothing guarded the package's own
doctests, which is what the sdist job runs.

So: a module that cannot be imported because an extra is missing is not
collected. Anything else that fails to import still is, and still errors --
this hides a missing optional dependency, not a broken module.

Deciding by attempting the import, rather than listing paths, means a module
added to an extra later is covered without anyone remembering this file. On a
full install nothing is skipped and the import is one xdoctest was about to do
anyway.
"""

import importlib
import pathlib

from magnet.exceptions import MissingOptionalDependency

_PACKAGE_ROOT = pathlib.Path(__file__).parent


def _module_name(path: pathlib.Path) -> str | None:
    """Dotted name for a file inside this package, or None if it is not one."""
    try:
        relative = path.relative_to(_PACKAGE_ROOT)
    except ValueError:
        return None
    if path.suffix != '.py' or path.name == 'conftest.py':
        return None
    parts = list(relative.with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join(['magnet', *parts])


def pytest_ignore_collect(collection_path, config):
    """Skip modules whose only problem is an uninstalled extra."""
    name = _module_name(pathlib.Path(str(collection_path)))
    if name is None:
        return None
    try:
        importlib.import_module(name)
    except MissingOptionalDependency:
        return True
    except Exception:
        # Not our business: let the collector import it and report properly.
        return None
    return None
