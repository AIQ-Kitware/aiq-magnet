"""
Small extensions for :mod:`lazy_loader`.

This module adds two lightweight capabilities that are useful in many packages:

1) **Module properties** (PEP 562): expose arbitrary computed attributes at the
   package level (e.g. ``yourpkg.plt``) without importing their dependencies
   until first access.

2) **Moved submodule backcompat**: keep ``import yourpkg.old_module`` working
   after moving it to a new location (without a stub file), and optionally
   emit a deprecation warning.

References:
    https://github.com/scientific-python/lazy-loader/issues/127

Example
-------
In ``yourpkg/__init__.py``:

.. code:: python

    from yourpkg.utils import lazy_loader_extensions as lle

    class __module_properties__:
        @property
        def some_property(self):
            return 1 + 1

        @property
        def heavy_thing(self):
            import heavy_dependency
            return heavy_dependency.Thing


    __getattr__, __dir__, __all__ = lle.attach(
        __name__,
        submodules={"submodule_a", "submodule_b"},
        submod_attrs={"submodule_a": ["SomeClass", "some_function"]},
        __module_properties__=__module_properties__,
        explicit=["some_property", "heavy_thing"],
        moved_modules={
            __name__ + ".old_module": __name__ + ".new.location.old_module",
        },
    )

Then:

.. code:: python

    import yourpkg
    yourpkg.some_property     # computed on first access, then cached
    yourpkg.heavy_thing       # imports heavy_dependency on first access

    import yourpkg.old_module # continues to work after move (and warns, if configured)

Notes
-----
- Module properties affect attribute access (e.g. ``yourpkg.some_property``).
- Moved-module backcompat targets import statements (e.g. ``import yourpkg.old_module``),
  which cannot be handled by module ``__getattr__`` alone.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import inspect
import sys
import warnings
from typing import Any, Mapping, Optional, Sequence, Tuple, Type
import lazy_loader


def _property_names(cls: Type) -> set[str]:
    """Return names of @property descriptors declared on a class."""
    names: set[str] = set()
    for name, obj in inspect.getmembers(cls):
        if isinstance(obj, property):
            names.add(name)
    return names
# ... existing imports ...


def _default_moved_warning(old_name: str, new_name: str) -> str:
    return (
        f"Module '{old_name}' has moved to '{new_name}'. "
        f"Please update imports."
    )


class _MovedModuleLoader(importlib.abc.Loader):
    def __init__(
        self,
        old_name: str,
        new_name: str,
        finder,  # _MovedModuleFinder instance (for warn-once bookkeeping)
    ):
        self.old_name = old_name
        self.new_name = new_name
        self.finder = finder

    def create_module(self, spec):
        mod = importlib.import_module(self.new_name)
        sys.modules[self.old_name] = mod

        if "." in self.old_name:
            parent_name, child = self.old_name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child, mod)
        return mod

    def exec_module(self, module):
        # Warn once per old module name
        if self.old_name in self.finder._warned:
            return
        self.finder._warned.add(self.old_name)

        warning_text = self.finder.warn.get(self.old_name)
        if warning_text:
            warnings.warn(
                warning_text,
                category=self.finder.category,
                stacklevel=3,
            )


class _MovedModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.moved: dict[str, str] = {}
        self.warn: dict[str, str] = {}
        self.category: type[Warning] = FutureWarning
        self._warned: set[str] = set()

    def update(
        self,
        moved_modules: Mapping[str, str],
        moved_module_warnings: Optional[Mapping[str, str]] = None,
        *,
        warning_category: type[Warning] = FutureWarning,
    ) -> None:
        self.moved.update(dict(moved_modules))
        self.category = warning_category

        # Auto-generate warnings if none provided
        if moved_module_warnings is None:
            for old, new in moved_modules.items():
                self.warn.setdefault(old, _default_moved_warning(old, new))
        else:
            self.warn.update(dict(moved_module_warnings))

    def find_spec(self, fullname: str, path, target=None):
        new_name = self.moved.get(fullname)
        if not new_name:
            return None
        loader = _MovedModuleLoader(fullname, new_name, self)
        return importlib.util.spec_from_loader(fullname, loader, is_package=False)


_FINDER_SINGLETON: Optional[_MovedModuleFinder] = None


def _ensure_finder() -> _MovedModuleFinder:
    global _FINDER_SINGLETON
    if _FINDER_SINGLETON is not None:
        return _FINDER_SINGLETON

    # Reuse an existing finder if already installed (e.g. reload scenarios).
    for f in sys.meta_path:
        if isinstance(f, _MovedModuleFinder):
            _FINDER_SINGLETON = f
            return f

    finder = _MovedModuleFinder()
    sys.meta_path.insert(0, finder)  # prefer our redirects early
    _FINDER_SINGLETON = finder
    return finder


def attach(
    package_name: str,
    *,
    submodules=None,
    submod_attrs=None,
    __module_properties__=None,
    explicit=None,
    moved_modules: Optional[Mapping[str, str]] = None,
    moved_module_warning_category: type[Warning] = FutureWarning,
) -> Tuple[Any, Any, Sequence[str]]:
    """
    Like :func:`lazy_loader.attach`, but optionally supports module properties and moved-module redirects.

    Args:
        package_name: Package/module name (typically ``__name__`` in your __init__.py).
        submodules, submod_attrs: Passed through to :func:`lazy_loader.attach`.
        __module_properties__: A class whose ``@property`` members are exposed as module-level attributes.
        explicit: Optional allowlist of property names to expose.
        moved_modules: Optional mapping ``{old_fqname: new_fqname}`` for moved submodules.
        moved_module_warning_category: Warning category for moved-module warnings.

    Returns:
        (__getattr__, __dir__, __all__)
    """

    base_getattr, __dir__, __all__ = lazy_loader.attach(
        package_name,
        submodules=submodules,
        submod_attrs=submod_attrs,
    )

    if moved_modules:
        finder = _ensure_finder()
        finder.update(
            moved_modules=moved_modules,
            moved_module_warnings=None,  # <-- auto-generate
            warning_category=moved_module_warning_category,
        )

    # 2) Wrap __getattr__ to serve module-level properties (warning logic belongs in properties).
    if __module_properties__ is None:
        return base_getattr, __dir__, __all__

    modprops = __module_properties__()
    prop_names = _property_names(__module_properties__)

    if explicit is not None:
        prop_names &= set(explicit)

    pkg = sys.modules[package_name]

    def __getattr__(name: str) -> Any:
        if name in prop_names:
            val = getattr(modprops, name)
            # Cache result so subsequent access is fast and stable
            pkg.__dict__[name] = val
            return val
        return base_getattr(name)

    # Keep __all__ consistent with newly exposed property names.
    __all__ = sorted(set(__all__) | set(prop_names))

    return __getattr__, __dir__, __all__
