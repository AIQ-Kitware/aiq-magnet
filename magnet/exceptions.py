from types import ModuleType


class SymbolResolutionError(ValueError):
    """
    Used when symbols fail to evaluate
    """


class MissingOptionalDependency(ImportError):
    """
    Raised when a module needs a package that one of magnet's extras
    provides and it is not installed. The message names the extra.
    """


def require_optional(
    module: str, extra: str, purpose: str | None = None
) -> ModuleType:
    """
    Import ``module`` or raise :class:`MissingOptionalDependency` naming the
    extra that provides it.

    Args:
        module (str): the top-level module the caller is about to import.
        extra (str): the ``aiq-magnet[<extra>]`` extra that installs it.
        purpose (str | None): what the caller does with it, for the message.

    Returns:
        ModuleType: the imported module.

    Example:
        >>> from magnet.exceptions import require_optional
        >>> require_optional('json', 'helm').__name__
        'json'
        >>> import pytest
        >>> with pytest.raises(MissingOptionalDependency):
        ...     require_optional('no_such_module_xyz', 'helm')
    """
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as ex:
        what = f' ({purpose})' if purpose else ''
        raise MissingOptionalDependency(
            f"{module!r} is not installed{what}. "
            f"Install it with: pip install 'aiq-magnet[{extra}]'"
        ) from ex
