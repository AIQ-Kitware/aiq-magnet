"""Read what kwdagger scheduled a node directory with (its ``invoke.sh``).

kwdagger renders each node's command into ``invoke.sh`` in the node directory.
That is the record of *how* the directory was scheduled, whichever recipe,
matrix, or container/lease wrapper produced it. The aiq-magnet-evals loaders
pin the files a node wrote (``evaluation.json``, ``comparison.json``) to that
record instead of trusting them to say which inputs, run, or projection they
used.

These are integrity checks against accidental or partial edits of one file.
Someone who rewrites ``invoke.sh`` and the outputs consistently is not
detected: nothing here is signed.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any


def invocation_args(node_dpath: Any) -> dict[str, dict[str, str]]:
    """``{module: {flag: value}}`` for each ``python -m <module> --flag=value`` in ``invoke.sh``.

    The first occurrence of a module wins. Returns ``{}`` when there is no
    readable record.
    """
    try:
        text = (Path(node_dpath) / 'invoke.sh').read_text()
    except OSError:
        return {}
    try:
        tokens = shlex.split(text.replace('\\\n', ' '), comments=True)
    except ValueError:
        return {}
    found: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for index, token in enumerate(tokens):
        if token == '-m' and index + 1 < len(tokens):
            module = tokens[index + 1]
            if module in found:
                current = None
            else:
                current = found[module] = {}
            continue
        if current is not None and token.startswith('--') and '=' in token:
            flag, value = token[2:].split('=', 1)
            current.setdefault(flag, value)
        elif current is not None and token in {'||', '&&', ';', '|'}:
            current = None
    return found


def localize(rendered: str | os.PathLike[str], rendered_node_dpath: str | os.PathLike[str],
             node_dpath: str | os.PathLike[str]) -> Path:
    """Map a path as rendered at scheduling time onto this node directory's tree.

    kwdagger renders paths under its root, relative when the root was given
    relative to the scheduling directory. Node directories are
    ``<root>/<node>/<id>``, so a rendered path under the rendered root maps onto
    the actual root the loader found the node in. Absolute paths are kept.
    """
    path = Path(rendered)
    if path.is_absolute():
        return path
    rendered_root = Path(rendered_node_dpath).parent.parent
    try:
        relative = path.relative_to(rendered_root)
    except ValueError:
        return Path.cwd() / path
    return Path(node_dpath).parent.parent / relative


def same_path(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    return Path(a).resolve() == Path(b).resolve()
