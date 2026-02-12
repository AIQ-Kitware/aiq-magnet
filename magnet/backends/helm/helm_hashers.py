"""magnet.backends.helm.rundiff.helm_hashers

Centralized hashing helpers for HELM run analysis / comparison.

These functions provide:

* A deterministic stable hash (base36) for arbitrary Python structures.
* Semi-readable stable ids where a human-friendly prefix is spliced into the
  hash to make diffs easier to scan.
* Convenience helpers for HELM stat-name objects.

This module was extracted from ``magnet.backends.helm.rundiff.compare`` to
avoid duplicating / subtly changing canonicalization + hashing rules.

Implementation notes
--------------------
``ubelt.hash_data`` already normalizes dictionary key ordering, which is the
primary canonicalization requirement for HELM's nested stat-name dicts.

We intentionally avoid adding extra canonicalization logic here (e.g. list
sorting) because list ordering can be semantic. If you later decide to add
"deep canonicalization" rules, add them here so all callers stay consistent.
"""

from __future__ import annotations

from typing import Any

import ubelt as ub


def stable_hash36(obj: Any) -> str:
    """Deterministic base36 hash used throughout rundiff.

    This mirrors the previous private helper ``_stable_hash36``.
    """
    return ub.hash_data(obj, base=36, hasher='sha256')


def nice_hash_id(obj: Any, *, rawstr: str | None = None, keep_prefix: int = 25) -> str:
    """Semi-readable stable id.

    The returned id is always the same length as the underlying hash, but we
    splice in a readable prefix (metric/split/...) to make diffs easier to
    scan.

    This mirrors the previous private helper ``_nice_hash_id``.

    Args:
        obj: Object to hash.
        rawstr: Human-readable prefix content to splice in.
        keep_prefix: If ``rawstr`` is longer than the hash, keep this many
            prefix characters.

    Returns:
        str: stable id string.
    """
    if rawstr is None:
        rawstr = ub.urepr(obj, compact=1, nl=0, nobr=1)
    hashstr = stable_hash36(obj)
    rawstr = rawstr.replace(' ', '')
    rawlen = len(rawstr)
    hashlen = len(hashstr)
    if rawlen < hashlen:
        return rawstr + hashstr[:-rawlen]
    else:
        return rawstr[:keep_prefix] + hashstr[:-keep_prefix]


def stat_name_id(name_obj: Any, *, count: Any = None) -> str:
    """Stable, semi-readable id for a HELM stat-name dict.

    Mirrors the previous helper in compare.py.

    Args:
        name_obj: The stat ``name`` object (typically a dict).
        count: Optional count to incorporate into the id.

    Returns:
        str: stable semi-readable id.
    """
    if not isinstance(name_obj, dict):
        raw = f"invalid_name,{ub.urepr(name_obj, compact=1, nl=0, nobr=1)},"
        obj = ('invalid_name', name_obj, count)
        return nice_hash_id(obj, rawstr=raw)

    # Prefer `name` as the human-readable base.
    base = name_obj.get('name', 'nobasename')
    rest = ub.udict(name_obj) - {'name'}
    compact = ub.urepr(rest, compact=1, nobr=1, nl=0)
    if count is None:
        raw = f"{base},{compact},"
        obj = {'name': name_obj}
    else:
        raw = f"{base},{compact},count={count},"
        obj = {'name': name_obj, 'count': count}
    return nice_hash_id(obj, rawstr=raw)


def row_id(row: Any, *, hint: str = 'row') -> str:
    """Stable-ish id for arbitrary rows (e.g. per-instance rows)."""
    raw = f"{hint},"
    return nice_hash_id(row, rawstr=raw)


def nice_stat_name_id(name_obj, *, drop_name_file_path=True):
    """
    Like stat_name_id, but guarantees the returned id begins with the metric name
    (and includes split / sub_split / perturbation name in the human prefix),
    while still being stable via hashing the canonicalized object.
    """
    import copy

    obj = copy.deepcopy(name_obj)

    # Optional: strip unstable path payloads (this is the one that bites HELM).
    if drop_name_file_path:
        try:
            pert = obj.get("perturbation", None)
            if isinstance(pert, dict):
                pert.pop("name_file_path", None)
        except Exception:
            pass

    metric = None
    if isinstance(obj, dict):
        metric = obj.get("name", None) or obj.get("metric", None)

    split = obj.get("split", None) if isinstance(obj, dict) else None
    sub_split = obj.get("sub_split", None) if isinstance(obj, dict) else None
    pert_name = None
    if isinstance(obj, dict):
        pert = obj.get("perturbation", None)
        if isinstance(pert, dict):
            pert_name = pert.get("name", None)

    # Nice, human prefix (stable + readable)
    prefix_parts = []
    if metric:
        prefix_parts.append(str(metric))
    if split:
        prefix_parts.append(f"split={split}")
    if sub_split:
        prefix_parts.append(f"sub={sub_split}")
    if pert_name:
        prefix_parts.append(f"pert={pert_name}")
    prefix = ",".join(prefix_parts) if prefix_parts else "stat"

    # Hash is still computed from the *full* canonicalized object
    return nice_hash_id(obj, prefix=prefix)
