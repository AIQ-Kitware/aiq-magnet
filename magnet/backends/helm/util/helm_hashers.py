"""magnet.backends.helm.helm_hashers

NOTE: this might be over-engineering. Treat hashes as internal only for now.

Centralized hashing helpers for HELM run analysis / comparison.

These functions provide:

* A deterministic stable hash (base36) for arbitrary Python structures.
* Semi-readable stable ids where a human-friendly prefix is spliced into the
  hash to make diffs easier to scan.
* Convenience helpers for HELM stat-name objects.

This module was extracted from ``magnet.backends.helm.compare`` to
avoid duplicating / subtly changing canonicalization + hashing rules.

Implementation notes
--------------------
``ubelt.hash_data`` already normalizes dictionary key ordering, which is the
primary canonicalization requirement for HELM's nested stat-name dicts.

We intentionally avoid adding extra canonicalization logic here (e.g. list
sorting) because list ordering can be semantic. If you later decide to add
"deep canonicalization" rules, add them here so all callers stay consistent.

Example:
     >>> from magnet.backends.helm.util import helm_hashers
     >>> name_obj = {'name': 'num_bytes', 'split': 'valid', 'perturbation': {'name': 'dialect', 'prob': 1.0}}
     >>> key = helm_hashers.stat_key(name_obj)
     >>> assert key.startswith('num_bytes,split=valid,pert=dialect')
     >>> pid = helm_hashers.perturbation_id(name_obj['perturbation'])
     >>> assert pid.startswith('dialect')
"""
from __future__ import annotations
from typing import Any
import ubelt as ub


def stable_hash36(obj: Any) -> str:
    """Deterministic base36 hash used throughout """
    # ub.hash_data already normalizes dict key ordering.
    return ub.hash_data(obj, base=36, hasher='sha256')


# --- Canonicalization -------------------------------------------------------

_DROP_KEYS_DEFAULT = {
    # HELM perturbations may embed environment-specific paths.
    'name_file_path',
    'mapping_file_path',
}


def canonicalize_for_hashing(obj: Any, *, drop_keys: set[str] | None = None) -> Any:
    """Canonicalize *conservatively* for stable hashing.

    We do **not** reorder lists or otherwise change semantics.
    We only (optionally) remove known environment-specific keys.
    """
    if drop_keys is None:
        drop_keys = _DROP_KEYS_DEFAULT
    if isinstance(obj, dict):
        return {k: canonicalize_for_hashing(v, drop_keys=drop_keys) for k, v in obj.items() if k not in drop_keys}
    if isinstance(obj, list):
        return [canonicalize_for_hashing(v, drop_keys=drop_keys) for v in obj]
    if isinstance(obj, tuple):
        return tuple(canonicalize_for_hashing(v, drop_keys=drop_keys) for v in obj)
    return obj


def nice_hash_id(obj: Any, *, rawstr: str, keep_prefix: int = 25) -> str:
    """Semi-readable stable id (legacy style).

    This matches the older behavior from ``compare.py``: the returned string has
    the same length as the underlying hash, but we splice in readable text.
    """
    hashstr = stable_hash36(obj)
    rawstr = rawstr.replace(' ', '')
    rawlen = len(rawstr)
    hashlen = len(hashstr)
    if rawlen < hashlen:
        return rawstr + hashstr[:-rawlen]
    return rawstr[:keep_prefix] + hashstr[:-keep_prefix]


def prefixed_hash_id(
    obj: Any,
    *,
    prefix: str,
    short_hash: int = 16,
    drop_keys: set[str] | None = None,
) -> str:
    """Readable key that *starts with* a prefix and ends with a short hash.

    This is the conservative "nice" identifier you described: human hint first,
    then a stable hash of the canonicalized object.

    The goal is legibility (for debugging) while retaining near-uniqueness.
    """
    canon = canonicalize_for_hashing(obj, drop_keys=drop_keys)
    h = stable_hash36(canon)[:short_hash]
    prefix2 = prefix.replace(' ', '')
    return f"{prefix2}::{h}"


def _compact_hint(obj: Any, *, maxlen: int = 70) -> str:
    """Compact one-line representation used in readable prefixes."""
    if obj is None:
        return ''
    try:
        text = ub.urepr(obj, compact=1, nl=0, nobr=1)
    except Exception:
        text = str(obj)
    text = text.replace(' ', '')
    if len(text) > maxlen:
        text = text[:maxlen - 3] + '...'
    return text


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


# --- Higher-level HELM-specific ids ----------------------------------------

def perturbation_id(pert: Any, *, short_hash: int = 16) -> str | None:
    """Stable-ish, readable id for a HELM perturbation dict.

    Returns ``None`` when the input is falsy / absent.
    """
    if not pert:
        return None
    if not isinstance(pert, dict):
        return prefixed_hash_id(pert, prefix='pert', short_hash=short_hash)
    name = pert.get('name', None) or 'pert'
    # Put the name up front, include a compact hint, and hash the rest.
    rest = ub.udict(pert) - {'name'}
    rest_canon = canonicalize_for_hashing(rest)
    hint = _compact_hint(rest_canon)
    prefix = str(name)
    if hint and hint != '{}':
        prefix = f"{prefix},{hint}"
    return prefixed_hash_id(rest_canon, prefix=prefix, short_hash=short_hash)


def stat_key(name_obj: Any, *, count: Any = None, short_hash: int = 16) -> str:
    """Readable key for HELM ``stat['name']`` dicts.

    This is what you originally wanted for debugging: keys begin with the
    metric name and include key selectors (split/sub_split/perturbation), while
    the end contains a short stable hash of the full object.
    """
    if not isinstance(name_obj, dict):
        prefix = "invalid_name"
        if count is not None:
            prefix += f",count={count}"
        return prefixed_hash_id((name_obj, count), prefix=prefix, short_hash=short_hash)

    metric = name_obj.get('name', None) or 'stat'
    split = name_obj.get('split', None)
    sub = name_obj.get('sub_split', None)
    pert = None
    if isinstance(name_obj.get('perturbation', None), dict):
        pert = perturbation_id(name_obj['perturbation'], short_hash=short_hash)

    # Include a compact representation of any *extra* selectors besides the
    # common ones we already print explicitly (split/sub_split/perturbation).
    rest = ub.udict(name_obj) - {'name', 'split', 'sub_split', 'perturbation'}
    rest_canon = canonicalize_for_hashing(rest)
    rest_hint = _compact_hint(rest_canon)

    parts = [str(metric)]
    if rest_hint and rest_hint != '{}':
        parts.append(rest_hint)
    if split is not None:
        parts.append(f"split={split}")
    if sub is not None:
        parts.append(f"sub={sub}")
    if pert is not None:
        # keep the readable head of the perturbation id (already includes a hint)
        parts.append(f"pert={pert.split('::', 1)[0]}")
    if count is not None:
        parts.append(f"count={count}")
    prefix = ','.join(parts)

    # Hash the canonicalized object (including perturbation dict) so variants
    # with same name but different args still disambiguate.
    payload = {'name': name_obj, 'count': count} if count is not None else {'name': name_obj}
    return prefixed_hash_id(payload, prefix=prefix, short_hash=short_hash)
