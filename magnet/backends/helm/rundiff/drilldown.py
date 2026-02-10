"""
Optional drill-down utilities (instance-level diffs).

This is intentionally separate so the main compare loop stays fast/light.

Typical usage:
  - if a run pair looks suspicious (core mismatch), call these helpers
    to diff per-instance stats / predictions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import ubelt as ub

def index_per_instance_stats(per_instance_stats: List[Dict[str, Any]], *, key_fields=("instance_id",)):
    """
    Build a stable lookup for per-instance stats.

    You may need to adjust `key_fields` depending on the scenario.
    """
    lut = {}
    for row in per_instance_stats:
        key = tuple(row.get(f, None) for f in key_fields)
        lut[key] = row
    return lut

def diff_per_instance_stats(A: List[Dict[str, Any]], B: List[Dict[str, Any]], *, key_fields=("instance_id",)):
    """
    Returns:
      - onlyA_keys
      - onlyB_keys
      - mismatched_keys (values differ)
    """
    lutA = index_per_instance_stats(A, key_fields=key_fields)
    lutB = index_per_instance_stats(B, key_fields=key_fields)

    keysA = set(lutA)
    keysB = set(lutB)
    isect = keysA & keysB
    onlyA = keysA - keysB
    onlyB = keysB - keysA

    mism = []
    for k in isect:
        a = lutA[k]
        b = lutB[k]
        if ub.hash_data(a, base=36) != ub.hash_data(b, base=36):
            mism.append(k)

    return {
        "nA": len(keysA),
        "nB": len(keysB),
        "onlyA": sorted(onlyA),
        "onlyB": sorted(onlyB),
        "mismatched": mism,
    }
