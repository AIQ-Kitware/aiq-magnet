"""magnet.backends.helm.helm_summarize_mixin

Lightweight, JSON-oriented helpers inspired by HELM's summarization code.

Why a mixin?
------------
You asked to keep "the ability for HelmRunAnalysis to do the things the
original HELM code was doing", but you also want to keep that logic
conceptually separate while you're still learning HELM's conventions.

This file provides a small mixin that mirrors the *most useful semantics* of
``helm.benchmark.presentation.summarize`` without depending on HELM's
presentation layer (tables / latex) or dataclasses.

We operate on the raw JSON rows (dicts) produced by HELM.

Key behaviors preserved
-----------------------
* Matcher-based stat selection (similar to HELM's ``MetricNameMatcher``).
* Quasi-exact-match fallback to exact-match.
* If matcher does not specify ``sub_split``, aggregate across all sub-splits.
  (HELM does this by rewriting name.sub_split=None and merging stats.)
* "Cell" semantics to distinguish:
    - no matching metric
    - matching metric but count == 0 (mean undefined)
    - a valid numeric value
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import ubelt as ub


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        y = float(x)
        if math.isnan(y):
            return None
        return y
    except Exception:
        return None


@dataclass(frozen=True)
class MetricNameMatcherLite:
    """A minimal matcher for HELM's JSON ``stat['name']`` dict.

    Fields left as ``None`` are treated as wildcards.

    Notes
    -----
    If ``sub_split`` is ``None`` we *match* any sub_split, but callers may
    choose to aggregate across those sub_splits.
    """

    name: str
    split: str | None = None
    sub_split: str | None = None
    perturbation: str | None = None

    def matches(self, name_obj: Any) -> bool:
        if not isinstance(name_obj, dict):
            return False
        if name_obj.get('name', None) != self.name:
            return False
        if self.split is not None and name_obj.get('split', None) != self.split:
            return False
        if self.sub_split is not None and name_obj.get('sub_split', None) != self.sub_split:
            return False
        if self.perturbation is not None:
            p = name_obj.get('perturbation', None)
            if not isinstance(p, dict):
                return False
            if p.get('name', None) != self.perturbation:
                return False
        return True


def _merge_mean_count(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge JSON stat rows (count-weighted mean).

    HELM uses ``Stat.merge`` and keeps many more fields. For our purposes,
    count-weighted mean is the key behavior needed for summaries.
    """
    total_count = 0
    total_sum = 0.0
    last: Mapping[str, Any] | None = None
    for r in rows:
        last = r
        c = int(r.get('count', 0) or 0)
        m = _safe_float(r.get('mean', None))
        if c > 0 and m is not None:
            total_sum += m * c
            total_count += c
        else:
            total_count += c

    merged: dict[str, Any] = {}
    if last is not None:
        merged.update(dict(last))
    merged['count'] = total_count
    merged['mean'] = (total_sum / total_count) if total_count > 0 else None
    return merged


class HelmSummarizeMixin:
    """Mixin that assumes ``self.stats() -> list[dict]`` exists."""

    # --- Stat selection -------------------------------------------------

    def matching_stats(self, matcher: MetricNameMatcherLite | Mapping[str, Any]) -> list[dict[str, Any]]:
        if isinstance(matcher, Mapping):
            matcher = MetricNameMatcherLite(**dict(matcher))
        out: list[dict[str, Any]] = []
        for row in self.stats():
            if matcher.matches(row.get('name', None)):
                out.append(row)
        return out

    def get_unique_stat_by_matcher(
        self,
        matcher: MetricNameMatcherLite | Mapping[str, Any],
        *,
        quasi_exact_match_fallback: bool = True,
        aggregate_subsplits: bool = True,
    ) -> dict[str, Any] | None:
        """Return the single matching stat (or ``None``).

        Mirrors the most useful behavior of HELM's
        ``helm.benchmark.presentation.summarize.get_unique_stat_by_matcher``.

        Raises
        ------
        KeyError
            If more than one unique stat remains after optional aggregation.
        """
        if isinstance(matcher, Mapping):
            matcher = MetricNameMatcherLite(**dict(matcher))

        matching = self.matching_stats(matcher)
        if not matching:
            if quasi_exact_match_fallback and matcher.name == 'quasi_exact_match':
                alt = MetricNameMatcherLite(
                    name='exact_match',
                    split=matcher.split,
                    sub_split=matcher.sub_split,
                    perturbation=matcher.perturbation,
                )
                matching = self.matching_stats(alt)
                if not matching:
                    return None
            else:
                return None

        if aggregate_subsplits and matcher.sub_split is None:
            buckets: dict[str, list[dict[str, Any]]] = {}
            for row in matching:
                name_obj = row.get('name', None)
                if not isinstance(name_obj, dict):
                    k = ub.hash_data(('invalid-name', name_obj), base=36)
                else:
                    # Group by everything except sub_split
                    n2 = dict(name_obj)
                    n2['sub_split'] = None
                    k = ub.hash_data(n2, base=36)
                buckets.setdefault(k, []).append(row)

            merged_rows: list[dict[str, Any]] = []
            for _k, bucket in buckets.items():
                merged = _merge_mean_count(bucket)
                name_obj = merged.get('name', None)
                if isinstance(name_obj, dict):
                    name_obj = dict(name_obj)
                    name_obj['sub_split'] = None
                    merged['name'] = name_obj
                merged_rows.append(merged)
            matching = merged_rows

        if len(matching) != 1:
            raise KeyError(f"Matcher {matcher!r} matched {len(matching)} stats")
        return matching[0]

    def describe_stat_cell(
        self,
        matcher: MetricNameMatcherLite | Mapping[str, Any],
        *,
        quasi_exact_match_fallback: bool = True,
    ) -> dict[str, Any]:
        """Return a lightweight, Cell-like description for a matcher."""
        try:
            stat = self.get_unique_stat_by_matcher(
                matcher,
                quasi_exact_match_fallback=quasi_exact_match_fallback,
                aggregate_subsplits=True,
            )
        except KeyError as ex:
            return {'value': None, 'case': 'ambiguous', 'description': str(ex), 'stat': None}

        if stat is None:
            return {'value': None, 'case': 'no_match', 'description': 'No matching metrics', 'stat': None}

        count = int(stat.get('count', 0) or 0)
        mean = _safe_float(stat.get('mean', None))
        if count == 0 or mean is None:
            return {
                'value': None,
                'case': 'count_zero',
                'description': 'Matching metrics, but count == 0 (mean undefined)',
                'stat': stat,
            }
        return {'value': mean, 'case': 'ok', 'description': f'count={count}', 'stat': stat}
