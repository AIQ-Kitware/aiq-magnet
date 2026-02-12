"""magnet.backends.helm.helm_run_analysis

Single-run analysis utilities wrapped in an object.

Why this exists
--------------
``HelmRun`` (in :mod:`magnet.helm_outputs`) is intentionally a *reader*.
This module defines :class:`HelmRunAnalysis`, which *wraps* a ``HelmRun`` and
adds cached analyses / indices that make higher-level tasks (e.g. run diffs)
much easier to write.

Design goals (match notebook-style workflows)
--------------------------------------------
* Keep computations *lazy* and cache results.
* Keep the public API tight (a few high-value methods).
* Provide stable-ish identifiers where HELM uses dict-typed "names".

Notes
-----
* We primarily operate on the json view (``run.json``) for speed and
  robustness across HELM versions.
* We do **conservative** canonicalization for hashing: only strip known
  environment-specific fields like path strings.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import ubelt as ub

from magnet.backends.helm import helm_hashers
from magnet.backends.helm.helm_metrics import classify_metric, metric_family


class HelmRunAnalysis(ub.NiceRepr):
    """Wrap a :class:`~magnet.helm_outputs.HelmRun` with cached analyses.

    Parameters
    ----------
    run:
        The underlying run reader.
    name:
        Optional human-friendly label used in summaries.

    Example:
        >>> from magnet.backends.helm.helm_outputs import HelmRun
        >>> from magnet.backends.helm.helm_run_analysis import HelmRunAnalysis
        >>> run = HelmRun.demo()
        >>> ana = HelmRunAnalysis(run)
        >>> info = ana.summary_dict(level='lite')
        >>> assert 'run_spec_name' in info
        >>> print(f'info = {ub.urepr(info, nl=1)}')
        >>> print(ana.summary_text(level='page'))
    """

    def __init__(self, run, *, name: str | None = None):
        self.run = run
        self.name = name
        # Raw JSON endpoints (expensive I/O) are cached here
        self._raw_cache: dict[str, Any] = {}
        # Derived analyses / indices are cached here
        self._cache: dict[Any, Any] = {}

    def __nice__(self):
        return self.name or str(self.run.path.name)

    # --- Raw JSON getters (cached) ------------------------------------

    def run_spec(self) -> dict[str, Any]:
        return self._raw('run_spec', lambda: self.run.json.run_spec())

    def scenario(self) -> dict[str, Any]:
        return self._raw('scenario', lambda: self.run.json.scenario())

    def scenario_state(self) -> dict[str, Any]:
        return self._raw('scenario_state', lambda: self.run.json.scenario_state())

    def stats(self) -> list[dict[str, Any]]:
        return self._raw('stats', lambda: self.run.json.stats())

    def per_instance_stats(self) -> list[dict[str, Any]]:
        return self._raw('per_instance_stats', lambda: self.run.json.per_instance_stats())

    def _raw(self, key: str, factory):
        if key not in self._raw_cache:
            self._raw_cache[key] = factory()
        return self._raw_cache[key]

    # --- Summaries -----------------------------------------------------

    def summary_dict(self, *, level: str = 'lite') -> dict[str, Any]:
        """Return a small stable summary.

        Levels
        ------
        lite:
            one-row friendly fields
        """
        cache_key = ('summary_dict', level)
        if cache_key in self._cache:
            return self._cache[cache_key]

        spec = self.run_spec() or {}
        scenario_state = self.scenario_state() or {}

        out: dict[str, Any] = {
            'label': self.name,
            'path': str(self.run.path),
            'run_spec_name': spec.get('name', None),
            'scenario_name': (spec.get('scenario_spec', {}) or {}).get('class_name', None),
            'num_stats': len(self.stats()),
            'num_request_states': len((scenario_state.get('request_states', []) or [])),
            'num_per_instance_stats_rows': (
                len(self.per_instance_stats()) if (self.run.path / 'per_instance_stats.json').exists() else None
            ),
        }
        if level not in {'lite'}:
            raise KeyError(level)

        self._cache[cache_key] = out
        return out

    def summary_text(self, *, level: str = 'line') -> str:
        """Human-readable summary.

        Levels
        ------
        line:
            single-line summary
        page:
            multi-line, ~one-page summary
        """
        info = self.summary_dict(level='lite')
        label = info.get('label') or ub.Path(info['path']).name
        if level == 'line':
            return (
                f"{label} | spec={info.get('run_spec_name')} | "
                f"stats={info.get('num_stats')} | req={info.get('num_request_states')}"
            )
        if level == 'page':
            lines = []
            lines.append(f"Run: {label}")
            lines.append(f"  path: {info.get('path')}")
            lines.append(f"  run_spec_name: {info.get('run_spec_name')}")
            lines.append(f"  scenario: {info.get('scenario_name')}")
            lines.append(f"  num_stats: {info.get('num_stats')}")
            lines.append(f"  num_request_states: {info.get('num_request_states')}")
            lines.append(f"  num_per_instance_stats_rows: {info.get('num_per_instance_stats_rows')}")
            return "\n".join(lines)
        raise KeyError(level)

    # --- Stats: inventory + index -------------------------------------

    def stat_index(
        self,
        *,
        drop_zero_count: bool = True,
        require_mean: bool = False,
        short_hash: int = 16,
    ) -> dict[str, 'StatMeta']:
        """Map a readable stat-key -> :class:`StatMeta`.

        The key starts with the metric name (and hints like split/pert name) and
        ends with a short hash to keep it stable and disambiguated.
        """
        cache_key = ('stat_index', drop_zero_count, require_mean, short_hash)
        if cache_key in self._cache:
            return self._cache[cache_key]

        idx: dict[str, StatMeta] = {}
        for row in self.stats():
            count = int(row.get('count', 0) or 0)
            if drop_zero_count and count == 0:
                continue
            mean = _safe_float(row.get('mean', None))
            if require_mean and mean is None:
                continue

            name_obj = row.get('name', None)
            metric = name_obj.get('name', None) if isinstance(name_obj, dict) else None
            split = name_obj.get('split', None) if isinstance(name_obj, dict) else None
            pert_id = None
            if isinstance(name_obj, dict) and isinstance(name_obj.get('perturbation', None), dict):
                pert_id = helm_hashers.perturbation_id(name_obj['perturbation'], short_hash=short_hash)
            is_pert = pert_id is not None

            mclass, mpref = classify_metric(metric)
            fam = metric_family(metric)

            key = helm_hashers.stat_key(name_obj, short_hash=short_hash)
            idx[key] = StatMeta(
                key=key,
                metric=metric,
                split=split,
                is_perturbed=is_pert,
                pert_id=pert_id,
                family=fam,
                metric_class=mclass,
                matched_prefix=mpref,
                count=count,
                mean=mean,
                name_obj=name_obj,
                raw=row,
            )

        self._cache[cache_key] = idx
        return idx

    def stats_inventory(self, *, drop_zero_count: bool = False) -> dict[str, Counter]:
        """Lightweight histograms over ``stats.json`` for exploration."""
        cache_key = ('stats_inventory', drop_zero_count)
        if cache_key in self._cache:
            return self._cache[cache_key]

        hist: dict[str, Counter] = {
            'counts': Counter(),
            'perturbed': Counter(),
            'splits': Counter(),
            'family': Counter(),
            'metric_class': Counter(),
        }
        for row in self.stats():
            c = int(row.get('count', 0) or 0)
            hist['counts'][c] += 1
            if drop_zero_count and c == 0:
                continue
            name_obj = row.get('name', None)
            metric = name_obj.get('name', None) if isinstance(name_obj, dict) else None
            split = name_obj.get('split', None) if isinstance(name_obj, dict) else None
            is_pert = bool(isinstance(name_obj, dict) and name_obj.get('perturbation', None))
            hist['perturbed'][is_pert] += 1
            hist['splits'][split] += 1
            hist['family'][metric_family(metric)] += 1
            hist['metric_class'][classify_metric(metric)[0]] += 1

        self._cache[cache_key] = hist
        return hist

    # --- Requests + per-instance stats join ----------------------------

    def joined_instance_stat_table(self, *, assert_assumptions: bool = True, short_hash: int = 16):
        """Join per-instance stats to request_states.

        Returns one row **per per-instance stat**, with the corresponding
        request_state attached.

        Why this exists
        ---------------
        HELM's assets are not trivially zippable:

        * ``scenario_state()['request_states']`` is effectively "one row per
          evaluated instance variant" and may contain multiple rows with the
          same ``(instance_id, train_trial_index)`` when perturbations are
          present.
        * ``per_instance_stats.json`` often contains **multiple bundles** for
          the same ``(instance_id, train_trial_index)`` (e.g., bookkeeping
          metrics in one bundle and a single task metric in another).

        The join strategy
        -----------------
        1) Index request_states by base key ``(instance_id, train_trial_index)``
           and *perturbation id*, where the perturbation id is derived from
           ``request_state['instance']['perturbation']``.
        2) Merge per_instance_stats bundles by base key.
        3) For each stat in the merged bundle, match:
             - If ``stat['name']['perturbation']`` exists, use its perturbation id.
             - Otherwise match the unperturbed request_state.

        Assumptions (asserted when ``assert_assumptions=True``)
        -------------------------------------------------------
        * per_instance_stats has at least one row for every base request key.
        * For each (instance_id, train_trial_index, perturbation_id) there is
          exactly one request_state.
        * Every per-instance stat can be matched to exactly one request_state.
        """

        request_states = (self.scenario_state().get('request_states', []) or [])
        perinstance_stats = self.per_instance_stats() or []

        if assert_assumptions:
            assert isinstance(request_states, list), type(request_states)
            assert isinstance(perinstance_stats, list), type(perinstance_stats)
            # per_instance_stats is often *longer* because of duplicate bundles
            assert len(perinstance_stats) >= 0

        def base_key_from_rs(rs: Mapping[str, Any]):
            inst = rs.get('instance', {}) or {}
            return (inst.get('id', None), rs.get('train_trial_index', None))

        def pert_id_from_instance(inst: Mapping[str, Any]):
            pert = inst.get('perturbation', None)
            return helm_hashers.perturbation_id(pert, short_hash=short_hash)

        def base_key_from_pi(pi: Mapping[str, Any]):
            return (pi.get('instance_id', None), pi.get('train_trial_index', None))

        def pert_id_from_stat(stat: Mapping[str, Any]):
            name_obj = stat.get('name', None) or {}
            if isinstance(name_obj, dict):
                pert = name_obj.get('perturbation', None)
                return helm_hashers.perturbation_id(pert, short_hash=short_hash)
            return None

        # --- Index request_states by (id, tti) then perturbation id ---
        req_index: dict[tuple[Any, Any], dict[str | None, list[dict[str, Any]]]] = {}
        for rs in request_states:
            inst = rs.get('instance', {}) or {}
            bkey = base_key_from_rs(rs)
            pid = pert_id_from_instance(inst)
            req_index.setdefault(bkey, {}).setdefault(pid, []).append(rs)

        if assert_assumptions:
            dupes = []
            for bkey, by_pid in req_index.items():
                for pid, rows in by_pid.items():
                    if len(rows) != 1:
                        dupes.append((bkey, pid, len(rows)))
            assert not dupes, (
                "request_states has duplicate (instance_id, train_trial_index, perturbation_id) keys. "
                f"Example(s): {dupes[:10]}"
            )

        # --- Merge per_instance_stats bundles by (id, tti) ---
        merged_pi: dict[tuple[Any, Any], dict[str, Any]] = {}
        for pi in perinstance_stats:
            k = base_key_from_pi(pi)
            entry = merged_pi.setdefault(k, {'instance_id': k[0], 'train_trial_index': k[1], 'stats': []})
            entry['stats'].extend(pi.get('stats', []) or [])

        if assert_assumptions:
            req_base = set(req_index.keys())
            pi_base = set(merged_pi.keys())
            missing = sorted(req_base - pi_base)
            extra = sorted(pi_base - req_base)
            assert not missing, f"Missing per_instance_stats base keys for request_states. Example(s): {missing[:20]}"
            assert not extra, f"Extra per_instance_stats base keys without request_states. Example(s): {extra[:20]}"

        # --- Produce joined rows: one per stat ---
        joined_rows: list[dict[str, Any]] = []
        unmatched: list[tuple[Any, Any, str | None, str | None]] = []

        for bkey, pi in merged_pi.items():
            by_pid = req_index.get(bkey, {})
            for stat in pi.get('stats', []) or []:
                pid = pert_id_from_stat(stat)
                rs_rows = by_pid.get(pid, None)
                if rs_rows is None:
                    # Common failure mode: pid=None but only perturbed request_states exist.
                    metric = None
                    name_obj = stat.get('name', None)
                    if isinstance(name_obj, dict):
                        metric = name_obj.get('name', None)
                    unmatched.append((bkey[0], bkey[1], pid, metric))
                    continue
                rs = rs_rows[0]
                joined_rows.append({
                    'instance_id': bkey[0],
                    'train_trial_index': bkey[1],
                    'perturbation_id': pid,
                    'stat': stat,
                    'request_state': rs,
                })

        if assert_assumptions:
            assert not unmatched, (
                "Some per-instance stat rows could not be matched to request_states via perturbation id. "
                f"Example(s): {unmatched[:20]}"
            )

        return joined_rows


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
class StatMeta:
    """A compact, normalized view of a HELM stat row."""

    key: str
    metric: str | None
    split: str | None
    is_perturbed: bool
    pert_id: str | None
    family: str
    metric_class: str
    matched_prefix: str | None
    count: int
    mean: float | None
    name_obj: Any
    raw: Mapping[str, Any]
