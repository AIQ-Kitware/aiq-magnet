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
        >>> info = ana.summary_dict(level=10)
        >>> assert 'run_spec_name' in info
        >>> ana.summary(level=1)
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

    def summary_dict(self,  *args, **kwargs) -> dict[str, Any]:
        # Hack for now while developing. TODO: move the implementation here and
        # fix the signature.
        return summary_dict(self, *args, **kwargs)

    def summary(self, *args, **kwargs):
        # Hack for now while developing. TODO: move the implementation here and
        # fix the signature.
        return summary(self, *args, **kwargs)

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
        request_states = self.scenario_state().get("request_states", []) or []
        perinstance_stats = self.per_instance_stats() or []
        tbl = JoinedInstanceStatTable(request_states, perinstance_stats, short_hash=short_hash)
        if assert_assumptions:
            tbl.assert_assumptions()
        return tbl


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


def _coerce_int(x: Any) -> int | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return None


def _nice_perturbation_id(pert: Any, *, short_hash: int = 12) -> str | None:
    """
    Conservative “nice” perturbation id:
    - None if pert is falsy / not a dict
    - prefix = pert['name'] if present
    - suffix = stable hash of canonicalized dict
    """
    if not isinstance(pert, dict) or not pert:
        return None
    name = pert.get("name", "pert")
    # Strip known unstable payloads if present (optional and conservative)
    canon = ub.udict(pert).copy()
    canon.pop("mapping_file_path", None)
    canon.pop("name_file_path", None)
    h = helm_hashers.stable_hash36(canon)[:short_hash]
    return f"{name}~{h}"


@dataclass(frozen=True, slots=True)
class InstanceVariantKey:
    """Identifies a specific evaluated variant of an instance."""
    instance_id: str | None
    train_trial_index: int | None
    perturbation_id: str | None

    @property
    def is_perturbed(self) -> bool:
        return self.perturbation_id is not None

    def as_tuple(self) -> tuple[Any, ...]:
        return (self.instance_id, self.train_trial_index, self.perturbation_id)


@dataclass(frozen=True, slots=True)
class InstanceStatKey:
    """Identifies a single metric row for a specific instance variant."""
    variant: InstanceVariantKey
    metric: str | None
    split: str | None
    sub_split: str | None
    stat_perturbation_id: str | None

    def as_tuple(self) -> tuple[Any, ...]:
        # legacy 7-tuple format
        return (
            self.variant.instance_id,
            self.variant.train_trial_index,
            self.variant.perturbation_id,
            self.metric,
            self.split,
            self.sub_split,
            self.stat_perturbation_id,
        )


@dataclass(frozen=True, slots=True)
class InstanceStatRow:
    """A joined per-instance stat row with attached request_state."""
    key: InstanceStatKey
    stat: dict[str, Any]
    request_state: dict[str, Any] | None

    @property
    def mean(self) -> float | None:
        return _safe_float(self.stat.get("mean", None))

    @property
    def count(self) -> int:
        return int(self.stat.get("count", 0) or 0)


class JoinedInstanceStatTable(ub.NiceRepr):
    """
    Join `scenario_state['request_states']` with `per_instance_stats`.

    This class is *pure*: it accepts JSON structures and builds indices.

    Example:
        >>> from magnet.backends.helm.helm_run_analysis import (
        ...     JoinedInstanceStatTable, InstanceVariantKey
        ... )
        >>> request_states = [
        ...     {
        ...         'instance': {'id': 'id1', 'split': 'test', 'input': {'text': 'hello'}},
        ...         'train_trial_index': 0,
        ...         'request': {'prompt': 'P0'},
        ...         'result': {'completions': [{'text': 'A'}]},
        ...     },
        ...     {
        ...         'instance': {
        ...             'id': 'id1', 'split': 'test', 'input': {'text': 'hello'},
        ...             'perturbation': {'name': 'dialect', 'prob': 1.0},
        ...         },
        ...         'train_trial_index': 0,
        ...         'request': {'prompt': 'P0 dialect'},
        ...         'result': {'completions': [{'text': 'B'}]},
        ...     },
        ... ]
        >>> perinstance_stats = [
        ...     # base stats split across two bundles
        ...     {'instance_id': 'id1', 'train_trial_index': 0,
        ...      'stats': [{'name': {'name': 'num_bytes', 'split': 'test'}, 'count': 1, 'mean': 10.0}]},
        ...     {'instance_id': 'id1', 'train_trial_index': 0,
        ...      'stats': [{'name': {'name': 'num_prompt_tokens', 'split': 'test'}, 'count': 1, 'mean': 3.0}]},
        ...     # perturbed stats: perturbation appears in the stat-name dict
        ...     {'instance_id': 'id1', 'train_trial_index': 0,
        ...      'stats': [{'name': {'name': 'num_bytes', 'split': 'test',
        ...                         'perturbation': {'name': 'dialect', 'prob': 1.0}},
        ...                'count': 1, 'mean': 12.0}]},
        ... ]
        >>> tbl = JoinedInstanceStatTable(request_states, perinstance_stats, short_hash=8)
        >>> _ = tbl.assert_assumptions()
        >>> variants = tbl.variant_keys_for_instance('id1')
        >>> assert len(variants) == 2
        >>> base = [v for v in variants if not v.is_perturbed][0]
        >>> pert = [v for v in variants if v.is_perturbed][0]
        >>> assert sorted([s['name']['name'] for s in tbl.stats_for_variant(base)]) == ['num_bytes', 'num_prompt_tokens']
        >>> # Key round-trip via legacy tuple
        >>> r0 = tbl.rows_for_variant(pert)[0]
        >>> k = r0.key.as_tuple()
        >>> assert tbl.get_row(k) is r0
    """

    def __init__(
        self,
        request_states: list[dict[str, Any]],
        perinstance_stats: list[dict[str, Any]],
        *,
        short_hash: int = 12,
    ):
        self.request_states = request_states
        self.perinstance_stats = perinstance_stats
        self.short_hash = short_hash

        self.request_state_by_variant: dict[InstanceVariantKey, dict[str, Any]] = {}
        self.stats_by_variant: dict[InstanceVariantKey, list[dict[str, Any]]] = {}
        self.rows_by_variant: dict[InstanceVariantKey, list[InstanceStatRow]] = {}
        self.row_by_key: dict[InstanceStatKey, InstanceStatRow] = {}

        self.diagnostics: dict[str, Any] = {
            "request_state_duplicates": [],
            "unmatched_variants": [],
        }

        self._build()

    def __nice__(self):
        return f"variants={len(self.request_state_by_variant)} rows={len(self.row_by_key)}"

    def __len__(self):
        return len(self.row_by_key)

    def __iter__(self):
        """Iterate over InstanceStatRow objects (joined rows)."""
        return iter(self.row_by_key.values())

    # --- core build ---

    def _build(self):
        # 1) index request_states by variant
        dupes = []
        for rs in self.request_states:
            inst = rs.get("instance") or {}
            iid = inst.get("id", None)
            tti = _coerce_int(rs.get("train_trial_index", None))
            pid = _nice_perturbation_id(inst.get("perturbation", None), short_hash=self.short_hash)
            vk = InstanceVariantKey(iid, tti, pid)
            if vk in self.request_state_by_variant:
                dupes.append((vk, self.request_state_by_variant[vk], rs))
                continue
            self.request_state_by_variant[vk] = rs
        self.diagnostics["request_state_duplicates"] = dupes

        # 2) merge perinstance bundles into per-variant groups
        tmp: dict[InstanceVariantKey, list[dict[str, Any]]] = {}

        for row in self.perinstance_stats:
            iid = row.get("instance_id", None)
            tti = _coerce_int(row.get("train_trial_index", None))
            stats = row.get("stats", []) or []
            # group stats inside this row by their stat-name perturbation
            per_pid: dict[str | None, list[dict[str, Any]]] = {}
            for stat in stats:
                name_obj = stat.get("name", None) or {}
                stat_pid = None
                if isinstance(name_obj, dict):
                    stat_pid = _nice_perturbation_id(name_obj.get("perturbation", None), short_hash=self.short_hash)
                per_pid.setdefault(stat_pid, []).append(stat)

            for stat_pid, subset in per_pid.items():
                vk = InstanceVariantKey(iid, tti, stat_pid)
                tmp.setdefault(vk, []).extend(subset)

        self.stats_by_variant = tmp

        # 3) join into InstanceStatRow objects
        unmatched = []
        for vk, stats in self.stats_by_variant.items():
            rs = self.request_state_by_variant.get(vk, None)

            # fallback: if stat pid None but only one request variant exists for this base key
            if rs is None and vk.perturbation_id is None:
                candidates = [k for k in self.request_state_by_variant.keys()
                              if (k.instance_id == vk.instance_id and k.train_trial_index == vk.train_trial_index)]
                if len(candidates) == 1:
                    rs = self.request_state_by_variant[candidates[0]]

            if rs is None:
                unmatched.append(vk)

            rows = []
            for stat in stats:
                name_obj = stat.get("name", None) or {}
                metric = name_obj.get("name", None) if isinstance(name_obj, dict) else None
                split = name_obj.get("split", None) if isinstance(name_obj, dict) else None
                sub_split = name_obj.get("sub_split", None) if isinstance(name_obj, dict) else None
                stat_pid = None
                if isinstance(name_obj, dict):
                    stat_pid = _nice_perturbation_id(name_obj.get("perturbation", None), short_hash=self.short_hash)

                sk = InstanceStatKey(vk, metric, split, sub_split, stat_pid)
                row_obj = InstanceStatRow(sk, stat, rs)
                rows.append(row_obj)
                self.row_by_key[sk] = row_obj

            self.rows_by_variant[vk] = rows

        self.diagnostics["unmatched_variants"] = unmatched

    # --- assertions (optional) ---

    def assert_assumptions(self) -> "JoinedInstanceStatTable":
        dupes = self.diagnostics.get("request_state_duplicates", [])
        assert not dupes, f"Duplicate request_state variant keys. Example={dupes[:1]!r}"

        unmatched = self.diagnostics.get("unmatched_variants", [])
        assert not unmatched, f"Some perinstance variants could not be matched to request_states. Example={unmatched[:5]!r}"
        return self

    # --- query helpers ---

    def variant_keys(self) -> list[InstanceVariantKey]:
        return sorted(self.request_state_by_variant.keys(), key=lambda k: (str(k.instance_id), k.train_trial_index or -1, str(k.perturbation_id)))

    def variant_keys_for_instance(
        self,
        instance_id: str,
        *,
        train_trial_index: int | None = None,
        include_perturbed: bool = True,
        include_unperturbed: bool = True,
    ) -> list[InstanceVariantKey]:
        out = []
        for k in self.request_state_by_variant.keys():
            if k.instance_id != instance_id:
                continue
            if train_trial_index is not None and k.train_trial_index != train_trial_index:
                continue
            if k.is_perturbed and not include_perturbed:
                continue
            if (not k.is_perturbed) and not include_unperturbed:
                continue
            out.append(k)
        return sorted(out, key=lambda k: (k.train_trial_index or -1, str(k.perturbation_id)))

    def request_state(self, variant: InstanceVariantKey) -> dict[str, Any] | None:
        return self.request_state_by_variant.get(variant, None)

    def rows_for_variant(self, variant: InstanceVariantKey) -> list[InstanceStatRow]:
        return self.rows_by_variant.get(variant, [])

    def stats_for_variant(self, variant: InstanceVariantKey) -> list[dict[str, Any]]:
        return [r.stat for r in self.rows_for_variant(variant)]

    def rows_for_instance(self, instance_id: str, *, include_perturbed: bool = True) -> list[InstanceStatRow]:
        rows = []
        for vk in self.variant_keys_for_instance(instance_id, include_perturbed=include_perturbed, include_unperturbed=True):
            rows.extend(self.rows_for_variant(vk))
        return rows

    def stats_for_instance(self, instance_id: str, *, include_perturbed: bool = True) -> list[dict[str, Any]]:
        return [r.stat for r in self.rows_for_instance(instance_id, include_perturbed=include_perturbed)]

    def get_row(self, key: InstanceStatKey | tuple[Any, ...]) -> InstanceStatRow | None:
        if isinstance(key, InstanceStatKey):
            return self.row_by_key.get(key, None)
        if isinstance(key, tuple) and len(key) == 7:
            vk = InstanceVariantKey(key[0], _coerce_int(key[1]), key[2])
            sk = InstanceStatKey(vk, key[3], key[4], key[5], key[6])
            return self.row_by_key.get(sk, None)
        raise TypeError(f"Unrecognized key type: {type(key)}")


def summary_dict(
    self,
    *,
    level: int | str = 1,
    short_hash: int = 12,
    include_instance_stats: bool | None = None,
    include_headline_instances: bool | None = None,
    drop_zero_count: bool = False,
    assert_join_assumptions: bool = False,
) -> dict[str, Any]:
    """
    Programmatic single-run summary.

    This is intended to be:
      * easy to print for a human,
      * stable-ish enough to compare between runs,
      * informative enough to decide if you need HelmRunDiff.

    Parameters
    ----------
    level:
        Numeric detail level. Higher => more details.

        Suggested semantics (current implementation):
          - level <= 0: ultra-lite (one-line-friendly fields)
          - level >= 1: include run-level inventories (families by class/support)
          - level >= 5: include instance-level inventories (joined per-instance stats)
          - level >= 10: include a few headline instances (truncated prompt/completion)

        Back-compat: 'lite' maps to 0.

    short_hash:
        Hash prefix length used in small signatures.

    include_instance_stats:
        If None: enabled when level >= 5.
        If True: always attempt to build instance inventories via joined_instance_stat_table().

    include_headline_instances:
        If None: enabled when level >= 10.
        If True: include exemplar instance variants with truncated text.

    drop_zero_count:
        If True: ignore count==0 rows in certain inventories. (Default False because
        “unsupported” (count==0) is explicitly useful in this summary.)

    assert_join_assumptions:
        If True, call the joiner with assertions enabled (loud failures). Keep
        False by default for runtime.

    Returns
    -------
    dict
        Structured summary with run-level and (optionally) instance-level inventories.
    """
    # Backwards compatibility
    if isinstance(level, str):
        if level == 'lite':
            level = 0
        else:
            raise KeyError(level)

    if include_instance_stats is None:
        include_instance_stats = (level >= 5)
    if include_headline_instances is None:
        include_headline_instances = (level >= 10)

    cache_key = (
        'summary_dict_v4',
        level, short_hash, include_instance_stats, include_headline_instances,
        drop_zero_count, assert_join_assumptions,
    )
    if cache_key in self._cache:
        return self._cache[cache_key]

    def _short_sig(obj) -> str | None:
        if obj is None:
            return None
        try:
            return helm_hashers.stable_hash36(obj)[:short_hash]
        except Exception:
            return None

    def _family_support_inventory(rows: list[dict[str, Any]], *, name_getter):
        """
        Build family inventories grouped by:
            metric_class -> supported/unsupported -> family -> count
        """
        fam_counts: dict[str, dict[str, Counter]] = {}
        metric_counts: dict[str, dict[str, Counter]] = {}
        split_counts: dict[str, dict[str, Counter]] = {}

        for row in rows:
            c = int(row.get('count', 0) or 0)
            if drop_zero_count and c == 0:
                continue

            name_obj = name_getter(row)
            metric = name_obj.get('name', None) if isinstance(name_obj, dict) else None
            split = name_obj.get('split', None) if isinstance(name_obj, dict) else None

            mclass, _ = classify_metric(metric)
            fam = metric_family(metric)

            support = 'supported' if c > 0 else 'unsupported'
            fam_counts.setdefault(mclass, {}).setdefault(support, Counter())[fam] += 1
            metric_counts.setdefault(mclass, {}).setdefault(support, Counter())[metric] += 1
            split_counts.setdefault(mclass, {}).setdefault(support, Counter())[split] += 1

        # Convert counters to stable sortable lists (count desc, then name)
        out = {}
        for mclass in sorted(fam_counts.keys()):
            out[mclass] = {}
            for support in ('supported', 'unsupported'):
                counter = fam_counts[mclass].get(support, Counter())
                out[mclass][support] = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
            # quick per-class totals
            out[mclass]['n_rows_supported'] = int(sum(fam_counts[mclass].get('supported', {}).values()))
            out[mclass]['n_rows_unsupported'] = int(sum(fam_counts[mclass].get('unsupported', {}).values()))
            # unique metric counts (per support)
            out[mclass]['n_unique_metrics_supported'] = int(sum(1 for k, v in metric_counts[mclass].get('supported', {}).items() if k is not None))
            out[mclass]['n_unique_metrics_unsupported'] = int(sum(1 for k, v in metric_counts[mclass].get('unsupported', {}).items() if k is not None))

        # Signature for “is the inventory basically the same?”
        try:
            sig_items = []
            for mclass, d in out.items():
                for support in ('supported', 'unsupported'):
                    for fam, cnt in d.get(support, []):
                        sig_items.append((mclass, support, fam, cnt))
            sig_items = sorted(sig_items)
            inv_sig = helm_hashers.stable_hash36(sig_items)[:short_hash]
        except Exception:
            inv_sig = None

        return out, inv_sig

    # --- core run identity ---
    spec = self.run_spec() or {}
    scen = self.scenario() or {}
    scen_state = self.scenario_state() or {}
    request_states = (scen_state.get('request_states', []) or [])
    stats = self.stats() or []

    label = self.name or str(self.run.path.name)

    # --- request_state base/variant inventory ---
    base_keys = []
    n_rs_pert = 0
    for rs in request_states:
        inst = rs.get('instance') or {}
        base_keys.append((inst.get('id', None), rs.get('train_trial_index', None)))
        if inst.get('perturbation', None):
            n_rs_pert += 1
    base_counter = Counter(base_keys)
    n_bases = len(base_counter)
    n_variants = len(request_states)
    max_variants_per_base = max(base_counter.values()) if base_counter else 0

    # --- run-level stat inventories ---
    run_stats_total = len(stats)
    run_stats_nonzero = sum(1 for r in stats if int(r.get('count', 0) or 0) != 0)
    run_stats_with_mean = sum(1 for r in stats if (int(r.get('count', 0) or 0) != 0) and ('mean' in r))
    run_stats_pert = sum(1 for r in stats if isinstance(r.get('name', None), dict) and bool(r['name'].get('perturbation', None)))

    # signatures: run spec / scenario / stat-name-set
    stat_name_ids = []
    for r in stats:
        try:
            stat_name_ids.append(helm_hashers.stat_name_id(r.get('name', None), count=r.get('count', None)))
        except Exception:
            stat_name_ids.append(ub.urepr(r.get('name', None), compact=1, nl=0, nobr=1))
    stat_name_ids = sorted(set(stat_name_ids))

    run_stats_fams, run_stats_fams_sig = _family_support_inventory(
        stats,
        name_getter=lambda r: (r.get('name', None) or {}),
    )

    # --- instance-level inventories via join ---
    inst_info = None
    if include_instance_stats:
        try:
            joined = self.joined_instance_stat_table(
                assert_assumptions=assert_join_assumptions,
                short_hash=short_hash,
            )
        except Exception as ex:
            inst_info = {'error': repr(ex)}
        else:
            # joined rows are "one per stat", so variants/base stats are reconstructed from request_state in rows
            # But we can count variants more directly from scenario_state:
            # (still useful to report join-derived counts)
            inst_rows_total = len(joined)
            inst_rows_nonzero = sum(1 for r in joined if int(r.count or 0) != 0)
            inst_rows_with_mean = sum(1 for r in joined if (int(r.count or 0) != 0) and (r.mean is not None))

            # per-instance family inventory: use the underlying stat dicts
            inst_stats = [r.stat for r in joined]
            inst_fams, inst_fams_sig = _family_support_inventory(
                inst_stats,
                name_getter=lambda s: (s.get('name', None) or {}),
            )

            inst_info = {
                'rows_total': inst_rows_total,
                'rows_nonzero': inst_rows_nonzero,
                'rows_with_mean': inst_rows_with_mean,
                'families_by_class': inst_fams,
                'inventory_sig': inst_fams_sig,
            }

    # --- headline instances (IDs + truncated text) ---
    headline = None
    if include_headline_instances:
        headline = []
        # deterministic-ish: group by base, pick 3 unperturbed then 3 perturbed
        unp = []
        per = []
        for rs in request_states:
            inst = rs.get('instance') or {}
            if inst.get('perturbation', None):
                per.append(rs)
            else:
                unp.append(rs)

        def _sortkey(rs):
            inst = rs.get('instance') or {}
            pert = inst.get('perturbation', None) or {}
            pname = pert.get('name', None) if isinstance(pert, dict) else None
            return (str(inst.get('id', '')), int(rs.get('train_trial_index', 0) or 0), str(pname))

        unp = sorted(unp, key=_sortkey)
        per = sorted(per, key=_sortkey)

        # keep small
        pick = unp[:3] + per[:3]
        for rs in pick:
            inst = rs.get('instance') or {}
            pert = inst.get('perturbation', None)
            pert_name = pert.get('name', None) if isinstance(pert, dict) else None
            pert_id = None
            try:
                pert_id = helm_hashers.perturbation_id(pert, short_hash=short_hash)
            except Exception:
                pert_id = None

            req = rs.get('request') or {}
            res = rs.get('result') or {}
            comps = (res.get('completions') or [])
            comp_text = comps[0].get('text', None) if comps else None

            headline.append({
                'instance_id': inst.get('id', None),
                'train_trial_index': rs.get('train_trial_index', None),
                'split': inst.get('split', None),
                'perturbation': pert_name,
                'perturbation_id': pert_id,
                'prompt': req.get('prompt', None),
                'completion': comp_text,
                'input': (inst.get('input') or {}).get('text', None) if isinstance(inst.get('input', None), dict) else inst.get('input', None),
            })

    out: dict[str, Any] = {
        'label': label,
        'path': str(self.run.path),
        'run_spec_name': spec.get('name', None),
        'scenario_name': (spec.get('scenario_spec', {}) or {}).get('class_name', None),

        'signatures': {
            'run_spec_sig': _short_sig(spec),
            'scenario_sig': _short_sig(scen) if scen else None,
            'stats_name_sig': _short_sig(stat_name_ids),
            'run_stats_families_sig': run_stats_fams_sig,
            'instance_stats_families_sig': (inst_info or {}).get('inventory_sig', None) if isinstance(inst_info, dict) else None,
        },

        'requests': {
            'request_states': n_variants,
            'bases': n_bases,
            'perturbed_request_states': n_rs_pert,
            'max_variants_per_base': max_variants_per_base,
        },

        'run_stats': {
            'total': run_stats_total,
            'nonzero': run_stats_nonzero,
            'with_mean': run_stats_with_mean,
            'perturbed': run_stats_pert,
            'families_by_class': run_stats_fams,
        },

        'instance_stats': inst_info,
        'headline_instances': headline,
    }

    self._cache[cache_key] = out
    return out


def summary(
    self,
    *,
    level: int = 1,
    writer=None,
    short_hash: int = 12,
    include_instance_stats: bool | None = None,
    include_headline_instances: bool | None = None,
    family_topn: int = 12,
    drop_zero_count: bool = False,
    assert_join_assumptions: bool = False,
    prompt_chars: int = 80,
    completion_chars: int = 80,
    input_chars: int = 80,
) -> None:
    """
    Write a human-readable single-run summary.

    Writer style
    ------------
    `writer` should be a callable that accepts a single string.
      * Default: `rich.print` (interactive friendly)
      * To accumulate: pass `lines.append`

    Level semantics
    --------------
    This function uses numeric levels rather than 'line'/'page'.

      level <= 0:
          single line intended for side-by-side display

      level >= 1:
          includes run-level inventories:
            - counts
            - metric families grouped by: (metric_class -> supported/unsupported -> family)

      level >= 5:
          includes instance-level inventories via joined_instance_stat_table()
          (often the most informative “is this the same run?” signal)

      level >= 10:
          prints headline instances (ids + truncated input/prompt/completion) so you can
          immediately look them up by id.

    Glossary / labels
    -----------------
    requests.request_states:
        Number of entries in scenario_state['request_states'] (includes perturbation variants).

    requests.bases:
        Number of unique (instance_id, train_trial_index) pairs found in request_states.

    requests.max_variants_per_base:
        Max number of request_states sharing the same base key. >1 is typical with perturbations.

    run_stats.families_by_class:
        For run-level stats.json, counts grouped by:
          metric_class ('core', 'bookkeeping', 'untracked', ...)
            -> support ('supported' if count>0 else 'unsupported' if count==0)
              -> family (heuristic prefix family via metric_family())

        This is often better than raw metric lists for “shape of the run”.

    instance_stats.*:
        Same structure, but computed from per-instance stats (joined to request_states).
    """
    if writer is None:
        try:
            from rich import print as rich_print
            writer = rich_print
        except Exception:
            writer = print

    # Heuristic: only show core at low levels; add bookkeeping at mid; untracked at high.
    if level < 5:
        classes_to_show = ('core',)
    elif level < 10:
        classes_to_show = ('core', 'bookkeeping')
    else:
        classes_to_show = ('core', 'bookkeeping', 'untracked')

    info = self.summary_dict(
        level=level,
        short_hash=short_hash,
        include_instance_stats=include_instance_stats,
        include_headline_instances=include_headline_instances,
        drop_zero_count=drop_zero_count,
        assert_join_assumptions=assert_join_assumptions,
    )

    label = info.get('label') or ub.Path(info.get('path')).name
    sig = info.get('signatures') or {}
    req = info.get('requests') or {}
    rs = info.get('run_stats') or {}
    inst = info.get('instance_stats', None)

    # --- truncation helpers ---
    try:
        from kwutil.slugify_ext import smart_truncate
    except Exception:
        smart_truncate = None

    is_rich = (getattr(writer, '__module__', '') or '').startswith('rich')
    if is_rich:
        try:
            from rich.markup import escape as _escape
        except Exception:
            _escape = None
    else:
        _escape = None

    def _clip(text, n):
        if text is None:
            return None
        text = str(text).replace('\r\n', '\n')
        # Using repr to make output more structured.
        text = repr(text)
        if smart_truncate is not None:
            out = smart_truncate(text, max_length=n, hash_len=8, separator=' ', trunc_loc=0.5)
        else:
            out = text[:n] + '…' if len(text) > n else text
        if _escape is not None:
            out = _escape(out)
        return out

    # --- level 0: one-liner ---
    if level <= 0:
        writer(
            f"{label} | spec={info.get('run_spec_name')} | "
            f"sig(spec)={sig.get('run_spec_sig')} | "
            f"sig(stats)={sig.get('stats_name_sig')} | "
            f"req={req.get('request_states')} bases={req.get('bases')} maxvar={req.get('max_variants_per_base')}"
        )
        return None

    # --- header ---
    writer(f"Run: {label}")
    writer(f"  path: {info.get('path')}")
    writer(f"  run_spec_name: {info.get('run_spec_name')}")
    writer(f"  scenario: {info.get('scenario_name')}")
    writer("  signatures:")
    writer(f"    run_spec_sig:   {sig.get('run_spec_sig')}")
    writer(f"    scenario_sig:   {sig.get('scenario_sig')}")
    writer(f"    stats_name_sig: {sig.get('stats_name_sig')}")

    # --- requests ---
    writer("  requests:")
    writer(
        f"    request_states={req.get('request_states')} "
        f"bases={req.get('bases')} perturbed={req.get('perturbed_request_states')} "
        f"max_variants_per_base={req.get('max_variants_per_base')}"
    )

    # --- run-level stats ---
    writer("  run-level stats (stats.json):")
    writer(
        f"    total={rs.get('total')} nonzero={rs.get('nonzero')} "
        f"with_mean={rs.get('with_mean')} perturbed={rs.get('perturbed')}"
    )

    fams_by_class = (rs.get('families_by_class') or {})
    writer("    families_by_class (class -> supported/unsupported -> family):")
    for cls in classes_to_show:
        if cls not in fams_by_class:
            continue
        cinfo = fams_by_class.get(cls) or {}
        writer(f"      {cls}: supported_rows={cinfo.get('n_rows_supported')} unsupported_rows={cinfo.get('n_rows_unsupported')}")
        supp = cinfo.get('supported', [])[:family_topn]
        unsupp = cinfo.get('unsupported', [])[:family_topn]
        if supp:
            writer("        supported:")
            for fam, cnt in supp:
                writer(f"          {fam}: {cnt}")
        if unsupp and level >= 10:
            writer("        unsupported:")
            for fam, cnt in unsupp:
                writer(f"          {fam}: {cnt}")

    # --- instance-level stats ---
    if inst is None:
        writer("  instance-level stats: (disabled)")
    elif isinstance(inst, dict) and inst.get('error'):
        writer(f"  instance-level stats: ERROR {inst.get('error')}")
    else:
        writer("  instance-level stats (joined per_instance_stats + request_states):")
        writer(
            f"    rows_total={inst.get('rows_total')} "
            f"rows_nonzero={inst.get('rows_nonzero')} rows_with_mean={inst.get('rows_with_mean')} "
            f"families_sig={sig.get('instance_stats_families_sig')}"
        )
        inst_fams = (inst.get('families_by_class') or {})
        writer("    families_by_class (class -> supported/unsupported -> family):")
        for cls in classes_to_show:
            if cls not in inst_fams:
                continue
            cinfo = inst_fams.get(cls) or {}
            writer(f"      {cls}: supported_rows={cinfo.get('n_rows_supported')} unsupported_rows={cinfo.get('n_rows_unsupported')}")
            supp = cinfo.get('supported', [])[:family_topn]
            unsupp = cinfo.get('unsupported', [])[:family_topn]
            if supp:
                writer("        supported:")
                for fam, cnt in supp:
                    writer(f"          {fam}: {cnt}")
            if unsupp and level >= 10:
                writer("        unsupported:")
                for fam, cnt in unsupp:
                    writer(f"          {fam}: {cnt}")

    # --- headline instances (high level) ---
    headline = info.get('headline_instances', None)
    if headline and level >= 10:
        writer("  headline instances (lookup by instance_id):")
        for h in headline:
            iid = h.get('instance_id')
            tti = h.get('train_trial_index')
            split = h.get('split')
            pert = h.get('perturbation')
            pid = h.get('perturbation_id')

            writer(f"    - id={iid} tti={tti} split={split} pert={pert} pid={pid}")
            inp = _clip(h.get('input'), input_chars)
            prm = _clip(h.get('prompt'), prompt_chars)
            cmp = _clip(h.get('completion'), completion_chars)
            if inp is not None:
                writer(f"      input: {inp}")
            if prm is not None:
                writer(f"      prompt: {prm}")
            if cmp is not None:
                writer(f"      completion: {cmp}")

    return None


def summary_text(self, *, level: int = 1, **kwargs) -> str:
    """
    Backwards-compatible string-returning helper.
    """
    lines: list[str] = []
    self.summary(level=level, writer=lines.append, **kwargs)
    return "\n".join(lines)
