"""magnet.backends.helm.rundiff.helm_run_analysis

Single-run analysis utilities wrapped in an object.

Why this exists
--------------
``HelmRun`` (in :mod:`magnet.helm_outputs`) is intentionally a *reader*.
This module defines :class:`HelmRunAnalysis`, which *wraps* a ``HelmRun`` and
adds cached analyses / indices that make higher-level tasks (e.g. pairwise
Diffs) much easier to write.

Design goals (match notebook-style workflows)
--------------------------------------------
* Keep computations *lazy* and cache results.
* Provide stable identifiers for stats / instances (hash-based).
* Keep the API ergonomic for interactive drilldowns.

Notes
-----
* We primarily operate on the json view (``run.json``) for speed and
  robustness across HELM versions.
* This module deliberately avoids any global config system.

Incorporating HELM summarize semantics
-------------------------------------
We borrow a few useful ideas from HELM's own summarization code
(``helm.benchmark.presentation.summarize``), but we keep the implementation
here self-contained and lightweight.

In particular:

* "Matcher"-based stat selection (similar to ``get_unique_stat_by_matcher``)
* When a matcher leaves ``sub_split`` unspecified, we aggregate across
  sub-splits.
* "Cell-like" missing/undefined semantics to distinguish between:

  1) no matching stats
  2) matching stats but ``count == 0`` (mean undefined)
  3) matching stats with ``count > 0``

We do **not** depend on HELM's table / frontend presentation types.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional, Union
import ubelt as ub
from magnet.backends.helm.rundiff import helm_hashers


# --- Metric registries ------------------------------------------------------

# IMPORTANT: keep this in sync with compare.py (duplicated on purpose to avoid
# import cycles while this refactor is in-flight).
class METRIC_PREFIXES:
    """Registry of metric prefixes we care about.

    Kept encapsulated in a class (not module-level tuples) to make it easy to
    swap / extend in notebooks without import-time side effects.
    """

    CORE_PREFIXES: tuple[str, ...] = (
        'exact_match',
        'quasi_exact_match',
        'prefix_exact_match',
        'quasi_prefix_exact_match',
        'classification_micro_f1',
        'classification_macro_f1',
        'f1_score',
        'rouge_l',
        'bleu_',
        'ifeval_strict_accuracy',
        'wildbench_score',
        'wildbench_score_rescaled',
        'omni_math_accuracy',
        'chain_of_thought_correctness',
        'math_equiv',
        'math_equiv_chain_of_thought',
        'safety_score',
        'safety_gpt_score',
        'safety_llama_score',
        'air_score',
        'air_category_',
    )

    BOOKKEEPING_PREFIXES: tuple[str, ...] = (
        # token/size/runtime / resource accounting
        'num_',
        'training_',
        'inference_',
        'batch_size',
        'max_prob',
        'logprob',
        'num_perplexity_tokens',
        'num_bytes',
        'perplexity',
        'bits_per_byte',
        'logprob_per_byte',
        # decoding / stopping bookkeeping
        'finish_reason_',
        'prompt_truncated',
        # calibration / fitting plumbing
        'ece_',
        'platt_',
        'selective_',
        # meta / dataset sizing
        'num_instances',
        'num_train_',
        'num_references',
    )


# --- Main analysis object ---------------------------------------------------


class HelmRunAnalysis(ub.NiceRepr):
    """Wrap a :class:`~magnet.helm_outputs.HelmRun` with cached analyses.

    Parameters
    ----------
    run:
        The underlying run reader.
    name:
        Human-friendly label used in summaries (optional).

    Example:
        >>> from magnet.backends.helm.helm_run_analysis import *  # NOQA
        >>> from magnet.backends.helm.helm_outputs import HelmRun
        >>> run = HelmRun.demo()
        >>> self = HelmRunAnalysis(run)
        >>> self.summary_lite()

    Ignore:
        Show different types of datas

        stats = self.stats()[0]
        print(f'stats = {ub.urepr(stats, nl=1)}')

        stats = self.stats()[-3]
        print(f'stats = {ub.urepr(stats, nl=1)}')

        request = self.scenario_state()['request_states'][0]
        print(f'request = {ub.urepr(request, nl=1)}')

        instance = request['instance']
        print(f'instance = {ub.urepr(instance, nl=1)}')

        perinstance = self.per_instance_stats()[0]
        print(f'perinstance={perinstance}')
    """

    def __init__(self, run):
        self.run = run
        self._a_cache: dict[str, Any] = {}
        self._cache: dict[Any, Any] = {}

    def __nice__(self):
        label = str(self.run.path.name)
        return label

    # --- Raw json getters (cached) ----------------------------------------

    def run_spec(self) -> dict[str, Any]:
        return self._a_cache_get('run_spec', lambda: self.run.json.run_spec())

    def scenario(self) -> dict[str, Any]:
        return self._a_cache_get('scenario', lambda: self.run.json.scenario())

    def scenario_state(self) -> dict[str, Any]:
        return self._a_cache_get('scenario_state', lambda: self.run.json.scenario_state())

    def stats(self) -> list[dict[str, Any]]:
        return self._a_cache_get('stats', lambda: self.run.json.stats())

    def per_instance_stats(self) -> list[dict[str, Any]]:
        return self._a_cache_get('per_instance_stats', lambda: self.run.json.per_instance_stats())

    def _a_cache_get(self, key: str, factory):
        if key not in self._a_cache:
            self._a_cache[key] = factory()
        return self._a_cache[key]

    # --- Stat inventory / indexing ----------------------------------------

    def iter_stat_meta(
        self,
        *,
        drop_zero_count: bool = True,
        ignore_name_file_path: bool = True,
    ) -> Iterator['StatMeta']:
        """Yield normalized metadata for each stat row."""
        for s in self.stats():
            count = int(s.get('count', 0) or 0)
            if drop_zero_count and count == 0:
                continue
            name_obj = s.get('name', None)
            metric = name_obj.get('name', None) if isinstance(name_obj, dict) else None
            split = name_obj.get('split', None) if isinstance(name_obj, dict) else None
            pert = None
            if isinstance(name_obj, dict):
                p = name_obj.get('perturbation', None)
                if isinstance(p, dict):
                    pert = p.get('name', None)
            is_pert = pert is not None
            mclass, mpref = classify_metric(metric)
            fam = metric_family(metric)
            key = stable_name_key(name_obj, ignore_name_file_path=ignore_name_file_path)
            mean = _safe_float(s.get('mean', None))
            yield StatMeta(
                key=key,
                metric=metric,
                split=split,
                is_perturbed=is_pert,
                pert_name=pert,
                family=fam,
                metric_class=mclass,
                matched_prefix=mpref,
                count=count,
                mean=mean,
                name_obj=name_obj,
                raw=s,
            )

    def stat_index(
        self,
        *,
        drop_zero_count: bool = True,
        ignore_name_file_path: bool = True,
        require_mean: bool = False,
    ) -> dict[str, 'StatMeta']:
        """Map stable stat-key -> :class:`StatMeta`."""
        cache_key = ('stat_index', drop_zero_count, ignore_name_file_path, require_mean)
        if cache_key in self._cache:
            return self._cache[cache_key]
        idx: dict[str, StatMeta] = {}
        for m in self.iter_stat_meta(
            drop_zero_count=drop_zero_count,
            ignore_name_file_path=ignore_name_file_path,
        ):
            if require_mean and m.mean is None:
                continue
            idx[m.key] = m
        self._cache[cache_key] = idx
        return idx

    def stats_inventory(self, *, drop_zero_count: bool = False) -> dict[str, Counter]:
        """Lightweight histograms over stats for exploration."""
        cache_key = ('stats_inventory', drop_zero_count)
        if cache_key in self._cache:
            return self._cache[cache_key]
        hist = {
            'counts': Counter(),
            'perturbed': Counter(),
            'splits': Counter(),
            'family': Counter(),
            'metric_class': Counter(),
        }
        for s in self.stats():
            c = int(s.get('count', 0) or 0)
            hist['counts'][c] += 1
            if drop_zero_count and c == 0:
                continue
            name = s.get('name', None)
            metric = name.get('name', None) if isinstance(name, dict) else None
            split = name.get('split', None) if isinstance(name, dict) else None
            is_pert = bool(isinstance(name, dict) and name.get('perturbation', None))
            hist['perturbed'][is_pert] += 1
            hist['splits'][split] += 1
            hist['family'][metric_family(metric)] += 1
            hist['metric_class'][classify_metric(metric)[0]] += 1
        self._cache[cache_key] = hist
        return hist

    # --- Scenario/request-state analysis ----------------------------------

    def request_state_key(self, rs: Mapping[str, Any]) -> 'RequestStateKey':
        """Extract a best-effort key from a request_state row."""
        instance = rs.get('instance', {}) if isinstance(rs, dict) else {}
        iid = instance.get('id', None)
        split = instance.get('split', None)
        sub_split = instance.get('sub_split', None)
        tti = rs.get('train_trial_index', None)
        if tti is None:
            tti = rs.get('train_trial_index', rs.get('trial_index', None))
        if isinstance(tti, str) and tti.isdigit():
            tti = int(tti)
        if isinstance(tti, float):
            tti = int(tti)
        pert_name = None
        pert = rs.get('perturbation', None)
        if isinstance(pert, dict):
            pert_name = pert.get('name', None)
        return RequestStateKey(
            instance_id=iid,
            split=split,
            sub_split=sub_split,
            train_trial_index=tti if isinstance(tti, int) else None,
            pert_name=pert_name,
        )

    def request_state_fingerprint(self, rs: Mapping[str, Any]) -> str:
        """A stable fingerprint used when ids don't line up."""
        if not isinstance(rs, dict):
            return stable_instance_fingerprint(rs)
        instance = rs.get('instance', None)
        request = rs.get('request', None)
        references = None
        if isinstance(instance, dict):
            references = instance.get('references', None)
        # Keep this minimal and stable.
        payload = {
            'instance': ub.udict(instance) if isinstance(instance, dict) else instance,
            'references': references,
        }
        # Prompts can contain env-specific paths; only keep prompt text.
        if isinstance(request, dict):
            prompt = request.get('prompt', None)
            payload['prompt'] = prompt
        return stable_instance_fingerprint(payload)

    def joined_instance_stat_table(self, *, assert_assumptions=True):
        """
        Build a joined “instance-stat → request_state” table.

        Motivation
        ----------
        HELM's `scenario_state()['request_states']` is naturally “one row per evaluated
        instance variant” and may contain multiple rows sharing the same
        `instance['id']` when perturbations are present (e.g., unperturbed + several
        perturbed variants). Meanwhile, HELM's `per_instance_stats()` is *not*
        guaranteed to be one row per request. It often contains multiple entries for
        the same `(instance_id, train_trial_index)`, where each entry contributes a
        partial bundle of stats (e.g., bookkeeping metrics in one bundle and a single
        task metric like `ifeval_strict_accuracy` in another).

        This method:
          1) Indexes request_states by a base key `(instance_id, train_trial_index)`
             and a perturbation id derived from `request_state['instance']['perturbation']`.
          2) Merges per_instance_stats bundles by `(instance_id, train_trial_index)`.
          3) Explodes merged per-instance stats into one row per stat and attaches
             the matching request_state:
               - If the stat name encodes a perturbation (stat['name']['perturbation']),
                 match the request_state for that perturbation.
               - Otherwise match the unperturbed request_state.

        Returns
        -------
        joined_rows : List[Dict]
            Each row contains:
                - 'instance_id' : str
                - 'train_trial_index' : int
                - 'perturbation_id' : Optional[str]
                - 'stat' : Dict (raw stat dict from per_instance_stats)
                - 'request_state' : Dict (raw request_state dict)
            If a stat row cannot be matched to a request_state (unexpected structure),
            it will trigger an assertion when `assert_assumptions=True`, otherwise the
            row is skipped and recorded in an internal `unmatched` list (see code).

        Assertions (when assert_assumptions=True)
        ----------------------------------------
        - per_instance_stats length is at least request_states length
          (since per_instance_stats may contain duplicate bundles).
        - For each (instance_id, train_trial_index, perturbation_id) there is exactly
          one request_state.
        - After merging, the set of (instance_id, train_trial_index) keys matches
          between request_states and per_instance_stats.
        - Every stat row can be matched to exactly one request_state via perturbation.

        Notes
        -----
        - Perturbation identity is matched primarily by perturbation 'name'. If you
          need stricter matching (e.g., perturbation params), swap `req_pert_id` /
          `stat_pert_id` to use your canonical hashers.
        """
        import collections

        request_states = self.scenario_state()['request_states']
        perinstance_stats = self.per_instance_stats()

        def req_base_key(rs):
            inst = rs.get('instance', {}) or {}
            return (inst.get('id', None), rs.get('train_trial_index', None))

        def req_pert_id(rs):
            inst = rs.get('instance', {}) or {}
            pert = inst.get('perturbation', None)
            if isinstance(pert, dict):
                return pert.get('name', None) or 'pert_dict'
            return None  # unperturbed

        def stat_pert_id(stat):
            n = stat.get('name') or {}
            pert = n.get('perturbation', None)
            if isinstance(pert, dict):
                return pert.get('name', None) or 'pert_dict'
            return None  # unperturbed / base metric

        def pi_key(pi):
            return (pi.get('instance_id', None), pi.get('train_trial_index', None))

        # -----------------------
        # Loud structural checks
        # -----------------------
        if assert_assumptions:
            assert isinstance(request_states, list), type(request_states)
            assert isinstance(perinstance_stats, list), type(perinstance_stats)
            assert len(perinstance_stats) >= len(request_states), (
                f"Expected perinstance_stats >= request_states, got "
                f"{len(perinstance_stats)} < {len(request_states)}"
            )

        # -------------------------------------------------------
        # Index request_states by (id, tti) then perturbation_id
        # -------------------------------------------------------
        req_index = {}
        for rs in request_states:
            bkey = req_base_key(rs)
            pid = req_pert_id(rs)
            req_index.setdefault(bkey, {}).setdefault(pid, []).append(rs)

        if assert_assumptions:
            dup_req = []
            for bkey, by_pert in req_index.items():
                for pid, lst in by_pert.items():
                    if len(lst) != 1:
                        dup_req.append((bkey, pid, len(lst)))
            assert not dup_req, (
                "Multiple request_states for same (instance_id, train_trial_index, perturbation_id). "
                f"Example(s): {dup_req[:10]}"
            )

        # -----------------------------------------
        # Merge per_instance_stats bundles by key
        # -----------------------------------------
        merged_pi = {}
        for pi in perinstance_stats:
            k = pi_key(pi)
            merged_pi.setdefault(k, {'instance_id': k[0], 'train_trial_index': k[1], 'stats': []})
            merged_pi[k]['stats'].extend(pi.get('stats', []))

        if assert_assumptions:
            req_base_set = set(req_index.keys())
            pi_set = set(merged_pi.keys())
            missing_in_pi = sorted(req_base_set - pi_set)
            extra_in_pi = sorted(pi_set - req_base_set)
            assert not missing_in_pi, (
                "Some request_states base keys have no per-instance entry after merge. "
                f"Example(s): {missing_in_pi[:20]}"
            )
            assert not extra_in_pi, (
                "Some per-instance entries have no matching request_states base key. "
                f"Example(s): {extra_in_pi[:20]}"
            )

        # -----------------------------------------
        # Build one row per per-instance stat + req
        # -----------------------------------------
        joined_rows = []
        unmatched = []
        for bkey, pi in merged_pi.items():
            by_pert = req_index[bkey]
            for stat in pi.get('stats', []):
                pid = stat_pert_id(stat)
                rs_list = by_pert.get(pid, None)
                if rs_list is None:
                    unmatched.append((bkey, pid, (stat.get('name') or {}).get('name', None)))
                    continue
                # asserted above to be exactly 1
                rs = rs_list[0]
                joined_rows.append({
                    'instance_id': bkey[0],
                    'train_trial_index': bkey[1],
                    'perturbation_id': pid,
                    'stat': stat,
                    'request_state': rs,
                })

        if assert_assumptions:
            assert not unmatched, (
                "Some per-instance stat rows could not be matched to a request_state via perturbation. "
                f"Example(s): {unmatched[:20]}"
            )

        return joined_rows


    def request_state_index(
        self,
        *,
        allow_duplicates: bool = False,
    ) -> dict[tuple[Any, ...], dict[str, Any]]:
        """Index request_states by a composite key.

        Keys are tuples to keep them hashable and explicit.

        The primary key is:
            (instance_id, split, sub_split, train_trial_index, pert_name)

        Additionally, we provide a secondary key:
            ('fp', fingerprint)

        If duplicates are found and allow_duplicates=False, we keep the first
        and record the collisions in ``_cache['request_state_duplicates']``.
        """
        cache_key = ('request_state_index', allow_duplicates)
        if cache_key in self._cache:
            return self._cache[cache_key]

        idx: dict[tuple[Any, ...], dict[str, Any]] = {}
        dups: list[tuple[tuple[Any, ...], int, int]] = []
        request_states = self.scenario_state()['request_states']
        for i, rs in enumerate(request_states):
            k = self.request_state_key(rs)
            primary = (k.instance_id, k.split, k.sub_split, k.train_trial_index, k.pert_name)
            fp = ('fp', self.request_state_fingerprint(rs))
            for key in (primary, fp):
                if key in idx and not allow_duplicates:
                    dups.append((key, idx[key].get('_index', -1), i))
                    continue
                row = rs if isinstance(rs, dict) else {'value': rs}
                if isinstance(row, dict) and '_index' not in row:
                    row = dict(row)
                    row['_index'] = i
                idx[key] = row

        self._cache['request_state_duplicates'] = dups
        self._cache[cache_key] = idx
        return idx

    # --- Per-instance stats helpers ---------------------------------------

    def per_instance_stat_rows(self) -> list[dict[str, Any]]:
        """Alias for json per_instance_stats with caching in this object."""
        return self.per_instance_stats()

    def per_instance_stat_inventory(self) -> dict[str, Counter]:
        """Basic histogram summary over per_instance_stats.json."""
        cache_key = 'per_instance_stat_inventory'
        if cache_key in self._cache:
            return self._cache[cache_key]
        hist = {
            'num_rows': Counter(),
            'has_perturbation': Counter(),
            'train_trial_index': Counter(),
            'stat_counts_per_row': Counter(),
        }
        rows = self.per_instance_stat_rows()
        hist['num_rows']['total'] = len(rows)
        for r in rows:
            if not isinstance(r, dict):
                continue
            hist['has_perturbation'][bool(r.get('perturbation', None))] += 1
            tti = r.get('train_trial_index', None)
            hist['train_trial_index'][tti] += 1
            stats = r.get('stats', None)
            if isinstance(stats, list):
                hist['stat_counts_per_row'][len(stats)] += 1
        self._cache[cache_key] = hist
        return hist

    # --- Convenience summaries --------------------------------------------

    def summary_lite(self) -> dict[str, Any]:
        """Small stable summary for tables / debugging."""
        cache_key = 'summary_lite'
        if cache_key in self._cache:
            return self._cache[cache_key]

        spec = self.run_spec()
        scenario_state = self.scenario_state()

        out = {
            'run_spec_name': spec.get('name', None),
            'scenario_name': (spec.get('scenario_spec', {}) or {}).get('class_name', None),
            'num_stats': len(self.stats()),
            'num_request_states': len(scenario_state.get('request_states', []) or []),
            'num_per_instance_stats_rows': (
                len(self.per_instance_stats()) if (self.run.path / 'per_instance_stats.json').exists() else None
            ),
        }
        self._cache[cache_key] = out
        return out

    # --- Stat selection (HELM summarize-inspired) -------------------------

    def matching_stats(self, matcher: Union[MetricNameMatcherLite, Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Return all stats whose name matches ``matcher``.

        Args:
            matcher: Either :class:`MetricNameMatcherLite` or a dict with keys
                ``name``, ``split``, ``sub_split``, ``perturbation``.

        Returns:
            list[dict[str, Any]]: matching raw stat rows.
        """
        if isinstance(matcher, Mapping):
            matcher = MetricNameMatcherLite(**dict(matcher))
        out: list[dict[str, Any]] = []
        for row in self.stats():
            name_obj = row.get('name', None)
            if matcher.matches(name_obj):
                out.append(row)
        return out

    def get_unique_stat_by_matcher(
        self,
        matcher: Union[MetricNameMatcherLite, Mapping[str, Any]],
        *,
        quasi_exact_match_fallback: bool = True,
        aggregate_subsplits: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Return the single stat matching ``matcher``.

        Mirrors the most useful behavior of HELM's
        ``helm.benchmark.presentation.summarize.get_unique_stat_by_matcher``:

        * If no stats match and you're asking for ``quasi_exact_match``,
          optionally fall back to ``exact_match``.
        * If the matcher does not specify ``sub_split`` and
          ``aggregate_subsplits=True``, merge across sub-splits.

        Returns:
            Optional[dict[str, Any]]: a raw row or a synthetic merged row.

        Raises:
            KeyError: if more than one unique stat exists after optional
                aggregation.
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

        # Aggregate all sub_splits when sub_split is unspecified.
        if aggregate_subsplits and matcher.sub_split is None:
            # group by everything except sub_split
            buckets: dict[str, list[dict[str, Any]]] = {}
            for row in matching:
                name_obj = row.get('name', None)
                if not isinstance(name_obj, dict):
                    k = helm_hashers.nice_hash_id(('invalid_name', name_obj))
                else:
                    n2 = dict(name_obj)
                    n2['sub_split'] = None
                    k = stable_name_key(n2)
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
        matcher: Union[MetricNameMatcherLite, Mapping[str, Any]],
        *,
        quasi_exact_match_fallback: bool = True,
    ) -> dict[str, Any]:
        """Return a lightweight, Cell-like description for a matcher.

        This mirrors the *semantics* of HELM's ``Summarizer.create_cell`` but
        returns a plain dict.

        Returns:
            dict[str, Any]:
                - ``value``: float | None
                - ``case``: str ("no_match" | "count_zero" | "ok")
                - ``description``: str
                - ``stat``: Optional[dict[str, Any]] (raw or merged)
        """
        try:
            stat = self.get_unique_stat_by_matcher(
                matcher,
                quasi_exact_match_fallback=quasi_exact_match_fallback,
                aggregate_subsplits=True,
            )
        except KeyError as ex:
            return {
                'value': None,
                'case': 'ambiguous',
                'description': str(ex),
                'stat': None,
            }

        if stat is None:
            return {
                'value': None,
                'case': 'no_match',
                'description': 'No matching metrics',
                'stat': None,
            }

        count = int(stat.get('count', 0) or 0)
        mean = _safe_float(stat.get('mean', None))
        if count == 0 or mean is None:
            return {
                'value': None,
                'case': 'count_zero',
                'description': 'Matching metrics, but count == 0 (mean undefined)',
                'stat': stat,
            }

        return {
            'value': mean,
            'case': 'ok',
            'description': f"count={count}",
            'stat': stat,
        }


# --- Lightweight matcher + merging -----------------------------------------

@dataclass(frozen=True)
class MetricNameMatcherLite:
    """A minimal matcher for HELM's json stat-name objects.

    This is intentionally much simpler than HELM's ``MetricNameMatcher``.

    Fields left as ``None`` are treated as wildcards.

    Notes
    -----
    * If ``sub_split`` is ``None`` we treat it as "aggregate across sub_splits"
      (i.e. matches any sub_split, but aggregation happens after selection).
    """

    name: str
    split: Optional[str] = None
    sub_split: Optional[str] = None
    perturbation: Optional[str] = None

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
    """Merge a sequence of json stat rows.

    HELM's native merge is done via ``Stat.merge``. Here we implement only what
    we need for summaries: count-weighted mean.

    Returns:
        dict: A synthetic stat row with merged ``count`` and ``mean``.
    """
    total_count = 0
    total_sum = 0.0
    last_row: Optional[Mapping[str, Any]] = None
    for r in rows:
        last_row = r
        c = int(r.get('count', 0) or 0)
        m = _safe_float(r.get('mean', None))
        if c > 0 and m is not None:
            total_sum += m * c
            total_count += c
        else:
            # Even if mean is undefined, count contributes to the merged count.
            total_count += c

    merged: dict[str, Any] = {}
    if last_row is not None:
        merged.update(dict(last_row))
    merged['count'] = total_count
    merged['mean'] = (total_sum / total_count) if total_count > 0 else None
    return merged


# ---

def classify_metric(metric_name: Optional[str]) -> tuple[str, Optional[str]]:
    """Return (metric_class, matched_prefix).

    metric_class ∈ {'core', 'bookkeeping', 'untracked'}
    """
    if not metric_name:
        return ('untracked', None)
    for p in METRIC_PREFIXES.CORE_PREFIXES:
        if metric_name.startswith(p):
            return ('core', p)
    for p in METRIC_PREFIXES.BOOKKEEPING_PREFIXES:
        if metric_name.startswith(p):
            return ('bookkeeping', p)
    return ('untracked', None)


def metric_family(metric_name: Optional[str]) -> str:
    """A lightweight family heuristic used for summaries."""
    if not metric_name:
        return '?'
    # hierarchical families
    if metric_name.startswith('air_'):
        return 'air'
    if metric_name.startswith('bias_metric:'):
        return 'bias_metric'
    if metric_name.startswith('safety_'):
        return 'safety'
    if metric_name.startswith('bbq_'):
        return 'bbq'
    if '@' in metric_name:
        return metric_name.split('@', 1)[0]
    return metric_name.split('_', 1)[0].split(':', 1)[0]


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        y = float(x)
        if math.isnan(y):
            return None
        return y
    except Exception:
        return None


def stable_name_key(
    name_obj: Any,
    *,
    ignore_name_file_path: bool = True,
) -> str:
    """Stable hash of a stat["name"] dict.

    This uses the centralized hashing rules in :mod:`helm_hashers`.

    The optional canonicalization step removes environment-specific
    ``perturbation.name_file_path``.
    """
    if not isinstance(name_obj, dict):
        # Keep behavior stable but more readable for debugging.
        return helm_hashers.stat_name_id(name_obj)

    name_obj2 = ub.udict(name_obj).copy()
    if ignore_name_file_path:
        p = name_obj2.get('perturbation', None)
        if isinstance(p, dict) and 'name_file_path' in p:
            p = ub.udict(p).copy()
            p.pop('name_file_path', None)
            name_obj2['perturbation'] = p

    # Use the same semi-readable stable id as compare.py.
    return helm_hashers.stat_name_id(name_obj2)


def stable_name_key_with_count(
    name_obj: Any,
    count: Any,
    *,
    ignore_name_file_path: bool = True,
) -> str:
    """Stable stat-name id that also incorporates count.

    This is useful for the "coverage by (name + count)" checks.
    """
    if not isinstance(name_obj, dict):
        return helm_hashers.stat_name_id(name_obj, count=count)
    name_obj2 = ub.udict(name_obj).copy()
    if ignore_name_file_path:
        p = name_obj2.get('perturbation', None)
        if isinstance(p, dict) and 'name_file_path' in p:
            p = ub.udict(p).copy()
            p.pop('name_file_path', None)
            name_obj2['perturbation'] = p
    return helm_hashers.stat_name_id(name_obj2, count=count)


def stable_instance_fingerprint(obj: Any) -> str:
    """Stable fingerprint for instance/request payloads.

    We use :func:`helm_hashers.row_id` so fingerprints have a readable prefix.
    """
    return helm_hashers.row_id(obj, hint='fp')


@dataclass(frozen=True)
class StatMeta:
    """A compact, normalized view of a HELM stat row."""

    key: str
    metric: Optional[str]
    split: Optional[str]
    is_perturbed: bool
    pert_name: Optional[str]
    family: str
    metric_class: str
    matched_prefix: Optional[str]
    count: int
    mean: Optional[float]
    name_obj: Any
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class RequestStateKey:
    """A best-effort key to address a request/instance inside scenario_state."""

    instance_id: Optional[str]
    split: Optional[str]
    sub_split: Optional[str]
    train_trial_index: Optional[int]
    pert_name: Optional[str]
