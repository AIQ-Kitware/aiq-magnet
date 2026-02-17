"""magnet.backends.helm.rundiff.compare

This module focuses on *analysis* and *comparison* of HELM run outputs.

Design goals (aligned with notebook-style usage):

* Keep the small functional API for pipelines (e.g. ``compare_run_pair``
  producing row fields for Sankey bucketing).
* Provide an ergonomic, stateful object (``HelmRunDiff``) for interactive
  investigation. Call methods to incrementally compute / format deeper
  diagnostics without constantly recomputing inputs.
* Prefer ``ub.IndexableWalker`` for generic structure diffs, but keep
  purpose-built helpers for HELM's stat structures.
* Avoid copying code: centralize stat-name canonicalization, metric
  classification (core vs bookkeeping vs untracked), and coverage logic.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import ubelt as ub


# --- Metric registries ------------------------------------------------------

CORE_PREFIXES: Tuple[str, ...] = (
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


BOOKKEEPING_PREFIXES: Tuple[str, ...] = (
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
    # calibration / fitting plumbing (often depends on details)
    'ece_',
    'platt_',
    'selective_',
    # meta / dataset sizing
    'num_instances',
    'num_train_',
    'num_references',
)


def _stable_hash36(obj: Any) -> str:
    """Deterministic base36 hash used throughout this module."""
    # ub.hash_data already normalizes dict key ordering, which is what we want.
    return ub.hash_data(obj, base=36, hasher='sha256')


def _nice_hash_id(obj: Any, *, rawstr: str, keep_prefix: int = 25) -> str:
    """Semi-readable stable id.

    The returned id is always the same length as the underlying hash, but we
    splice in a readable prefix (metric/split/...) to make diffs easier to scan.
    """
    hashstr = _stable_hash36(obj)
    rawstr = rawstr.replace(' ', '')
    rawlen = len(rawstr)
    hashlen = len(hashstr)
    if rawlen < hashlen:
        return rawstr + hashstr[:-rawlen]
    else:
        return rawstr[:keep_prefix] + hashstr[:-keep_prefix]


def stat_name_id(name_obj: Any, *, count: Any = None) -> str:
    """Stable, semi-readable id for a stat name (and optionally its count)."""
    if not isinstance(name_obj, dict):
        raw = f"invalid_name,{ub.urepr(name_obj, compact=1, nl=0, nobr=1)},"
        obj = ('invalid_name', name_obj, count)
        return _nice_hash_id(obj, rawstr=raw)

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
    return _nice_hash_id(obj, rawstr=raw)


def row_id(row: Any, *, hint: str = 'row') -> str:
    """Stable-ish id for arbitrary rows (e.g. per-instance rows)."""
    raw = f"{hint},"
    return _nice_hash_id(row, rawstr=raw)


def is_core_metric_name(metric_name: str) -> bool:
    return any(metric_name.startswith(p) for p in CORE_PREFIXES)


def is_bookkeeping_metric_name(metric_name: str) -> bool:
    return any(metric_name.startswith(p) for p in BOOKKEEPING_PREFIXES)


def classify_metric(metric_name: Optional[str]) -> Tuple[str, Optional[str]]:
    """Return (metric_class, matched_prefix).

    metric_class ∈ {'core', 'bookkeeping', 'untracked'}
    """
    if not metric_name:
        return ('untracked', None)
    for p in CORE_PREFIXES:
        if metric_name.startswith(p):
            return ('core', p)
    for p in BOOKKEEPING_PREFIXES:
        if metric_name.startswith(p):
            return ('bookkeeping', p)
    return ('untracked', None)


def metric_family(metric_name: Optional[str]) -> str:
    """A lightweight family heuristic (first token before '_' or ':')."""
    if not metric_name:
        return '?'
    return metric_name.split('_', 1)[0].split(':', 1)[0]


# --- Stat canonicalization ---------------------------------------------------

NameObj = Dict[str, Any]
StatObj = Dict[str, Any]


def _pert_name(name_obj: Optional[NameObj]) -> Optional[str]:
    if isinstance(name_obj, dict) and name_obj.get('perturbation', None):
        return name_obj['perturbation'].get('name', 'pert')
    return None


def _stat_meta_from_name(name_obj: Optional[NameObj]) -> Dict[str, Any]:
    """Extract consistently useful facets from a HELM stat name dict."""
    metric = None
    split = None
    pn = None
    is_pert = False
    if isinstance(name_obj, dict):
        metric = name_obj.get('name', None)
        split = name_obj.get('split', None)
        pn = _pert_name(name_obj)
        is_pert = pn is not None
    mclass, mpref = classify_metric(metric)
    return {
        'metric': metric,
        'split': split,
        'is_perturbed': is_pert,
        'pert_name': pn,
        'family': metric_family(metric),
        'metric_class': mclass,
        'matched_prefix': mpref,
    }


def _stat_label_from_name(name_obj: Optional[NameObj]) -> str:
    if not isinstance(name_obj, dict):
        return '?'
    metric = name_obj.get('name', None)
    split = name_obj.get('split', None)
    pn = _pert_name(name_obj)
    if pn:
        return f'{metric} split={split} pert={pn}'
    return f'{metric} split={split}'


def _namekey(stat: StatObj, *, include_count: bool) -> str:
    """A key suitable for coverage comparisons.

    include_count=True makes mismatched counts appear in coverage.
    """
    name_obj = stat.get('name', None)
    if include_count:
        return stat_name_id(name_obj, count=stat.get('count', None))
    else:
        return stat_name_id(name_obj)


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


def _isclose(a: Any, b: Any, *, rel_tol=1e-4, abs_tol=1e-8) -> bool:
    fa = _safe_float(a)
    fb = _safe_float(b)
    if fa is None or fb is None:
        return a == b
    return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)


def _fmt_float(x: Any, sig: int = 4) -> str:
    fx = _safe_float(x)
    if fx is None:
        return str(x)
    return f'{fx:.{sig}g}'


def _pretty_path(path: Any) -> str:
    if isinstance(path, (list, tuple)):
        return '.'.join(map(str, path))
    return str(path)


def format_walker_diff(
    diff: Mapping[str, Any],
    *,
    label_a: str = 'A',
    label_b: str = 'B',
    max_items: int = 12,
    indent: str = '   ',
) -> List[str]:
    """Format ``ub.IndexableWalker.diff`` output into readable lines."""
    lines: List[str] = []
    if not diff:
        return lines

    faillist = diff.get('faillist', []) or []
    # Note: direction depends on how diff was computed.
    unique_b = sorted(diff.get('unique1', []) or [])
    unique_a = sorted(diff.get('unique2', []) or [])

    if faillist:
        lines.append(f'{indent}Value mismatches ({len(faillist)}):')
        for item in faillist[:max_items]:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                path, vb, va = item[0], item[1], item[2]
                p = _pretty_path(path)
                lines.append(
                    f'{indent}  {p}: {label_a}={ub.urepr(va, nl=0)}  {label_b}={ub.urepr(vb, nl=0)}'
                )
            else:
                lines.append(f'{indent}  {ub.urepr(item, nl=0)}')
        if len(faillist) > max_items:
            lines.append(f'{indent}  ... +{len(faillist) - max_items} more')

    if unique_a:
        lines.append(f'{indent}Only in {label_a} ({len(unique_a)}):')
        for item in unique_a[:max_items]:
            path = item[0] if isinstance(item, (list, tuple)) else item
            lines.append(f'{indent}  {_pretty_path(path)}')
        if len(unique_a) > max_items:
            lines.append(f'{indent}  ... +{len(unique_a) - max_items} more')

    if unique_b:
        lines.append(f'{indent}Only in {label_b} ({len(unique_b)}):')
        for item in unique_b[:max_items]:
            path = item[0] if isinstance(item, (list, tuple)) else item
            lines.append(f'{indent}  {_pretty_path(path)}')
        if len(unique_b) > max_items:
            lines.append(f'{indent}  ... +{len(unique_b) - max_items} more')

    return lines


# --- Coverage helpers -------------------------------------------------------


def coverage_counts(
    keys_a: Sequence[Any], keys_b: Sequence[Any]
) -> Dict[str, int]:
    sa = set(keys_a)
    sb = set(keys_b)
    isect = sa & sb
    union = sa | sb
    return {
        'nA': len(sa),
        'nB': len(sb),
        'isect': len(isect),
        'union': len(union),
        'onlyA': len(sa - sb),
        'onlyB': len(sb - sa),
    }


def _topk(counter: Counter, k: int = 8) -> List[Tuple[Any, int]]:
    return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:k]


# --- Stats indexing ---------------------------------------------------------


class _StatsIndex:
    """Index a HELM stat list for coverage + value comparisons."""

    def __init__(self, stats: Sequence[StatObj]):
        self.stats = [s for s in stats if isinstance(s, dict) and 'name' in s]

        # name-hash -> stat
        self.by_name: Dict[str, StatObj] = {}
        self.meta_by_name: Dict[str, Dict[str, Any]] = {}
        self._dupes: Counter = Counter()

        for s in self.stats:
            name_obj = s.get('name', None)
            if not isinstance(name_obj, dict):
                continue
            k = stat_name_id(name_obj)
            if k in self.by_name:
                self._dupes[k] += 1
                # Keep first by default; duplicates are rare but possible.
                continue
            self.by_name[k] = s
            self.meta_by_name[k] = _stat_meta_from_name(name_obj)

    def name_keys(self) -> List[str]:
        return list(self.by_name.keys())

    def name_count_keys(self) -> List[str]:
        return [
            _namekey(self.by_name[k], include_count=True)
            for k in self.by_name.keys()
        ]

    def dupes(self) -> Counter:
        return self._dupes


# --- Core metric comparison (pure function API) -----------------------------


def extract_core_stats(
    stats_list: Sequence[StatObj],
    *,
    require_unperturbed: bool = True,
    require_count_gt0: bool = True,
) -> Dict[Tuple[str, str], StatObj]:
    """Return {(metric_name, split): stat} for core metrics."""
    out: Dict[Tuple[str, str], StatObj] = {}
    for s in stats_list:
        if not isinstance(s, dict):
            continue
        if require_count_gt0 and s.get('count', 0) == 0:
            continue
        name = s.get('name', None)
        if not isinstance(name, dict):
            continue
        metric = name.get('name', None)
        if require_unperturbed and ('perturbation' in name):
            continue
        if not metric or not is_core_metric_name(metric):
            continue
        split = name.get('split', None)
        if split is None:
            continue
        out[(metric, split)] = s
    return out


def compare_core_stats(
    stats_a: Sequence[StatObj],
    stats_b: Sequence[StatObj],
    *,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-8,
    topn: int = 10,
) -> Dict[str, Any]:
    """Compare *unperturbed* core metrics between two stats lists."""

    core_a = extract_core_stats(stats_a)
    core_b = extract_core_stats(stats_b)
    keys_a = set(core_a.keys())
    keys_b = set(core_b.keys())
    isect = keys_a & keys_b
    union = keys_a | keys_b

    mism_rows = []
    agree = 0
    for k in sorted(isect):
        sa = core_a[k]
        sb = core_b[k]
        ma = sa.get('mean', None)
        mb = sb.get('mean', None)
        same = _isclose(ma, mb, rel_tol=rel_tol, abs_tol=abs_tol)
        if same:
            agree += 1
        else:
            fa = _safe_float(ma)
            fb = _safe_float(mb)
            absd = None if fa is None or fb is None else abs(fa - fb)
            reld = (
                None if absd is None or fb is None else absd / (abs(fb) + 1e-12)
            )
            mism_rows.append(
                {
                    'key': k,
                    'metric': k[0],
                    'split': k[1],
                    'mean_a': ma,
                    'mean_b': mb,
                    'abs': absd,
                    'rel': reld,
                }
            )

    mism_rows = sorted(
        mism_rows, key=lambda r: -(r['abs'] if r['abs'] is not None else -1)
    )
    top_rows = mism_rows[:topn]

    cov = len(isect) / max(len(union), 1)
    agr = agree / max(len(isect), 1)
    return {
        'nA': len(keys_a),
        'nB': len(keys_b),
        'isect': len(isect),
        'union': len(union),
        'core_coverage': cov,
        'core_agreement': agr,
        'core_mismatches': len(mism_rows),
        'core_top_mismatches': top_rows,
    }


def format_core_report(
    core: Dict[str, Any], *, title: str = 'CORE METRIC DIFF', topn: int = 8
) -> str:
    nA = core.get('nA', None)
    nB = core.get('nB', None)
    isect = core.get('isect', None)
    union = core.get('union', None)
    cov = core.get('core_coverage', None)
    agree = core.get('core_agreement', None)
    mism = core.get('core_mismatches', None)

    lines = []
    lines.append('=' * 80)
    lines.append(title)
    lines.append(f'nA={nA} nB={nB}  isect={isect} union={union}')
    lines.append(
        f'coverage={_fmt_float(cov, 4)}  agreement(isclose)={_fmt_float(agree, 4)}  mismatches={mism}'
    )

    rows = core.get('core_top_mismatches', []) or []
    if rows:
        lines.append('')
        lines.append(f'Top {min(topn, len(rows))} mean deltas:')
        for r in rows[:topn]:
            metric = r.get('metric', None)
            split = r.get('split', None)
            absd = r.get('abs', None)
            reld = r.get('rel', None)
            ma = r.get('mean_a', None)
            mb = r.get('mean_b', None)
            lines.append(
                f'  abs={_fmt_float(absd, 4)} rel={_fmt_float(reld, 4)} | {metric} split={split} | A={_fmt_float(ma, 6)} B={_fmt_float(mb, 6)}'
            )
    lines.append('=' * 80)
    return '\n'.join(lines)


# --- Sankey-oriented comparison (pure function API) -------------------------


def compare_bucket_indices(
    A: Sequence[StatObj],
    B: Sequence[StatObj],
    *,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-8,
) -> Dict[str, Any]:
    """Compare the *entire* stats list using indexable walkers.

    This intentionally remains coarse and fast: it's for bucketing (Sankey).
    """
    A_sorted = A
    B_sorted = B
    wa = ub.IndexableWalker(A_sorted)
    wb = ub.IndexableWalker(B_sorted)
    diff = wa.diff(wb, rel_tol=rel_tol, abs_tol=abs_tol)
    out = {
        'similarity': diff.get('similarity', None),
        'num_approximations': diff.get('num_approximations', None),
        'num_differences': diff.get('num_differences', None),
        'num_similarities': diff.get('num_similarities', None),
        'n_unique1': len(diff.get('unique1', []) or []),
        'n_unique2': len(diff.get('unique2', []) or []),
        'n_faillist': len(diff.get('faillist', []) or []),
        'n_passlist': len(diff.get('passlist', []) or []),
    }
    return out


def summarize_for_sankey(comp_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Map a coarse diff summary into a few labels for Sankey bucketing."""
    # The main funnel you were using:
    sim = comp_summary.get('similarity', None)
    n_unique1 = comp_summary.get('n_unique1', 0)
    n_unique2 = comp_summary.get('n_unique2', 0)

    # Simple buckets. You can refine in the caller.
    if sim is None:
        bucket = 'unknown'
    else:
        if sim >= 0.99:
            bucket = 'base_task: full'
        elif sim >= 0.95:
            bucket = 'base_task: partial'
        elif sim >= 0.80:
            bucket = 'base_task: low'
        else:
            bucket = 'base_task: very low'

    if (n_unique1 + n_unique2) > 0:
        bucket = 'base_task: coverage mismatch'

    return {
        'agreement_bucket_base_task': bucket,
    }


def compare_run_pair(
    stats_a: Sequence[StatObj],
    stats_b: Sequence[StatObj],
    *,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-8,
) -> Dict[str, Any]:
    """Convenience wrapper used by the notebook script.

    Returns a dict you can ``helm_row.update(out)`` with.
    """
    bucket = compare_bucket_indices(
        stats_a, stats_b, rel_tol=rel_tol, abs_tol=abs_tol
    )
    out = {
        'diffinfo': bucket,
    }
    out.update(summarize_for_sankey(bucket))
    out['core'] = compare_core_stats(
        stats_a, stats_b, rel_tol=rel_tol, abs_tol=abs_tol
    )
    return out


# --- HelmRunDiff: stateful comparison for HelmRun objects -----------------------


class HelmRunDiff:
    """Stateful comparison of two ``HelmRun`` objects.

    This class is intended to be used interactively:
        rd = HelmRunDiff(run_a, run_b, a_name='HELM', b_name='kwdagger')
        print(rd.summary_l1())
        print(rd.summary_values())

    It caches reads from each run (A/B) and caches computed comparisons.
    """

    def __init__(
        self,
        run_a: Any,
        run_b: Any,
        *,
        a_name: str = 'A',
        b_name: str = 'B',
    ) -> None:
        self.run_a = run_a
        self.run_b = run_b
        self.a_name = a_name
        self.b_name = b_name

        self._a_cache: Dict[str, Any] = {}
        self._b_cache: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}

    # -- low-level cached accessors

    def _get_a(self, key: str, func):
        if key not in self._a_cache:
            self._a_cache[key] = func()
        return self._a_cache[key]

    def _get_b(self, key: str, func):
        if key not in self._b_cache:
            self._b_cache[key] = func()
        return self._b_cache[key]

    # -- run JSON access

    def run_spec_a(self) -> Dict[str, Any]:
        return self._get_a('run_spec', lambda: self.run_a.json.run_spec())

    def run_spec_b(self) -> Dict[str, Any]:
        return self._get_b('run_spec', lambda: self.run_b.json.run_spec())

    def scenario_a(self) -> Any:
        return self._get_a('scenario', lambda: self.run_a.json.scenario())

    def scenario_b(self) -> Any:
        return self._get_b('scenario', lambda: self.run_b.json.scenario())

    def scenario_state_a(self) -> Any:
        return self._get_a(
            'scenario_state', lambda: self.run_a.json.scenario_state()
        )

    def scenario_state_b(self) -> Any:
        return self._get_b(
            'scenario_state', lambda: self.run_b.json.scenario_state()
        )

    def stats_a(self) -> List[StatObj]:
        return self._get_a('stats', lambda: self.run_a.json.stats())

    def stats_b(self) -> List[StatObj]:
        return self._get_b('stats', lambda: self.run_b.json.stats())

    def per_instance_stats_a(self) -> Any:
        return self._get_a(
            'per_instance_stats', lambda: self.run_a.json.per_instance_stats()
        )

    def per_instance_stats_b(self) -> Any:
        return self._get_b(
            'per_instance_stats', lambda: self.run_b.json.per_instance_stats()
        )

    # -- derived cached objects

    def _stats_index_a(self) -> _StatsIndex:
        return self._get_a('stats_index', lambda: _StatsIndex(self.stats_a()))

    def _stats_index_b(self) -> _StatsIndex:
        return self._get_b('stats_index', lambda: _StatsIndex(self.stats_b()))

    # -- reporting helpers

    @staticmethod
    def _mark(ok: Optional[bool]) -> str:
        if ok is True:
            return '✅'
        if ok is False:
            return '❌'
        return '⚠️'

    def report_core(self, *, rel_tol=1e-4, abs_tol=1e-8, topn=10):
        core = compare_core_stats(self.stats_a(), self.stats_b(),
                                  rel_tol=rel_tol, abs_tol=abs_tol, topn=topn)
        return format_core_report(core, topn=topn)

    # ------------------------------------------------------------------
    # Summary Level 1: structure/coverage only (no value checking)
    # ------------------------------------------------------------------

    def summary_l1(
        self, *, max_show: int = 220, max_diff_items: int = 10
    ) -> str:
        lines: List[str] = []
        lines.append('=' * 80)
        lines.append('RUNDIFF SUMMARY (L1)')
        lines.append('')

        # 0) run spec name
        name_a = self.run_spec_a().get('name', None)
        name_b = self.run_spec_b().get('name', None)
        same_name = (name_a == name_b) and (name_a is not None)
        if same_name:
            lines.append(f'0) Run spec name: {self._mark(True)} {name_a}')
        else:
            lines.append(f'0) Run spec name: {self._mark(False)}')
            lines.append(f'   {self.a_name}: {name_a}')
            lines.append(f'   {self.b_name}: {name_b}')
        lines.append('')

        # 1) run spec full
        spec_a = self.run_spec_a()
        spec_b = self.run_spec_b()
        same_spec = ub.hash_data(spec_a, base=36) == ub.hash_data(
            spec_b, base=36
        )
        if same_spec:
            lines.append(f'1) Run spec: {self._mark(True)}')
        else:
            lines.append(f'1) Run spec: {self._mark(False)}')

        import kwutil
        spec_repr_a = kwutil.slugify_ext.smart_truncate(ub.urepr(spec_a, compact=1, nl=0, nobr=1), max_length=max_show)
        spec_repr_b = kwutil.slugify_ext.smart_truncate(ub.urepr(spec_b, compact=1, nl=0, nobr=1), max_length=max_show)

        lines.append(f'   {self.a_name}: {spec_repr_a}')
        lines.append(f'   {self.b_name}: {spec_repr_b}')
        if not same_spec:
            diff = ub.IndexableWalker(spec_b).diff(spec_a)
            # Remove passlist to avoid noise.
            diff = ub.udict(diff) - {'passlist'}
            lines.extend(
                format_walker_diff(
                    diff,
                    label_a=self.a_name,
                    label_b=self.b_name,
                    max_items=max_diff_items,
                )
            )
        lines.append('')

        # 2) scenario
        scen_a = self.scenario_a()
        scen_b = self.scenario_b()
        if scen_a is None and scen_b is None:
            lines.append('2) Scenario: ⚠️ (unknown)')
        else:
            same_scen = scen_a == scen_b
            lines.append(f'2) Scenario: {self._mark(same_scen)}')
        lines.append(f'   {self.a_name}: {scen_a}')
        lines.append(f'   {self.b_name}: {scen_b}')
        if scen_a is not None and scen_b is not None and scen_a != scen_b:
            diff = ub.IndexableWalker(scen_a).diff(scen_b)
            diff = ub.udict(diff) - {'passlist'}
            lines.extend(
                format_walker_diff(
                    diff,
                    label_a=self.a_name,
                    label_b=self.b_name,
                    max_items=max_diff_items,
                )
            )
        lines.append('')

        # 3) stats coverage by name
        idx_a = self._stats_index_a()
        idx_b = self._stats_index_b()
        cov = coverage_counts(idx_a.name_keys(), idx_b.name_keys())
        lines.append('3) Stats coverage by name (ignoring count/value):')
        lines.append(
            f'   nA={cov["nA"]} nB={cov["nB"]}  isect={cov["isect"]} union={cov["union"]}  onlyA={cov["onlyA"]} onlyB={cov["onlyB"]}'
        )
        if idx_a.dupes() or idx_b.dupes():
            lines.append(
                f'   ⚠️ duplicate stat names: A={sum(idx_a.dupes().values())} B={sum(idx_b.dupes().values())}'
            )
        lines.append('')

        # NEW: class breakdown using the SAME coverage_counts()
        def _filter_by_class(keys, idx, want):
            # idx.meta_by_name maps name-key -> meta dict (with metric_class)
            out = []
            for k in keys:
                meta = idx.meta_by_name.get(k, None)
                mclass = meta.get('metric_class', 'untracked') if meta else 'untracked'
                if mclass == want:
                    out.append(k)
            return out

        lines.append("   Breakdown by metric_class (coverage_counts on filtered keys):")
        for mclass in ('core', 'bookkeeping', 'untracked'):
            ka = _filter_by_class(idx_a.name_keys(), idx_a, mclass)
            kb = _filter_by_class(idx_b.name_keys(), idx_b, mclass)
            cc = coverage_counts(ka, kb)
            lines.append(
                f"     {mclass:11s}: nA={cc['nA']} nB={cc['nB']} isect={cc['isect']} onlyA={cc['onlyA']} onlyB={cc['onlyB']}"
            )
        lines.append("")

        # 4) stats coverage by (name+count)
        cov2 = coverage_counts(idx_a.name_count_keys(), idx_b.name_count_keys())
        lines.append('4) Stats coverage by (name + count) (ignoring value):')
        lines.append(
            f'   nA={cov2["nA"]} nB={cov2["nB"]}  isect={cov2["isect"]} union={cov2["union"]}  onlyA={cov2["onlyA"]} onlyB={cov2["onlyB"]}'
        )
        lines.append('')

        # 5) instance coverage (best-effort)
        try:
            pia = self.per_instance_stats_a()
            pib = self.per_instance_stats_b()
            ids_a = self._extract_instance_ids(pia)
            ids_b = self._extract_instance_ids(pib)
            covi = coverage_counts(ids_a, ids_b)
            lines.append('5) Instance coverage (per_instance_stats overlap):')
            lines.append(
                f'   nA={covi["nA"]} nB={covi["nB"]}  isect={covi["isect"]} union={covi["union"]}  onlyA={covi["onlyA"]} onlyB={covi["onlyB"]}'
            )
        except Exception as ex:
            lines.append('5) Instance coverage: ⚠️ (could not compute)')
            lines.append(f'   reason: {type(ex).__name__}: {ex}')

        lines.append('=' * 80)
        return '\n'.join(lines)

    @staticmethod
    def _extract_instance_ids(per_instance_stats: Any) -> List[Any]:
        """Best-effort extraction of per-instance IDs.

        HELM has changed this schema across versions; keep robust.
        Returns a list of hashable identifiers (tuples or strings).
        """
        ids = []
        for row in per_instance_stats:
            if isinstance(row, dict):
                if 'instance_id' in row:
                    ids.append(('instance_id', row['instance_id']))
                elif 'id' in row:
                    ids.append(('id', row['id']))
                else:
                    # fallback: hash the whole row (not great, but provides overlap signal)
                    ids.append(('hash', row_id(row, hint='inst')))
            else:
                ids.append(('hash', row_id(row, hint='inst')))
        return ids

    # ------------------------------------------------------------------
    # Summary Level 2: mean-value checking (split by perturbed × class)
    # ------------------------------------------------------------------

    def value_summary(
        self,
        *,
        rel_tol: float = 1e-4,
        abs_tol: float = 1e-8,
        require_count_gt0: bool = True,
    ) -> Dict[str, Any]:
        """Compute value-aware stats summary for intersecting stat names."""
        key = ('value_summary', rel_tol, abs_tol, require_count_gt0)
        if key in self._cache:
            return self._cache[key]

        idx_a = self._stats_index_a()
        idx_b = self._stats_index_b()
        isect = sorted(set(idx_a.by_name) & set(idx_b.by_name))

        def new_bucket():
            return {
                'total': 0,  # total intersecting names in this bucket
                'comparable': 0,  # count>0 (if requested) and both have mean
                'matched': 0,
                'mismatched': 0,
                'families_all': Counter(),
                'families_match': Counter(),
                'families_mism': Counter(),
                'prefix_all': Counter(),
                'prefix_match': Counter(),
                'prefix_mism': Counter(),
                'metric_all': Counter(),
                'metric_mism': Counter(),
                'metric_match': Counter(),
                'pert_all': Counter(),
                'pert_mism': Counter(),
                'rows_mism': [],
            }

        groups = {
            'unperturbed': {
                'core': new_bucket(),
                'bookkeeping': new_bucket(),
                'untracked': new_bucket(),
            },
            'perturbed': {
                'core': new_bucket(),
                'bookkeeping': new_bucket(),
                'untracked': new_bucket(),
            },
        }

        for k in isect:
            sa = idx_a.by_name[k]
            sb = idx_b.by_name[k]
            meta = idx_a.meta_by_name.get(
                k, _stat_meta_from_name(sa.get('name', None))
            )

            pg = 'perturbed' if meta['is_perturbed'] else 'unperturbed'
            mc = meta['metric_class']
            bucket = groups[pg][mc]
            bucket['total'] += 1

            fam = meta['family']
            pref = meta['matched_prefix'] or fam
            metric = meta['metric']
            pn = meta['pert_name']

            bucket['families_all'][fam] += 1
            bucket['prefix_all'][pref] += 1
            if metric:
                bucket['metric_all'][metric] += 1
            if pn:
                bucket['pert_all'][pn] += 1

            # decide if we can compare
            if require_count_gt0:
                if (sa.get('count', 0) == 0) or (sb.get('count', 0) == 0):
                    continue
            ma = sa.get('mean', None)
            mb = sb.get('mean', None)
            if ma is None and mb is None:
                continue

            bucket['comparable'] += 1
            same = _isclose(ma, mb, rel_tol=rel_tol, abs_tol=abs_tol)
            if same:
                bucket['matched'] += 1
                bucket['families_match'][fam] += 1
                bucket['prefix_match'][pref] += 1
                if metric:
                    bucket['metric_match'][metric] += 1
            else:
                bucket['mismatched'] += 1
                bucket['families_mism'][fam] += 1
                bucket['prefix_mism'][pref] += 1
                if metric:
                    bucket['metric_mism'][metric] += 1
                if pn:
                    bucket['pert_mism'][pn] += 1

                fa = _safe_float(ma)
                fb = _safe_float(mb)
                absd = None if fa is None or fb is None else abs(fa - fb)
                reld = (
                    None
                    if absd is None or fb is None
                    else absd / (abs(fb) + 1e-12)
                )
                bucket['rows_mism'].append(
                    {
                        'name': sa.get('name', None),
                        'meta': meta,
                        'mean_a': ma,
                        'mean_b': mb,
                        'abs': absd,
                        'rel': reld,
                    }
                )

        out = {
            'isect_names': len(isect),
            'groups': groups,
            'rel_tol': rel_tol,
            'abs_tol': abs_tol,
            'require_count_gt0': require_count_gt0,
        }
        self._cache[key] = out
        return out

    def summary_values(
        self,
        *,
        rel_tol: float = 1e-4,
        abs_tol: float = 1e-8,
        topn: int = 6,
        require_count_gt0: bool = True,
    ) -> str:
        """Human-readable report for ``value_summary``."""
        vs = self.value_summary(
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            require_count_gt0=require_count_gt0,
        )
        groups = vs['groups']
        lines: List[str] = []
        lines.append(
            '6) Value check on intersecting stats (compare mean only):'
        )
        lines.append(
            '   split by perturbation × (core / bookkeeping / untracked)'
        )
        lines.append(f'   comparable requires count>0: {require_count_gt0}')

        def summarize_bucket(label: str, bucket: Dict[str, Any]) -> List[str]:
            out: List[str] = []
            total = bucket['total']
            comp = bucket['comparable']
            mism = bucket['mismatched']
            match = bucket['matched']
            agr = match / max(comp, 1)
            out.append(
                f'     {label}: total={total} comparable={comp} matched={match} mismatched={mism} agreement={_fmt_float(agr, 4)}'
            )
            if total == 0:
                return out

            out.append(
                f'       top families(all): {_topk(bucket["families_all"], 6)}'
            )
            if comp:
                out.append(
                    f'       top families(match): {_topk(bucket["families_match"], 6)}'
                )
                out.append(
                    f'       top families(mism): {_topk(bucket["families_mism"], 6)}'
                )

            out.append(
                f'       top prefixes(all): {_topk(bucket["prefix_all"], 6)}'
            )
            if mism:
                out.append(
                    f'       top prefixes(mism): {_topk(bucket["prefix_mism"], 6)}'
                )

            if bucket['pert_all']:
                out.append(
                    f'       top perturbations(all): {_topk(bucket["pert_all"], 6)}'
                )
            if bucket['pert_mism']:
                out.append(
                    f'       top perturbations(mism): {_topk(bucket["pert_mism"], 6)}'
                )

            if label == 'untracked':
                out.append(
                    f'       top metric names(all): {_topk(bucket["metric_all"], 6)}'
                )
                if mism:
                    out.append(
                        f'       top metric names(mism): {_topk(bucket["metric_mism"], 6)}'
                    )

            if mism:
                rows = sorted(
                    bucket['rows_mism'],
                    key=lambda r: -(r['abs'] if r['abs'] is not None else -1),
                )
                out.append(
                    f'       Top {min(topn, len(rows))} abs(mean) deltas:'
                )
                for r in rows[:topn]:
                    name_obj = r['name']
                    meta = r['meta']
                    pn = meta['pert_name'] or '-'
                    out.append(
                        '         - '
                        f'abs={_fmt_float(r["abs"], 4)} rel={_fmt_float(r["rel"], 4)} | '
                        f'fam={meta["family"]} class={meta["metric_class"]} | '
                        f'split={meta["split"]} pert={pn} | '
                        f'{_stat_label_from_name(name_obj)} | '
                        f'{self.a_name}={_fmt_float(r["mean_a"], 6)} {self.b_name}={_fmt_float(r["mean_b"], 6)}'
                    )
            return out

        for pg in ['unperturbed', 'perturbed']:
            lines.append(f'   {pg}:')
            lines.extend(summarize_bucket('core', groups[pg]['core']))
            lines.extend(
                summarize_bucket('bookkeeping', groups[pg]['bookkeeping'])
            )
            lines.extend(summarize_bucket('untracked', groups[pg]['untracked']))
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Instance-level drilldown for CORE metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _instance_key(row: Any) -> Any:
        """
        Best-effort stable per-instance key.

        Prefer explicit IDs, otherwise fall back to a hash of row content.
        """
        if isinstance(row, dict):
            for k in ('instance_id', 'id', 'request_id', 'uid', 'example_id'):
                assert k == 'instance_id', 'should not be anything else'
                if k in row:
                    return (k, row[k])
        # fallback: stable hash of the row
        return ('hash', row_id(row, hint='inst'))

    @staticmethod
    def _iter_stat_like_dicts(obj: Any):
        """
        Yield dicts that look like HELM stat objects from arbitrary nested structures.
        We use this to be schema-robust across HELM versions.
        """
        walker = ub.IndexableWalker(obj)
        for path, val in walker:
            if isinstance(val, dict) and 'name' in val and isinstance(val.get('name', None), dict):
                # must have a metric name
                n = val['name'].get('name', None)
                if n is None:
                    continue
                # per-instance rows sometimes store values under 'value' or 'mean'
                if ('value' in val) or ('mean' in val) or ('sum' in val):
                    yield val

    @staticmethod
    def _stat_value(stat: Dict[str, Any]) -> Any:
        """Best-effort numeric value for a stat-like dict."""
        if 'value' in stat:
            return stat.get('value', None)
        if 'mean' in stat:
            return stat.get('mean', None)
        if 'sum' in stat and stat.get('count', 0):
            # sometimes mean can be reconstructed
            try:
                return stat['sum'] / stat['count']
            except Exception:
                return stat.get('sum', None)
        return None

    def _per_instance_core_index(self, which: str) -> Dict[Any, Dict[Tuple[str, str, Optional[str]], Any]]:
        """
        Build:
            inst_key -> {(metric_name, split, pert_name_or_None): value}
        Only for CORE_PREFIXES metrics.

        Cached under _a_cache/_b_cache depending on `which`.
        """
        cache_key = 'per_instance_core_index'
        if which == 'a':
            store = self._a_cache
            get_pis = self.per_instance_stats_a
        elif which == 'b':
            store = self._b_cache
            get_pis = self.per_instance_stats_b
        else:
            raise KeyError(which)

        if cache_key in store:
            return store[cache_key]

        per_inst = get_pis()
        out: Dict[Any, Dict[Tuple[str, str, Optional[str]], Any]] = {}

        for row in per_inst:
            ...
            ik = self._instance_key(row)
            slot = out.setdefault(ik, {})
            # Find stat-like dicts inside this row, schema-agnostic
            for stat in self._iter_stat_like_dicts(row):
                name_obj = stat.get('name', None)
                metric = name_obj.get('name', None) if isinstance(name_obj, dict) else None
                if not metric or not is_core_metric_name(metric):
                    continue
                split = name_obj.get('split', None) if isinstance(name_obj, dict) else None
                if split is None:
                    split = '?'  # be robust
                pn = _pert_name(name_obj)  # None if unperturbed
                v = self._stat_value(stat)
                # Keep last-write-wins if duplicates occur within row
                slot[(metric, split, pn)] = v

        store[cache_key] = out
        return out

    def drilldown_core_metric_instances(
        self,
        *,
        topn: int = 10,
        rel_tol: float = 1e-4,
        abs_tol: float = 1e-8,
        min_comparable: int = 5,
        only_show_different: bool = True,
    ) -> str:
        """
        Drill down into per-instance CORE metric differences.

        For each core metric (metric_name, split), separately for:
            - unperturbed (pn is None)
            - perturbed (pn is not None, grouped by pn)

        Report:
            comparable, same, different, agreement
            top-N most different instances (abs delta) with A/B values

        Notes:
            - For boolean-ish per-instance values, abs-delta will be 1.0 when different.
            - If only perturbed instances differ, this is flagged as "less concerning".
        """
        cache_key = ('drilldown_core_metric_instances', topn, rel_tol, abs_tol, min_comparable, only_show_different)
        if cache_key in self._cache:
            return self._cache[cache_key]

        idx_a = self._per_instance_core_index('a')
        idx_b = self._per_instance_core_index('b')

        insts_a = set(idx_a.keys())
        insts_b = set(idx_b.keys())
        inst_isect = insts_a & insts_b

        # Collect all metric-keys observed (metric, split, pn)
        all_keys = set()
        for ik in inst_isect:
            all_keys.update(idx_a.get(ik, {}).keys())
            all_keys.update(idx_b.get(ik, {}).keys())

        # Partition into unperturbed vs perturbed
        unpert_keys = sorted([k for k in all_keys if k[2] is None])
        pert_keys = sorted([k for k in all_keys if k[2] is not None])

        def _compare_key(metric_key: Tuple[str, str, Optional[str]]):
            metric, split, pn = metric_key
            comparable = 0
            same = 0
            diffs = []
            for ik in inst_isect:
                va = idx_a.get(ik, {}).get(metric_key, None)
                vb = idx_b.get(ik, {}).get(metric_key, None)
                if va is None or vb is None:
                    continue
                comparable += 1
                if _isclose(va, vb, rel_tol=rel_tol, abs_tol=abs_tol):
                    same += 1
                else:
                    fa = _safe_float(va)
                    fb = _safe_float(vb)
                    absd = None if (fa is None or fb is None) else abs(fa - fb)
                    # For non-numerics, absd may be None; still keep but sort last.
                    diffs.append((absd, ik, va, vb))

            different = comparable - same
            agreement = same / max(comparable, 1)
            # sort diffs by absd desc, None last
            diffs = sorted(
                diffs,
                key=lambda t: (-(t[0] if t[0] is not None else -1), str(t[1])),
            )
            return {
                'metric': metric,
                'split': split,
                'pert_name': pn,
                'comparable': comparable,
                'same': same,
                'different': different,
                'agreement': agreement,
                'top_diffs': diffs[:topn],
            }

        # Group comparisons by (metric, split) for unperturbed
        unpert_by_ms: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for metric, split, pn in unpert_keys:
            rep = _compare_key((metric, split, pn))
            unpert_by_ms[(metric, split)] = rep

        # Group comparisons by (metric, split) for perturbed, and within that by pert_name
        pert_by_ms: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for metric, split, pn in pert_keys:
            rep = _compare_key((metric, split, pn))
            key_ms = (metric, split)
            slot = pert_by_ms.setdefault(
                key_ms,
                {
                    'metric': metric,
                    'split': split,
                    'by_pert': {},   # pn -> rep
                    'comparable': 0,
                    'same': 0,
                    'different': 0,
                },
            )
            slot['by_pert'][pn] = rep
            slot['comparable'] += rep['comparable']
            slot['same'] += rep['same']
            slot['different'] += rep['different']

        # Determine “only perturbed differs” overall signal
        total_unpert_diff = sum(v['different'] for v in unpert_by_ms.values())
        # total_unpert_comp = sum(v['comparable'] for v in unpert_by_ms.values())
        total_pert_diff = sum(v['different'] for v in pert_by_ms.values())
        # total_pert_comp = sum(v['comparable'] for v in pert_by_ms.values())

        only_perturbed_diff = (total_unpert_diff == 0) and (total_pert_diff > 0)

        # Format
        lines: List[str] = []
        lines.append('=' * 80)
        lines.append('CORE METRIC INSTANCE DRILLDOWN')
        lines.append(f'Runs: {self.a_name} vs {self.b_name}')
        lines.append(f'Instance overlap: isect={len(inst_isect)} (A={len(insts_a)} B={len(insts_b)})')
        lines.append(f'only_perturbed_diff={only_perturbed_diff}')
        lines.append('')

        def _fmt_rep(rep: Dict[str, Any], indent: str = '  ') -> List[str]:
            out = []
            comp = rep['comparable']
            if comp < min_comparable:
                return out
            if only_show_different and rep['different'] == 0:
                return out

            metric = rep['metric']
            split = rep['split']
            pn = rep['pert_name']
            tag = 'unpert' if pn is None else f'pert={pn}'
            out.append(
                f"{indent}{metric} split={split} {tag}: comparable={comp} same={rep['same']} different={rep['different']} agreement={_fmt_float(rep['agreement'], 4)}"
            )
            if rep['different'] > 0 and rep['top_diffs']:
                out.append(f"{indent}  top {min(topn, len(rep['top_diffs']))} differing instances:")
                row_map_a = self._per_instance_row_map('a')
                row_map_b = self._per_instance_row_map('b')

                for absd, ik, va, vb in rep['top_diffs']:
                    absd_s = _fmt_float(absd, 4) if absd is not None else '?'
                    out.append(
                        f"{indent}    - inst={ik} absΔ={absd_s} | {self.a_name}={_fmt_float(va, 4)} {self.b_name}={_fmt_float(vb, 4)}"
                    )

                    ra = row_map_a.get(ik, None)
                    rb = row_map_b.get(ik, None)

                    # If the instance rows are exactly the same object/structure, print once.
                    # Otherwise print both (this helps catch subtle schema / content mismatches).
                    same_row = False
                    try:
                        same_row = (ra == rb)
                    except Exception:
                        same_row = False

                    if same_row:
                        out.append(f"{indent}      instance (same in both):")
                        out.extend(self._fmt_instance_row(ra, indent=indent + "        "))
                    else:
                        out.append(f"{indent}      instance differs:")
                        out.append(f"{indent}        {self.a_name}:")
                        out.extend(self._fmt_instance_row(ra, indent=indent + "          "))
                        out.append(f"{indent}        {self.b_name}:")
                        out.extend(self._fmt_instance_row(rb, indent=indent + "          "))
            return out

        # Unperturbed section
        lines.append('Unperturbed core metrics (per-instance):')
        # Sort by most differences, then by metric name
        for (metric, split), rep in sorted(unpert_by_ms.items(), key=lambda kv: (-kv[1]['different'], kv[0][0], kv[0][1])):
            lines.extend(_fmt_rep(rep, indent='  '))
        if len(lines) == 0 or lines[-1] != '':
            lines.append('')

        # Perturbed section
        lines.append('Perturbed core metrics (per-instance):')
        for (metric, split), agg in sorted(pert_by_ms.items(), key=lambda kv: (-kv[1]['different'], kv[0][0], kv[0][1])):
            comp = agg['comparable']
            if comp < min_comparable:
                continue
            if only_show_different and agg['different'] == 0:
                continue
            agreement = agg['same'] / max(agg['comparable'], 1)
            lines.append(
                f"  {metric} split={split}: comparable={comp} same={agg['same']} different={agg['different']} agreement={_fmt_float(agreement, 4)}"
            )
            # show breakdown by perturbation name, sorted by differences
            for pn, rep in sorted(agg['by_pert'].items(), key=lambda kv: (-kv[1]['different'], str(kv[0]))):
                lines.extend(_fmt_rep(rep, indent='    '))

        lines.append('=' * 80)
        text = '\n'.join(lines)
        self._cache[cache_key] = text
        return text

    def _per_instance_row_map(self, which: str):
        """
        Cache: inst_key -> raw per_instance_stats row (dict-like)
        """
        cache_key = 'per_instance_row_map'
        if which == 'a':
            store = self._a_cache
            get_pis = self.per_instance_stats_a
        elif which == 'b':
            store = self._b_cache
            get_pis = self.per_instance_stats_b
        else:
            raise KeyError(which)

        if cache_key in store:
            return store[cache_key]

        per_inst = get_pis()
        row_map = {}
        for row in per_inst:
            ik = self._instance_key(row)
            # first-win is fine (duplicates should be rare); change to last-win if you prefer
            row_map.setdefault(ik, row)

        store[cache_key] = row_map
        return row_map

    @staticmethod
    def _fmt_instance_row(row, *, maxlen=220, indent="      "):
        """
        Compact but informative representation of an instance row.
        You can tweak what you want shown over time.
        """
        if row is None:
            return [indent + "None"]
        try:
            txt = ub.urepr(row, compact=1, nl=0, nobr=1)
        except Exception:
            txt = repr(row)
        if len(txt) > maxlen:
            txt = txt[:maxlen] + " ..."
        return [indent + txt]

    def _build_instance_lookup(self, which: str, *, id_fields=None):
        """
        Build + cache lookup tables for per_instance_stats.

        Caches (per-side):
          - inst_row_map: inst_key -> row
          - rawid_to_instkey: (field, raw_id) -> inst_key   (only if unique)
          - duplicates: dict with details

        Duplicate policy:
          - If multiple rows share the same (field, raw_id), we record them in duplicates
            and do NOT create a unique mapping for that raw id.
        """
        if id_fields is None:
            id_fields = ('instance_id', 'id', 'request_id', 'uid', 'example_id')

        cache_key = ('instance_lookup', tuple(id_fields))
        if which == 'a':
            store = self._a_cache
            get_pis = self.per_instance_stats_a
            side_name = self.a_name
        elif which == 'b':
            store = self._b_cache
            get_pis = self.per_instance_stats_b
            side_name = self.b_name
        else:
            raise KeyError(which)

        if cache_key in store:
            return store[cache_key]

        per_inst = get_pis()

        inst_row_map = {}
        rawid_to_instkey = {}
        dup = {
            'side': side_name,
            'n_rows': 0,
            'dup_inst_key': {},     # inst_key -> [rows...]
            'dup_raw_id': {},       # (field, raw_id) -> [inst_keys...]
        }

        for row in per_inst:
            dup['n_rows'] += 1
            ik = self._instance_key(row)

            # inst_key duplicates
            if ik in inst_row_map:
                # record both the existing and new rows
                dup['dup_inst_key'].setdefault(ik, [inst_row_map[ik]]).append(row)
            else:
                inst_row_map[ik] = row

            # raw id duplicates
            if isinstance(row, dict):
                for f in id_fields:
                    if f in row:
                        rid = row[f]
                        rawk = (f, rid)
                        rawid_to_instkey.setdefault(rawk, []).append(ik)

        # finalize rawid_to_instkey to only unique ones; record duplicates
        rawid_unique = {}
        for rawk, iks in rawid_to_instkey.items():
            if len(iks) == 1:
                rawid_unique[rawk] = iks[0]
            else:
                dup['dup_raw_id'][rawk] = iks

        out = {
            'inst_row_map': inst_row_map,
            'rawid_to_instkey': rawid_unique,
            'duplicates': dup,
        }
        store[cache_key] = out
        return out

    def lookup_instance(self, instance_id, which='a', *, id_fields=None):
        """
        O(1) lookup of per_instance_stats row by raw id.

        Returns:
          row or None

        If duplicates exist for that id, returns None and you should consult
        `instance_lookup_warnings()` for details.
        """
        look = self._build_instance_lookup(which, id_fields=id_fields)
        inst_row_map = look['inst_row_map']
        rawid_to_instkey = look['rawid_to_instkey']

        if id_fields is None:
            id_fields = ('instance_id', 'id', 'request_id', 'uid', 'example_id')

        # try each possible field-name
        for f in id_fields:
            rawk = (f, instance_id)
            ik = rawid_to_instkey.get(rawk, None)
            if ik is not None:
                return inst_row_map.get(ik, None)

        # also allow passing already-normalized inst_key tuples
        if isinstance(instance_id, tuple) and len(instance_id) == 2:
            # could be inst_key already
            return inst_row_map.get(instance_id, None)

        return None

    def instance_lookup_warnings(self, which='a', *, id_fields=None, max_show=8) -> str:
        """
        Return a warning summary about duplicates in per-instance stats.
        """
        look = self._build_instance_lookup(which, id_fields=id_fields)
        dup = look['duplicates']
        lines = []
        side = dup['side']
        lines.append(f"[InstanceLookup] side={side} n_rows={dup['n_rows']}")
        n_dup_key = len(dup['dup_inst_key'])
        n_dup_raw = len(dup['dup_raw_id'])
        if n_dup_key == 0 and n_dup_raw == 0:
            lines.append("  no duplicates detected")
            return "\n".join(lines)

        if n_dup_key:
            lines.append(f"  duplicate inst_key count: {n_dup_key}")
            for ik, rows in list(dup['dup_inst_key'].items())[:max_show]:
                lines.append(f"    - inst_key={ik} n_rows={len(rows)}")
        if n_dup_raw:
            lines.append(f"  duplicate raw-id count: {n_dup_raw}")
            for rawk, iks in list(dup['dup_raw_id'].items())[:max_show]:
                lines.append(f"    - raw_id={rawk} n_inst_keys={len(iks)} sample={iks[:3]}")
        if n_dup_key > max_show or n_dup_raw > max_show:
            lines.append(f"  ... truncated to max_show={max_show}")
        return "\n".join(lines)
