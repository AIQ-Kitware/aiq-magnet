"""
Run-to-run comparison.

Produces compact, "Sankey-friendly" features from two runs, plus optional
drill-down artifacts (e.g. mismatch family counts).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import ubelt as ub

from .run_analysis import build_bucket_index

try:
    # for type checking; avoid hard import costs in some contexts
    from magnet.helm_outputs import HelmRun
except Exception:  # pragma: no cover
    HelmRun = Any  # type: ignore

# --- Core metric matching ---
CORE_PREFIXES = (
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


def is_core_metric_name(metric_name: str) -> bool:
    return any(metric_name.startswith(p) for p in CORE_PREFIXES)


def extract_core_stats(stats_list, *, require_unperturbed=True, require_count_gt0=True):
    out = {}
    for s in stats_list:
        if require_count_gt0 and s.get('count', 0) == 0:
            continue
        name = s['name']
        metric = name['name']
        if require_unperturbed and ('perturbation' in name):
            continue
        if not is_core_metric_name(metric):
            continue
        # Use a canonical key that ignores perturbation (already excluded here)
        key = ub.hash_data(ub.udict(name), base=36)
        out[key] = {
            'metric': metric,
            'split': name.get('split', None),
            'mean': None if s.get('mean', None) is None else float(s['mean']),
            'count': int(s.get('count', 0)),
        }
    return out


def compare_core_stats(helm_stats, kwdg_stats, *, rel_tol=1e-4, abs_tol=1e-8, topn=10):
    A = extract_core_stats(helm_stats)
    B = extract_core_stats(kwdg_stats)

    keysA = set(A)
    keysB = set(B)
    isect = keysA & keysB
    onlyA = keysA - keysB
    onlyB = keysB - keysA

    mism = []
    close = 0
    for k in isect:
        a = A[k]['mean']
        b = B[k]['mean']
        if a is None or b is None:
            continue
        ok = math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        close += int(ok)
        if not ok:
            absd = abs(a - b)
            reld = absd / (abs(b) + 1e-12)
            mism.append((absd, reld, k))
    mism.sort(reverse=True)

    cov = len(isect) / max(1, len(keysA | keysB))
    agree = close / max(1, len(isect))

    return {
        'core_nA': len(A),
        'core_nB': len(B),
        'core_isect': len(isect),
        'core_union': len(keysA | keysB),
        'core_coverage': cov,
        'core_agreement': agree,
        'core_mismatches': len(mism),
        'core_onlyA': len(onlyA),
        'core_onlyB': len(onlyB),
        'core_top_mismatches': [
            {
                'metric': A[k]['metric'],
                'split': A[k]['split'],
                'A_mean': A[k]['mean'],
                'B_mean': B[k]['mean'],
                'absdiff': absd,
                'reldiff': reld,
            }
            for absd, reld, k in mism[:topn]
        ],
    }


# --- Bucketed index comparison (base/pert x task/ops) ---


def compare_bucket_indices(A, B, *, rel_tol=1e-4, abs_tol=1e-8):
    summary = {}
    all_buckets = set(A.keys()) | set(B.keys())
    for bucket in sorted(all_buckets):
        summary[bucket] = {}
        splits = set(A.get(bucket, {}).keys()) | set(B.get(bucket, {}).keys())
        for split in sorted(splits):
            a = A.get(bucket, {}).get(split, {})
            b = B.get(bucket, {}).get(split, {})

            keys_a = set(a.keys())
            keys_b = set(b.keys())
            isect = keys_a & keys_b
            union = keys_a | keys_b

            n_isect = len(isect)
            n_union = len(union)
            cov = (n_isect / n_union) if n_union else 1.0

            n_close = 0
            fam_mismatch = ub.ddict(int)

            for k in isect:
                va, fa = a[k]
                vb, fb = b[k]
                if va is None or vb is None:
                    continue
                ok = math.isclose(va, vb, rel_tol=rel_tol, abs_tol=abs_tol)
                n_close += int(ok)
                if not ok:
                    fam = fa if fa is not None else fb
                    fam_mismatch[fam] += 1

            agree = (n_close / n_isect) if n_isect else 0.0

            summary[bucket][split] = {
                'coverage': cov,
                'agreement': agree,
                'n_isect': n_isect,
                'n_union': n_union,
                'n_close': n_close,
                'fam_mismatch_top': sorted(
                    fam_mismatch.items(), key=lambda x: x[1], reverse=True
                )[:5],
            }
    return summary


def _agg_weighted(comp_summary, bucket: str, key: str):
    tot = 0.0
    wsum = 0.0
    for split, rec in comp_summary.get(bucket, {}).items():
        w = rec.get('n_isect', 0)
        val = rec.get(key, None)
        if val is None:
            continue
        tot += val * w
        wsum += w
    return (tot / wsum) if wsum else None


def summarize_for_sankey(comp_summary):
    feats = {}
    for bucket in ['base_task', 'base_ops', 'pert_task', 'pert_ops']:
        feats[f'{bucket}_coverage'] = _agg_weighted(comp_summary, bucket, 'coverage')
        feats[f'{bucket}_agreement'] = _agg_weighted(comp_summary, bucket, 'agreement')

    bt_cov = feats['base_task_coverage'] or 0.0
    bt_ag = feats['base_task_agreement'] or 0.0

    if bt_cov < 0.98:
        label = 'base_task: coverage mismatch'
    elif bt_ag >= 0.99:
        label = 'base_task: near match'
    elif bt_ag >= 0.95:
        label = 'base_task: close'
    elif bt_ag >= 0.80:
        label = 'base_task: partial'
    else:
        label = 'base_task: low'

    feats['agreement_bucket_base_task'] = label
    return feats


def top_mismatch_families(comp_summary, bucket: str, topn: int = 5):
    fam_counts = ub.ddict(int)
    for split, rec in comp_summary.get(bucket, {}).items():
        for fam, n in rec.get('fam_mismatch_top', []):
            fam_counts[fam] += n
    return sorted(fam_counts.items(), key=lambda x: x[1], reverse=True)[:topn]


def compare_run_pair(
    helm_stats: List[Dict[str, Any]],
    kwdg_stats: List[Dict[str, Any]],
    *,
    rel_tol=1e-4,
    abs_tol=1e-8,
) -> Dict[str, Any]:
    A = build_bucket_index(helm_stats, require_mean=True, drop_zero_count=True)
    B = build_bucket_index(kwdg_stats, require_mean=True, drop_zero_count=True)

    comp = compare_bucket_indices(A, B, rel_tol=rel_tol, abs_tol=abs_tol)
    feats = summarize_for_sankey(comp)
    core = compare_core_stats(helm_stats, kwdg_stats, rel_tol=rel_tol, abs_tol=abs_tol)

    return {
        'base_task_cov': feats['base_task_coverage'],
        'base_task_agree': feats['base_task_agreement'],
        'agreement_bucket_base_task': feats['agreement_bucket_base_task'],
        'core_info': core,
        'base_task_mismatch_fams': top_mismatch_families(comp, 'base_task'),
    }


# --- helpers for Sankey plans ---
def attempt_status(row: Dict[str, Any]) -> str:
    return 'attempted' if row.get('reproduced_step1', False) else 'not_attempted'


def agreement_label(row: Dict[str, Any]) -> str:
    return row.get('agreement_bucket_base_task', 'unknown')


def index_per_instance_stats(
    per_instance_stats: List[Dict[str, Any]], *, key_fields=('instance_id',)
):
    """
    Build a stable lookup for per-instance stats.

    You may need to adjust `key_fields` depending on the scenario.
    """
    lut = {}
    for row in per_instance_stats:
        key = tuple(row.get(f, None) for f in key_fields)
        lut[key] = row
    return lut


def diff_per_instance_stats(
    A: List[Dict[str, Any]], B: List[Dict[str, Any]], *, key_fields=('instance_id',)
):
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
        'nA': len(keysA),
        'nB': len(keysB),
        'onlyA': sorted(onlyA),
        'onlyB': sorted(onlyB),
        'mismatched': mism,
    }


class RunDiff:
    """
    Lazy run-vs-run comparator with caching.

    - `_a_cache`: items read from run_a (stats, scenario_state, per_instance_stats, etc.)
    - `_b_cache`: items read from run_b
    - `_cache`: computed comparisons (bucket compare, core compare, reports, etc.)

    Notebook usage pattern:
        rd = RunDiff(run_a, run_b)
        row.update(rd.summary_base_task())
        row.update(rd.summary_core())

        # later
        print(rd.report_base_task())
        print(rd.report_core(topn=20))
        dd = rd.drilldown_scenario_state()
        pi = rd.drilldown_per_instance_stats(key_fields=("instance_id",))
    """

    def __init__(self, run_a: HelmRun, run_b: HelmRun):
        self.run_a = run_a
        self.run_b = run_b
        self._a_cache: Dict[str, Any] = {}
        self._b_cache: Dict[str, Any] = {}
        self._cache: Dict[Any, Any] = {}

    # -----------------------
    # Lazy reads from each run
    # -----------------------
    def _a(self, key: str, factory):
        if key not in self._a_cache:
            self._a_cache[key] = factory()
        return self._a_cache[key]

    def _b(self, key: str, factory):
        if key not in self._b_cache:
            self._b_cache[key] = factory()
        return self._b_cache[key]

    def stats_a(self) -> List[Dict[str, Any]]:
        return self._a('stats', lambda: self.run_a.json.stats())

    def stats_b(self) -> List[Dict[str, Any]]:
        return self._b('stats', lambda: self.run_b.json.stats())

    def scenario_state_a(self) -> Dict[str, Any]:
        return self._a('scenario_state', lambda: self.run_a.json.scenario_state())

    def scenario_state_b(self) -> Dict[str, Any]:
        return self._b('scenario_state', lambda: self.run_b.json.scenario_state())

    def per_instance_stats_a(self) -> List[Dict[str, Any]]:
        return self._a(
            'per_instance_stats', lambda: self.run_a.json.per_instance_stats()
        )

    def per_instance_stats_b(self) -> List[Dict[str, Any]]:
        return self._b(
            'per_instance_stats', lambda: self.run_b.json.per_instance_stats()
        )

    # -----------------------
    # Cached computed objects
    # -----------------------
    def _cached(self, key, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    # ---- Bucketed comparisons (your “base_task / ops / pert” machinery) ----
    def bucket_indices(self):
        """
        Returns preprocessed indices for bucketed comparisons.

        This should call into your existing canonicalization / indexing routines
        in `run_analysis.py` (or wherever you kept them).
        """
        from magnet.backends.helm.rundiff import run_analysis

        def factory():
            idx_a = run_analysis.build_bucket_index(
                self.stats_a(),
                drop_zero_count=True,
                require_mean=True,
            )
            idx_b = run_analysis.build_bucket_index(
                self.stats_b(),
                drop_zero_count=True,
                require_mean=True,
            )
            return idx_a, idx_b

        return self._cached('bucket_indices', factory)

    def bucket_compare(self, *, rel_tol=1e-4, abs_tol=1e-8):
        """
        Compare bucket indices and cache result.
        """
        from magnet.backends.helm.rundiff import compare as compare_mod

        key = ('bucket_compare', float(rel_tol), float(abs_tol))

        def factory():
            idx_a, idx_b = self.bucket_indices()
            return compare_mod.compare_bucket_indices(
                idx_a, idx_b, rel_tol=rel_tol, abs_tol=abs_tol
            )

        return self._cached(key, factory)

    def summary_base_task(self, *, rel_tol=1e-4, abs_tol=1e-8) -> Dict[str, Any]:
        """
        Small scalar summary for tables / Sankey.

        Assumes your compare module has a summarizer that produces:
            base_task_coverage, base_task_agreement, agreement_bucket_base_task
        """
        from magnet.backends.helm.rundiff import compare as compare_mod

        comp = self.bucket_compare(rel_tol=rel_tol, abs_tol=abs_tol)
        feats = compare_mod.summarize_for_sankey(comp)
        return {
            'base_task_cov': feats.get('base_task_coverage'),
            'base_task_agree': feats.get('base_task_agreement'),
            'agreement_bucket_base_task': feats.get('agreement_bucket_base_task'),
        }

    def report_base_task(self, *, rel_tol=1e-4, abs_tol=1e-8, topn=5) -> str:
        """
        Human-readable report (families, coverage deltas, top mean deltas).
        Keep this deterministic and compact for notebook scanning.
        """
        from magnet.backends.helm.rundiff import compare as compare_mod

        comp = self.bucket_compare(rel_tol=rel_tol, abs_tol=abs_tol)
        return compare_mod.format_bucket_report(comp, topn=topn)

    # ---- “Core” metric comparisons ----
    def core_compare(self, *, rel_tol=1e-4, abs_tol=1e-8, topn=10):
        from magnet.backends.helm.rundiff import compare as compare_mod

        key = ('core_compare', float(rel_tol), float(abs_tol), int(topn))

        def factory():
            return compare_mod.compare_core_stats(
                self.stats_a(),
                self.stats_b(),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                topn=topn,
            )

        return self._cached(key, factory)

    def summary_core(self, *, rel_tol=1e-4, abs_tol=1e-8) -> Dict[str, Any]:
        core = self.core_compare(rel_tol=rel_tol, abs_tol=abs_tol, topn=0)
        return {
            'core_cov': core.get('core_coverage'),
            'core_agree': core.get('core_agreement'),
            'core_mismatches': core.get('core_mismatches'),
            'core_onlyA': core.get('core_onlyA'),
            'core_onlyB': core.get('core_onlyB'),
        }

    def report_core(self, *, rel_tol=1e-4, abs_tol=1e-8, topn=10) -> str:
        from magnet.backends.helm.rundiff import compare as compare_mod

        core = self.core_compare(rel_tol=rel_tol, abs_tol=abs_tol, topn=topn)
        return compare_mod.format_core_report(core)

    # -----------------------
    # Drilldowns (expensive)
    # -----------------------
    def drilldown_scenario_state(self):
        """
        Raw structural diff of scenario_state.
        """
        key = 'scenario_state_diff'

        def factory():
            A = self.scenario_state_a()
            B = self.scenario_state_b()
            # ub.IndexableWalker expects indexable structures; scenario_state is dict-like
            return ub.IndexableWalker(A).diff(ub.IndexableWalker(B))

        return self._cached(key, factory)

    def drilldown_per_instance_stats(self, *, key_fields=('instance_id',)):
        """
        Diff per-instance stats keyed by one or more fields.
        """
        key = ('per_instance_stats_diff', tuple(key_fields))

        def factory():
            A = self.per_instance_stats_a()
            B = self.per_instance_stats_b()
            return diff_per_instance_stats(A, B, key_fields=key_fields)

        return self._cached(key, factory)

    # -----------------------
    # Convenience
    # -----------------------
    def clear_cache(self):
        """
        Clear computed comparisons (keeps per-run read caches by default).
        Useful if you change tolerances / definitions in code while iterating.
        """
        self._cache.clear()

    def clear_all(self):
        """
        Clear everything (including run-local read caches).
        """
        self._cache.clear()
        self._a_cache.clear()
        self._b_cache.clear()
