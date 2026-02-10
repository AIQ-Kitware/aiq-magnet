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

# --- Core metric matching ---
CORE_PREFIXES = (
    "exact_match",
    "quasi_exact_match",
    "prefix_exact_match",
    "quasi_prefix_exact_match",
    "classification_micro_f1",
    "classification_macro_f1",
    "f1_score",
    "rouge_l",
    "bleu_",
    "ifeval_strict_accuracy",
    "wildbench_score",
    "wildbench_score_rescaled",
    "omni_math_accuracy",
    "chain_of_thought_correctness",
    "math_equiv",
    "math_equiv_chain_of_thought",
    "safety_score",
    "safety_gpt_score",
    "safety_llama_score",
    "air_score",
    "air_category_",
)

def is_core_metric_name(metric_name: str) -> bool:
    return any(metric_name.startswith(p) for p in CORE_PREFIXES)

def extract_core_stats(stats_list, *, require_unperturbed=True, require_count_gt0=True):
    out = {}
    for s in stats_list:
        if require_count_gt0 and s.get("count", 0) == 0:
            continue
        name = s["name"]
        metric = name["name"]
        if require_unperturbed and ("perturbation" in name):
            continue
        if not is_core_metric_name(metric):
            continue
        # Use a canonical key that ignores perturbation (already excluded here)
        key = ub.hash_data(ub.udict(name), base=36)
        out[key] = {
            "metric": metric,
            "split": name.get("split", None),
            "mean": None if s.get("mean", None) is None else float(s["mean"]),
            "count": int(s.get("count", 0)),
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
        a = A[k]["mean"]
        b = B[k]["mean"]
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
        "core_nA": len(A),
        "core_nB": len(B),
        "core_isect": len(isect),
        "core_union": len(keysA | keysB),
        "core_coverage": cov,
        "core_agreement": agree,
        "core_mismatches": len(mism),
        "core_onlyA": len(onlyA),
        "core_onlyB": len(onlyB),
        "core_top_mismatches": [
            {
                "metric": A[k]["metric"],
                "split": A[k]["split"],
                "A_mean": A[k]["mean"],
                "B_mean": B[k]["mean"],
                "absdiff": absd,
                "reldiff": reld,
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
                "coverage": cov,
                "agreement": agree,
                "n_isect": n_isect,
                "n_union": n_union,
                "n_close": n_close,
                "fam_mismatch_top": sorted(
                    fam_mismatch.items(), key=lambda x: x[1], reverse=True
                )[:5],
            }
    return summary

def _agg_weighted(comp_summary, bucket: str, key: str):
    tot = 0.0
    wsum = 0.0
    for split, rec in comp_summary.get(bucket, {}).items():
        w = rec.get("n_isect", 0)
        val = rec.get(key, None)
        if val is None:
            continue
        tot += val * w
        wsum += w
    return (tot / wsum) if wsum else None

def summarize_for_sankey(comp_summary):
    feats = {}
    for bucket in ["base_task", "base_ops", "pert_task", "pert_ops"]:
        feats[f"{bucket}_coverage"] = _agg_weighted(comp_summary, bucket, "coverage")
        feats[f"{bucket}_agreement"] = _agg_weighted(comp_summary, bucket, "agreement")

    bt_cov = feats["base_task_coverage"] or 0.0
    bt_ag = feats["base_task_agreement"] or 0.0

    if bt_cov < 0.98:
        label = "base_task: coverage mismatch"
    elif bt_ag >= 0.99:
        label = "base_task: near match"
    elif bt_ag >= 0.95:
        label = "base_task: close"
    elif bt_ag >= 0.80:
        label = "base_task: partial"
    else:
        label = "base_task: low"

    feats["agreement_bucket_base_task"] = label
    return feats

def top_mismatch_families(comp_summary, bucket: str, topn: int = 5):
    fam_counts = ub.ddict(int)
    for split, rec in comp_summary.get(bucket, {}).items():
        for fam, n in rec.get("fam_mismatch_top", []):
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
        "base_task_cov": feats["base_task_coverage"],
        "base_task_agree": feats["base_task_agreement"],
        "agreement_bucket_base_task": feats["agreement_bucket_base_task"],
        "core_info": core,
        "base_task_mismatch_fams": top_mismatch_families(comp, "base_task"),
    }

# --- helpers for Sankey plans ---
def attempt_status(row: Dict[str, Any]) -> str:
    return "attempted" if row.get("reproduced_step1", False) else "not_attempted"

def agreement_label(row: Dict[str, Any]) -> str:
    return row.get("agreement_bucket_base_task", "unknown")
