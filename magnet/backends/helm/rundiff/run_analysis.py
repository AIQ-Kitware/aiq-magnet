"""
Single-run analysis utilities.

Focus: turning HELM stats into stable indices + useful metadata.
Comparison logic lives in compare.py.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable

import ubelt as ub

# Families you generally don't want to treat as "score correctness"
OPS_FAMILIES = {"ops", "finish", "num", "prompt"}

def deep_sort_keys(obj):
    """
    Recursively sort dict keys to make hashing stable.
    Leaves list order intact (usually desirable).
    """
    if isinstance(obj, dict):
        return {k: deep_sort_keys(obj[k]) for k in sorted(obj.keys())}
    elif isinstance(obj, (list, tuple)):
        return [deep_sort_keys(v) for v in obj]
    else:
        return obj

def metric_family(name: str) -> str:
    # hierarchical families
    if name.startswith("air_"):
        return "air"
    if name.startswith("bias_metric:"):
        return "bias_metric"
    if name.startswith("safety_"):
        return "safety"
    if name.startswith("bbq_"):
        return "bbq"

    # operational metrics you might want to ignore for score agreement
    if name in {
        "num_prompt_tokens",
        "num_completion_tokens",
        "num_output_tokens",
        "inference_runtime",
        "training_co2_cost",
        "training_energy_cost",
        "batch_size",
        "num_bytes",
        "num_perplexity_tokens",
        "max_prob",
        "logprob",
        "perplexity",
        "bits_per_byte",
        "logprob_per_byte",
    }:
        return "ops"

    if name.startswith("num_"):
        return "num"
    if name.startswith("finish_reason_"):
        return "finish"
    if name.startswith("prompt_"):
        return "prompt"

    if "@" in name:
        return name.split("@", 1)[0]
    m = re.match(r"^[a-z]+", name)
    return m.group(0) if m else name

def stable_name_key(name_obj, *, ignore_name_file_path=True) -> str:
    """
    Stable hash of stat["name"] dict (after sorting keys).
    Optionally strip name_file_path because it can differ across envs.
    """
    name_obj = ub.udict(name_obj).copy()
    if ignore_name_file_path:
        p = name_obj.get("perturbation", None)
        if isinstance(p, dict) and "name_file_path" in p:
            p = ub.udict(p).copy()
            p.pop("name_file_path", None)
            name_obj["perturbation"] = p
    name_obj = deep_sort_keys(name_obj)
    return ub.hash_data(name_obj, base=36)

def stat_meta(stat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata needed for grouping diffs.
    """
    name = stat["name"]
    metric = name["name"]
    split = name.get("split", None)
    perturb = name.get("perturbation", None)
    pert_name = perturb.get("name", None) if perturb else None
    is_pert = perturb is not None
    fam = metric_family(metric)
    kind = "ops" if fam in OPS_FAMILIES else "task"
    key = stable_name_key(name)
    mean = stat.get("mean", None)
    count = stat.get("count", 0)
    return {
        "key": key,
        "metric": metric,
        "family": fam,
        "kind": kind,
        "split": split,
        "is_pert": is_pert,
        "pert_name": pert_name,
        "count": int(count),
        "mean": None if mean is None else float(mean),
        "name_obj": name,
    }

def index_stats(stats_list: Iterable[Dict[str, Any]], *, drop_zero_count=True) -> Dict[str, Dict[str, Any]]:
    """
    key -> meta dict
    """
    idx = {}
    for s in stats_list:
        if drop_zero_count and s.get("count", 0) == 0:
            continue
        m = stat_meta(s)
        idx[m["key"]] = m
    return idx

def build_bucket_index(
    stats_list: Iterable[Dict[str, Any]],
    *,
    drop_zero_count=True,
    require_mean=True,
):
    """
    Build an index:
      idx[bucket][split][key] = (mean, family)

    where bucket is one of:
      base_task, base_ops, pert_task, pert_ops
    """
    idx = ub.ddict(lambda: ub.ddict(dict))
    for s in stats_list:
        if drop_zero_count and s.get("count", 0) == 0:
            continue
        m = stat_meta(s)
        if require_mean and m["mean"] is None:
            continue
        bucket = ("pert" if m["is_pert"] else "base") + "_" + m["kind"]
        idx[bucket][m["split"]][m["key"]] = (m["mean"], m["family"])
    return idx

def summarize_stats_inventory(stats_list: Iterable[Dict[str, Any]]):
    """
    Lightweight histogram summary over a stat list (for exploration).
    """
    hist = {
        "counts": Counter(),
        "perturbed": Counter(),
        "splits": Counter(),
        "metric_family": Counter(),
    }
    for s in stats_list:
        hist["counts"][s.get("count", 0)] += 1
        if s.get("count", 0) == 0:
            continue
        name = s["name"]
        hist["splits"][name.get("split", None)] += 1
        is_pert = "perturbation" in name
        hist["perturbed"][is_pert] += 1
        fam = metric_family(name["name"])
        hist["metric_family"][fam] += 1
    return hist
