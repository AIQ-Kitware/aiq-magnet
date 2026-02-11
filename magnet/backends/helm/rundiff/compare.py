"""
Run-to-run comparison.

Produces compact, "Sankey-friendly" features from two runs, plus optional
drill-down artifacts (e.g. mismatch family counts).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
import typing
import ubelt as ub
from .run_analysis import build_bucket_index
from collections import Counter
from typing import Iterable


if typing.TYPE_CHECKING:
    from magnet.helm_outputs import HelmRun

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

BOOKKEEPING_PREFIXES = (
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


def _fmt_float(x, nd=3):
    if x is None:
        return 'None'
    try:
        if isinstance(x, (int,)):
            return str(x)
        if math.isnan(x):
            return 'nan'
    except Exception:
        ...
    return f'{float(x):.{nd}g}'


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return float('inf') if a != 0 else 0.0
    return a / b


def _topk(counter: Counter, k: int) -> list[tuple[Any, int]]:
    return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:k]


def _get_any(d: Dict[str, Any], keys: Iterable[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _stat_label_from_key(key: Any) -> str:
    """
    Try to make a readable label out of a metric-key / name object.
    Works for:
      - dict keys like {'name': 'exact_match', 'split': 'test', ...}
      - tuples
      - strings
    """
    if isinstance(key, str):
        return key
    if isinstance(key, dict):
        name = key.get('name', key.get('metric', None))
        split = key.get('split', None)
        if split is not None and name is not None:
            return f'{name} split={split}'
        if name is not None:
            return str(name)
        return ub.urepr(key, compact=1, nl=0, nobr=1)
    if isinstance(key, tuple):
        return ' | '.join(map(str, key))
    return str(key)


def _delta_rows_from_core(core: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize "top mismatch rows" from whatever compare_core_stats produced.
    Expected to find one of:
      core['top'] / core['top_rows'] / core['core_top_mismatches']
    Each row should ideally have: key, mean_a, mean_b, abs, rel
    """
    rows = _get_any(
        core, ['core_top_mismatches', 'top_rows', 'top', 'rows'], default=[]
    )
    if rows is None:
        rows = []
    normed = []
    for r in rows:
        if isinstance(r, dict):
            normed.append(r)
        else:
            # sometimes it might be a tuple like (key, mean_a, mean_b, abs, rel)
            try:
                key, mean_a, mean_b, absd, reld = r
                normed.append(
                    {
                        'key': key,
                        'mean_a': mean_a,
                        'mean_b': mean_b,
                        'abs': absd,
                        'rel': reld,
                    }
                )
            except Exception:
                normed.append({'key': r})
    return normed


def format_core_report(
    core: Dict[str, Any], *, title: str = 'CORE METRIC DIFF', topn: int = 8
) -> str:
    """
    Pretty print core metric comparison dict.

    The dict is expected (but not required) to have:
      nA, nB, isect, union, core_coverage, core_agreement, core_mismatches
      and optionally: core_top_mismatches (list of rows)
    """
    nA = _get_any(core, ['nA', 'core_nA'], None)
    nB = _get_any(core, ['nB', 'core_nB'], None)
    isect = _get_any(core, ['isect', 'core_isect'], None)
    union = _get_any(core, ['union', 'core_union'], None)

    cov = _get_any(core, ['core_coverage', 'coverage'], None)
    agree = _get_any(core, ['core_agreement', 'agreement'], None)
    mism = _get_any(core, ['core_mismatches', 'mismatches'], None)

    lines = []
    lines.append('=' * 80)
    lines.append(title)
    if any(v is not None for v in [nA, nB, isect, union]):
        lines.append(f'nA={nA} nB={nB}  isect={isect} union={union}')
    lines.append(
        f'coverage={_fmt_float(cov, 4)}  agreement(isclose)={_fmt_float(agree, 4)}  mismatches={mism}'
    )

    rows = _delta_rows_from_core(core)
    if rows:
        # sort by abs if available
        def sort_key(r):
            return -(
                r.get('abs', float('-inf'))
                if r.get('abs', None) is not None
                else float('-inf')
            )

        rows_sorted = sorted(rows, key=sort_key)[:topn]
        lines.append('')
        lines.append(f'Top {min(topn, len(rows_sorted))} mean deltas:')
        for r in rows_sorted:
            key = r.get('key', r.get('name', None))
            mean_a = _get_any(r, ['mean_a', 'A', 'a', 'value_a', 'valueA'], None)
            mean_b = _get_any(r, ['mean_b', 'B', 'b', 'value_b', 'valueB'], None)
            absd = _get_any(r, ['abs', 'abs_delta', 'absd'], None)
            reld = _get_any(r, ['rel', 'rel_delta', 'reld'], None)

            # if abs/rel missing, compute if possible
            try:
                if absd is None and mean_a is not None and mean_b is not None:
                    absd = abs(float(mean_a) - float(mean_b))
                if reld is None and mean_a is not None and mean_b is not None:
                    reld = _safe_div(
                        abs(float(mean_a) - float(mean_b)), abs(float(mean_b)) + 1e-12
                    )
            except Exception:
                ...

            label = _stat_label_from_key(key)
            lines.append(
                f'  abs={_fmt_float(absd, 3)} rel={_fmt_float(reld, 3)} | {label} | A={_fmt_float(mean_a, 6)} B={_fmt_float(mean_b, 6)}'
            )
    lines.append('=' * 80)
    return '\n'.join(lines)


def _summarize_namekeys(namekeys: Iterable[Any]) -> Dict[str, Counter]:
    """
    Given an iterable of 'name keys' (dict-ish), summarize counts by a few facets
    similar to your earlier printouts: is_pert, kind, split, family, pert_name.

    If keys are dicts that include fields like:
      name, split, kind, is_perturbed, perturbation_name, family
    we use them; otherwise we do best-effort inference.
    """
    summ = {
        'is_pert': Counter(),
        'kind': Counter(),
        'split': Counter(),
        'family': Counter(),
        'pert_name': Counter(),
        'metric': Counter(),
    }

    for k in namekeys:
        if isinstance(k, dict):
            metric = k.get('name', k.get('metric', None))
            split = k.get('split', None)
            kind = k.get('kind', None)
            fam = k.get('family', None)

            # pert presence
            is_pert = k.get('is_perturbed', None)
            pert = k.get('pert_name', k.get('perturbation_name', None))

            if is_pert is None:
                is_pert = pert is not None

            summ['is_pert'][bool(is_pert)] += 1
            if kind is not None:
                summ['kind'][kind] += 1
            if split is not None:
                summ['split'][split] += 1
            if fam is not None:
                summ['family'][fam] += 1
            if pert is not None:
                summ['pert_name'][pert] += 1
            if metric is not None:
                summ['metric'][metric] += 1
        else:
            # if it's a string, at least count it as "metric"
            summ['metric'][str(k)] += 1

    return summ


def _format_counter_block(title: str, counter: Counter, topk: int = 6) -> List[str]:
    items = _topk(counter, topk)
    return [f'  {title}: {items}']


def format_bucket_report(
    comp: Dict[str, Any], *, title: str = 'RUN DIFF', topn: int = 3
) -> str:
    """
    Pretty print the richer bucket comparison output.

    This is designed to support your earlier text output like:

      coverage(isect/union)=37/41=0.902
      onlyA=4 onlyB=0 value_mismatches=26

      [Coverage] Present only in A summary:
        is_pert: ...
        kind: ...
        split: ...
        family: ...
        pert_name: ...

      [Value] Mismatch breakdown:
        is_perturbed: ...
        top families: ...

      [Value] Top 3 absolute mean deltas:
        abs=... rel=... | fam=... kind=... split=... pert=... | metric=...

    It will do best-effort extraction depending on what comp contains.
    """
    lines = []
    lines.append('=' * 80)

    # If caller wants to prepend run name, do it outside (the script already prints run_spec_name)
    lines.append(title)

    # expected fields (best effort)
    cov_isect = _get_any(comp, ['isect', 'coverage_isect', 'n_isect'], None)
    cov_union = _get_any(comp, ['union', 'coverage_union', 'n_union'], None)
    onlyA = _get_any(comp, ['onlyA', 'n_onlyA'], None)
    onlyB = _get_any(comp, ['onlyB', 'n_onlyB'], None)
    value_mismatches = _get_any(
        comp, ['value_mismatches', 'n_value_mismatches', 'mismatches'], None
    )

    # coverage ratio
    cov_ratio = _get_any(comp, ['coverage', 'coverage_ratio'], None)
    if cov_ratio is None and cov_isect is not None and cov_union is not None:
        try:
            cov_ratio = float(cov_isect) / float(cov_union)
        except Exception:
            cov_ratio = None

    if cov_isect is not None and cov_union is not None:
        lines.append(
            f'coverage(isect/union) = {cov_isect}/{cov_union} = {_fmt_float(cov_ratio, 4)}'
        )
    else:
        lines.append(f'coverage = {_fmt_float(cov_ratio, 4)}')

    # mismatch counts
    if any(v is not None for v in [onlyA, onlyB, value_mismatches]):
        lines.append(
            f'onlyA={onlyA}  onlyB={onlyB}  value_mismatches={value_mismatches}'
        )

    # ---- Coverage-only summaries (present only in A / only in B) ----
    onlyA_keys = _get_any(
        comp,
        ['onlyA_keys', 'onlyA_namekeys', 'unique_a', 'uniqueA', 'onlyA'],
        default=None,
    )
    onlyB_keys = _get_any(
        comp,
        ['onlyB_keys', 'onlyB_namekeys', 'unique_b', 'uniqueB', 'onlyB'],
        default=None,
    )

    # Some compare implementations store sets/lists directly in onlyA/onlyB; if it's small-int in onlyA, ignore.
    if isinstance(onlyA_keys, (int, float)):
        onlyA_keys = None
    if isinstance(onlyB_keys, (int, float)):
        onlyB_keys = None

    if onlyA_keys:
        lines.append('')
        lines.append('[Coverage] Present only in A summary:')
        summ = _summarize_namekeys(onlyA_keys)
        lines.extend(_format_counter_block('is_pert', summ['is_pert']))
        lines.extend(_format_counter_block('kind', summ['kind']))
        lines.extend(_format_counter_block('split', summ['split']))
        lines.extend(_format_counter_block('family', summ['family']))
        lines.extend(_format_counter_block('pert_name', summ['pert_name']))

    if onlyB_keys:
        lines.append('')
        lines.append('[Coverage] Present only in B summary:')
        summ = _summarize_namekeys(onlyB_keys)
        lines.extend(_format_counter_block('is_pert', summ['is_pert']))
        lines.extend(_format_counter_block('kind', summ['kind']))
        lines.extend(_format_counter_block('split', summ['split']))
        lines.extend(_format_counter_block('family', summ['family']))
        lines.extend(_format_counter_block('pert_name', summ['pert_name']))

    # ---- Value mismatch summaries ----
    mism_rows = _get_any(
        comp,
        [
            'mismatch_rows',
            'value_mismatch_rows',
            'mismatches_rows',
            'value_mismatches_rows',
        ],
        default=None,
    )
    if mism_rows is None:
        mism_rows = _get_any(comp, ['top_mismatches', 'mismatch_list'], default=[])

    # If comp includes a “mismatch_summary” already, prefer it
    mismatch_summary = _get_any(
        comp, ['mismatch_summary', 'value_mismatch_summary'], default=None
    )

    if mismatch_summary is not None:
        lines.append('')
        lines.append('[Value] Mismatch breakdown:')
        # Expect dict-like: {is_perturbed: Counter, families: Counter}
        if isinstance(mismatch_summary, dict):
            if 'is_perturbed' in mismatch_summary:
                lines.append(
                    f'  is_perturbed: {dict(mismatch_summary["is_perturbed"])}'
                )
            if 'families' in mismatch_summary:
                fams = mismatch_summary['families']
                if isinstance(fams, Counter):
                    lines.append(f'  top families: {_topk(fams, 10)}')
                else:
                    lines.append(f'  top families: {fams}')
        else:
            lines.append(f'  {mismatch_summary}')
    elif mism_rows:
        # Build our own summary if rows have 'key' dicts
        is_pert = Counter()
        fams = Counter()
        for r in mism_rows:
            key = r.get('key', r.get('name_key', None)) if isinstance(r, dict) else None
            if isinstance(key, dict):
                is_pert[bool(key.get('is_perturbed', key.get('is_pert', False)))] += 1
                fam = key.get('family', None)
                if fam is not None:
                    fams[fam] += 1
        if is_pert or fams:
            lines.append('')
            lines.append('[Value] Mismatch breakdown:')
            if is_pert:
                lines.append(f'  is_perturbed: {dict(is_pert)}')
            if fams:
                lines.append(f'  top families: {_topk(fams, 10)}')

    # ---- Top absolute mean deltas ----
    top_rows = _get_any(
        comp,
        ['top_abs_deltas', 'top_mean_deltas', 'top_deltas', 'top_rows'],
        default=None,
    )
    if top_rows is None:
        # fall back: sort mism_rows by abs if present
        if isinstance(mism_rows, list) and mism_rows:

            def key_abs(r):
                if isinstance(r, dict):
                    return r.get('abs', r.get('abs_delta', -1))
                return -1

            top_rows = sorted(
                [r for r in mism_rows if isinstance(r, dict)], key=key_abs, reverse=True
            )
        else:
            top_rows = []

    if top_rows:
        lines.append('')
        lines.append(f'[Value] Top {min(topn, len(top_rows))} absolute mean deltas:')
        for r in list(top_rows)[:topn]:
            key = _get_any(r, ['key', 'name_key'], None)
            absd = _get_any(r, ['abs', 'abs_delta'], None)
            reld = _get_any(r, ['rel', 'rel_delta'], None)
            mean_a = _get_any(r, ['mean_a', 'A', 'a'], None)
            mean_b = _get_any(r, ['mean_b', 'B', 'b'], None)

            fam = kind = split = pert = metric = None
            if isinstance(key, dict):
                fam = key.get('family', None)
                kind = key.get('kind', None)
                split = key.get('split', None)
                pert = key.get('pert_name', key.get('perturbation_name', None))
                metric = key.get('name', None)

            pert_disp = pert if pert is not None else '-'

            # If abs/rel missing, compute
            try:
                if absd is None and mean_a is not None and mean_b is not None:
                    absd = abs(float(mean_a) - float(mean_b))
                if reld is None and mean_a is not None and mean_b is not None:
                    reld = _safe_div(
                        abs(float(mean_a) - float(mean_b)), abs(float(mean_b)) + 1e-12
                    )
            except Exception:
                ...

            left = f'abs={_fmt_float(absd, 4)} rel={_fmt_float(reld, 4)}'
            mid = f'fam={fam} kind={kind} split={split} pert={pert_disp}'
            right = f'metric={metric}'
            if mean_a is not None or mean_b is not None:
                right += f' | A={_fmt_float(mean_a, 6)} B={_fmt_float(mean_b, 6)}'
            lines.append(f'  {left} | {mid} | {right}')

    lines.append('=' * 80)
    return '\n'.join(lines)


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
        key = ('bucket_compare', float(rel_tol), float(abs_tol))

        def factory():
            idx_a, idx_b = self.bucket_indices()
            return compare_bucket_indices(
                idx_a, idx_b, rel_tol=rel_tol, abs_tol=abs_tol
            )

        return self._cached(key, factory)

    def summary_base_task(self, *, rel_tol=1e-4, abs_tol=1e-8) -> Dict[str, Any]:
        """
        Small scalar summary for tables / Sankey.

        Assumes your compare module has a summarizer that produces:
            base_task_coverage, base_task_agreement, agreement_bucket_base_task
        """
        comp = self.bucket_compare(rel_tol=rel_tol, abs_tol=abs_tol)
        feats = summarize_for_sankey(comp)
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

        comp = self.bucket_compare(rel_tol=rel_tol, abs_tol=abs_tol)
        return format_bucket_report(comp, topn=topn)

    # ---- “Core” metric comparisons ----
    def core_compare(self, *, rel_tol=1e-4, abs_tol=1e-8, topn=10):
        key = ('core_compare', float(rel_tol), float(abs_tol), int(topn))

        def factory():
            return compare_core_stats(
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
        core = self.core_compare(rel_tol=rel_tol, abs_tol=abs_tol, topn=topn)
        return format_core_report(core)

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

    # --------- small helpers ----------
    def _mark(self, ok: bool) -> str:
        return '✅' if ok else '❌'

    def _safe_get(self, d, key, default=None):
        try:
            return d.get(key, default)
        except Exception:
            return default

    def _get_run_spec(self, run):
        """
        Best effort: prefer run.json.run_spec() if available; fall back to other attrs.
        """
        if hasattr(run, 'json') and hasattr(run.json, 'run_spec'):
            try:
                return run.json.run_spec()
            except Exception:
                pass
        # fallback: sometimes run spec name is in metadata; caller may also store it elsewhere
        return getattr(run, 'run_spec', None)

    def _get_scenario_class(self, run):
        """
        Best effort: prefer scenario class from run_spec (if present), else from scenario_state.
        """
        rs = self._get_run_spec(run)
        if isinstance(rs, dict):
            # common keys you might see
            for k in ['scenario_class', 'scenario', 'scenarioClass', 'scenario_name']:
                if k in rs:
                    return rs[k]
        # fallback: scenario_state often includes it
        if hasattr(run, 'json') and hasattr(run.json, 'scenario_state'):
            try:
                st = run.json.scenario_state()
                if isinstance(st, dict):
                    for k in [
                        'scenario_class',
                        'scenario',
                        'scenarioClass',
                        'scenario_name',
                    ]:
                        if k in st:
                            return st[k]
            except Exception:
                pass
        return None

    def _namekey(self, stat, *, include_count: bool) -> str:
        """
        Defines a stable 'identity' for a stat for coverage (not values).
        By default: use stat['name'] (dict) only.
        If include_count: add stat['count'] as part of identity.

        Note: we intentionally DO NOT canonicalize ordering here; ub.hash_data handles dict order.
        """
        name_obj = stat.get('name', None)
        if include_count:
            key_obj = {'name': name_obj, 'count': stat.get('count', None)}
        else:
            key_obj = {'name': name_obj}
        return ub.hash_data(key_obj, base=36)

    def _coverage_counts(self, keys_a, keys_b):
        set_a = set(keys_a)
        set_b = set(keys_b)
        isect = set_a & set_b
        union = set_a | set_b
        return {
            'nA': len(set_a),
            'nB': len(set_b),
            'isect': len(isect),
            'union': len(union),
            'onlyA': len(set_a - set_b),
            'onlyB': len(set_b - set_a),
        }

    def _extract_instance_ids(self, per_instance_stats):
        """
        Best effort: pick a stable-ish id key.
        Many HELM per-instance rows include 'instance_id'. If not, fallback to hashing row.
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
                    ids.append(('hash', ub.hash_data(row, base=36)))
            else:
                ids.append(('hash', ub.hash_data(row, base=36)))
        return ids

    # --------- the requested first-level summary ----------
    def summary_level1(self, *, max_show: int = 160) -> str:
        """
        Level-1 summary (coverage / identity only; no value comparison):

        1) Run spec equality check
        2) Scenario equality check
        3) Stats coverage by name (ignoring count/value)
        4) Stats coverage by (name + count) (still ignoring value)
        5) Instance coverage (using per_instance_stats overlap)
        """
        lines = []
        lines.append('=' * 80)
        lines.append('RUNDIFF SUMMARY (L1)')
        lines.append('')

        # 0) run spec name check (fast + readable)
        run_spec_a = self.run_a.json.run_spec()
        run_spec_b = self.run_b.json.run_spec()
        run_spec_name_a = run_spec_a.get('name', None)
        run_spec_name_b = run_spec_b.get('name', None)

        same_name = (run_spec_name_a == run_spec_name_b) and (
            run_spec_name_a is not None
        )
        if same_name:
            lines.append(f'0) Run spec name: {self._mark(True)} {run_spec_name_a}')
        else:
            lines.append(f'0) Run spec name: {self._mark(False)}')
            lines.append(f'   A: {run_spec_name_a}')
            lines.append(f'   B: {run_spec_name_b}')
        lines.append('')

        # 1) run spec structural check (print diffs, not full blobs)
        # Choose a consistent diff direction: diff = B.diff(A)
        # Then:
        #   - faillist entries show B vs A values
        #   - unique1 are only in B
        #   - unique2 are only in A
        spec_diff_raw = ub.IndexableWalker(run_spec_b).diff(run_spec_a)
        spec_diff = ub.udict(spec_diff_raw) - {'passlist'}

        same_spec = (
            len(spec_diff.get('faillist', [])) == 0
            and len(spec_diff.get('unique1', [])) == 0
            and len(spec_diff.get('unique2', [])) == 0
        )

        lines.append(f'1) Run spec (full): {self._mark(same_spec)}')
        if not same_spec:
            lines.extend(
                _format_diff(spec_diff, label_a='A', label_b='B', max_items=12)
            )
        lines.append('')

        # 2) scenario check (again: diff-first)
        scen_a = self.run_a.json.scenario()
        scen_b = self.run_b.json.scenario()

        if scen_a is None and scen_b is None:
            lines.append('2) Scenario: ⚠️ (unknown)')
            lines.append('')
        else:
            scen_diff_raw = ub.IndexableWalker(scen_b).diff(scen_a)
            scen_diff = ub.udict(scen_diff_raw) - {'passlist'}

            same_scen = (
                len(scen_diff.get('faillist', [])) == 0
                and len(scen_diff.get('unique1', [])) == 0
                and len(scen_diff.get('unique2', [])) == 0
            )

            lines.append(f'2) Scenario: {self._mark(same_scen)}')
            if not same_scen:
                lines.extend(
                    _format_diff(scen_diff, label_a='A', label_b='B', max_items=12)
                )
            else:
                # short one-liner when equal
                lines.append(f'   {ub.urepr(scen_a, compact=1, nl=0, nobr=1)}')
            lines.append('')

        # 3) stats coverage (names only)
        stats_a = self.stats_a()
        stats_b = self.stats_b()

        # Build *displayable* ids alongside hashed ids so we can preview differences.
        # Identity here: just the 'name' object (dict) -> hash
        def stat_name_obj(stat):
            return stat.get('name', None)

        def stat_name_disp(name_obj):
            # compact display: metric + split + maybe a hint if perturbation present
            if isinstance(name_obj, dict):
                metric = name_obj.get('name', None)
                split = name_obj.get('split', None)
                pert = name_obj.get('perturbation', None)
                if pert:
                    pert_name = pert.get('name', 'pert')
                    return f'{metric} split={split} pert={pert_name}'
                else:
                    return f'{metric} split={split}'
            return ub.urepr(name_obj, compact=1, nl=0, nobr=1)

        def name_hash(name_obj):
            return ub.hash_data({'name': name_obj}, base=36)

        # maps: hash -> display string (keep first)
        disp_a = {}
        disp_b = {}
        keys_a = []
        keys_b = []
        for s in stats_a:
            if isinstance(s, dict):
                h = name_hash(stat_name_obj(s))
                keys_a.append(h)
                disp_a.setdefault(h, stat_name_disp(stat_name_obj(s)))
        for s in stats_b:
            if isinstance(s, dict):
                h = name_hash(stat_name_obj(s))
                keys_b.append(h)
                disp_b.setdefault(h, stat_name_disp(stat_name_obj(s)))

        cov = self._coverage_counts(keys_a, keys_b)
        lines.append('3) Stats coverage by name (ignoring count/value):')
        lines.append(
            f'   nA={cov["nA"]} nB={cov["nB"]}  isect={cov["isect"]} union={cov["union"]}  onlyA={cov["onlyA"]} onlyB={cov["onlyB"]}'
        )

        # quick preview of what differs (top-k)
        if cov['onlyA'] or cov['onlyB']:
            set_a = set(keys_a)
            set_b = set(keys_b)
            onlyA = sorted(set_a - set_b)
            onlyB = sorted(set_b - set_a)
            k = 8
            if onlyA:
                lines.append(
                    f'   Only in A (showing {min(k, len(onlyA))}/{len(onlyA)}):'
                )
                for h in onlyA[:k]:
                    lines.append(f'     - {disp_a.get(h, h)}')
            if onlyB:
                lines.append(
                    f'   Only in B (showing {min(k, len(onlyB))}/{len(onlyB)}):'
                )
                for h in onlyB[:k]:
                    lines.append(f'     - {disp_b.get(h, h)}')
        lines.append('')

        # 4) stats coverage (name + count)
        def namecount_hash(stat):
            name_obj = stat.get('name', None)
            cnt = stat.get('count', None)
            return ub.hash_data({'name': name_obj, 'count': cnt}, base=36)

        disp2_a = {}
        disp2_b = {}
        keys_a2 = []
        keys_b2 = []
        for s in stats_a:
            if isinstance(s, dict):
                h = namecount_hash(s)
                keys_a2.append(h)
                # add count to display
                nobj = stat_name_obj(s)
                disp2_a.setdefault(
                    h, stat_name_disp(nobj) + f' count={s.get("count", None)}'
                )
        for s in stats_b:
            if isinstance(s, dict):
                h = namecount_hash(s)
                keys_b2.append(h)
                nobj = stat_name_obj(s)
                disp2_b.setdefault(
                    h, stat_name_disp(nobj) + f' count={s.get("count", None)}'
                )

        cov2 = self._coverage_counts(keys_a2, keys_b2)
        lines.append('4) Stats coverage by (name + count) (ignoring value):')
        lines.append(
            f'   nA={cov2["nA"]} nB={cov2["nB"]}  isect={cov2["isect"]} union={cov2["union"]}  onlyA={cov2["onlyA"]} onlyB={cov2["onlyB"]}'
        )

        # This section is most useful if we ALSO show “same name but different count”.
        # Compute per-name count multiset diffs.
        def name_only_key(stat):
            return ub.hash_data({'name': stat.get('name', None)}, base=36)

        counts_by_name_a = {}
        counts_by_name_b = {}
        for s in stats_a:
            if isinstance(s, dict):
                k = name_only_key(s)
                counts_by_name_a.setdefault(k, Counter())[s.get('count', None)] += 1
        for s in stats_b:
            if isinstance(s, dict):
                k = name_only_key(s)
                counts_by_name_b.setdefault(k, Counter())[s.get('count', None)] += 1

        count_mismatch_names = []
        for k in set(counts_by_name_a) & set(counts_by_name_b):
            if counts_by_name_a[k] != counts_by_name_b[k]:
                count_mismatch_names.append(k)

        if cov2['onlyA'] or cov2['onlyB']:
            set_a2 = set(keys_a2)
            set_b2 = set(keys_b2)
            onlyA2 = sorted(set_a2 - set_b2)
            onlyB2 = sorted(set_b2 - set_a2)
            k = 8
            if onlyA2:
                lines.append(
                    f'   Only in A (showing {min(k, len(onlyA2))}/{len(onlyA2)}):'
                )
                for h in onlyA2[:k]:
                    lines.append(f'     - {disp2_a.get(h, h)}')
            if onlyB2:
                lines.append(
                    f'   Only in B (showing {min(k, len(onlyB2))}/{len(onlyB2)}):'
                )
                for h in onlyB2[:k]:
                    lines.append(f'     - {disp2_b.get(h, h)}')

        if count_mismatch_names:
            k = 8
            lines.append(
                f'   Same metric-name but different count-distribution (showing {min(k, len(count_mismatch_names))}/{len(count_mismatch_names)}):'
            )
            for nk in count_mismatch_names[:k]:
                # try to recover a representative display from either side
                # (we can’t map name-only hash back reliably, but we can approximate by searching disp_a keys)
                # Better: build a reverse map once; here keep it cheap.
                label = None
                for h, d in disp_a.items():
                    if h == nk:
                        label = d
                        break
                if label is None:
                    for h, d in disp_b.items():
                        if h == nk:
                            label = d
                            break
                if label is None:
                    label = nk
                lines.append(f'     - {label}')
                lines.append(f'       A counts: {dict(counts_by_name_a.get(nk, {}))}')
                lines.append(f'       B counts: {dict(counts_by_name_b.get(nk, {}))}')

        lines.append('')

        # 5) instance coverage
        try:
            pia = self.per_instance_stats_a()
            pib = self.per_instance_stats_b()
            ids_a = self._extract_instance_ids(pia)  # your tuple ids
            ids_b = self._extract_instance_ids(pib)

            covi = self._coverage_counts(ids_a, ids_b)
            lines.append('5) Instance coverage (per_instance_stats overlap):')
            lines.append(
                f'   nA={covi["nA"]} nB={covi["nB"]}  isect={covi["isect"]} union={covi["union"]}  onlyA={covi["onlyA"]} onlyB={covi["onlyB"]}'
            )

            # show what key type we ended up using (instance_id vs hash fallback)
            def key_type_hist(ids):
                return Counter(
                    [t[0] if isinstance(t, tuple) and len(t) else '?' for t in ids]
                )

            ktypes_a = key_type_hist(ids_a)
            ktypes_b = key_type_hist(ids_b)
            if ktypes_a != ktypes_b:
                lines.append(
                    f'   key-types differ: A={dict(ktypes_a)}  B={dict(ktypes_b)}'
                )

        except Exception as ex:
            lines.append('5) Instance coverage: ⚠️ (could not compute)')
            lines.append(f'   reason: {type(ex).__name__}: {ex}')

        def _stat_disp(name_obj: dict) -> str:
            metric = name_obj.get('name', None) if isinstance(name_obj, dict) else None
            split = name_obj.get('split', None) if isinstance(name_obj, dict) else None
            pn = _pert_name(name_obj)
            if pn:
                return f'{metric} split={split} pert={pn}'
            return f'{metric} split={split}'

        def _fmt_float(x, nd=4):
            try:
                return f'{float(x):.{nd}g}'
            except Exception:
                return str(x)

        def _classify_metric(metric_name: str):
            """
            Returns: ('core'|'bookkeeping'|'untracked', matched_prefix)
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

        def _top(counter, k=8):
            return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

        def _summarize_bucket(bucket_label, bucket, *, topk=6):
            """
            bucket is a dict with:
              n, mism, rows, prefix_all, prefix_mism, metric_all, metric_mism
            """
            lines = []
            n = bucket['n']
            mism = bucket['mism']
            agree = 1 - mism / max(n, 1)
            lines.append(
                f'     {bucket_label}: comparable={n}  mismatched={mism}  agreement={_fmt_float(agree, 4)}'
            )

            if n == 0:
                return lines

            # show what exists vs what mismatches
            lines.append(f'       top prefixes(all): {_top(bucket["prefix_all"], 6)}')
            if mism:
                lines.append(
                    f'       top prefixes(mism): {_top(bucket["prefix_mism"], 6)}'
                )

            # show top untracked metric names if this is untracked
            if bucket_label == 'untracked':
                # metric_all/mism store full metric names
                lines.append(
                    f'       top metric names(all): {_top(bucket["metric_all"], 6)}'
                )
                if mism:
                    lines.append(
                        f'       top metric names(mism): {_top(bucket["metric_mism"], 6)}'
                    )

            if mism:
                rows_sorted = sorted(
                    bucket['rows'],
                    key=lambda r: (float('-inf') if r['abs'] is None else -r['abs']),
                )
                lines.append(
                    f'       Top {min(topk, len(rows_sorted))} abs(mean) deltas:'
                )
                for r in rows_sorted[:topk]:
                    pert_tag = r['pert_name'] if r['pert_name'] else '-'
                    lines.append(
                        '         - '
                        f'abs={ub.urepr(r["abs"], nl=0)} rel={ub.urepr(r["rel"], nl=0)} | '
                        f'class={r["metric_class"]} pref={r["matched_prefix"]} | '
                        f'split={r["split"]} pert={pert_tag} | '
                        f'{_stat_disp(r["name"])} | '
                        f'A={ub.urepr(r["mean_a"], nl=0)} B={ub.urepr(r["mean_b"], nl=0)}'
                    )
            return lines

        # 6) value-aware summary for intersecting stats (compare mean only), split by perturbed and metric class
        rel_tol = 1e-4
        abs_tol = 1e-8
        FIELDS = ['mean']  # keep it mean-only for now

        def name_hash(name_obj):
            return ub.hash_data({'name': name_obj}, base=36)

        stats_a = self.stats_a()
        stats_b = self.stats_b()

        lut_a = {}
        lut_b = {}
        for s in stats_a:
            if isinstance(s, dict) and 'name' in s:
                lut_a.setdefault(name_hash(s['name']), s)
        for s in stats_b:
            if isinstance(s, dict) and 'name' in s:
                lut_b.setdefault(name_hash(s['name']), s)

        isect = sorted(set(lut_a) & set(lut_b))

        # Structure:
        # groups[pert_group][metric_class] -> bucket stats
        def _new_bucket():
            return {
                'n': 0,
                'mism': 0,
                'rows': [],
                'prefix_all': Counter(),
                'prefix_mism': Counter(),
                'metric_all': Counter(),
                'metric_mism': Counter(),
            }

        groups = {
            'unperturbed': {
                'core': _new_bucket(),
                'bookkeeping': _new_bucket(),
                'untracked': _new_bucket(),
            },
            'perturbed': {
                'core': _new_bucket(),
                'bookkeeping': _new_bucket(),
                'untracked': _new_bucket(),
            },
        }

        for k in isect:
            sa = lut_a[k]
            sb = lut_b[k]
            name_obj = sa.get('name', None)

            if isinstance(name_obj, dict):
                is_pert = bool(name_obj.get('perturbation', None))
                metric = name_obj.get('name', None)
                split = name_obj.get('split', None)
                pn = _pert_name(name_obj)
            else:
                is_pert = False
                metric = None
                split = None
                pn = None

            metric_class, matched_prefix = _classify_metric(metric)
            pert_group = 'perturbed' if is_pert else 'unperturbed'
            bucket = groups[pert_group][metric_class]

            bucket['n'] += 1
            if matched_prefix is not None:
                bucket['prefix_all'][matched_prefix] += 1
            else:
                # For untracked, also keep a simple prefix hint = first token before '_' or ':'
                if metric:
                    hint = metric.split('_', 1)[0].split(':', 1)[0]
                    bucket['prefix_all'][hint] += 1
                else:
                    bucket['prefix_all']['?'] += 1

            if metric:
                bucket['metric_all'][metric] += 1

            # Compare selected fields
            field_mismatch = False
            for f in FIELDS:
                va = sa.get(f, None)
                vb = sb.get(f, None)
                if va is None and vb is None:
                    continue
                if not _isclose(va, vb, rel_tol=rel_tol, abs_tol=abs_tol):
                    field_mismatch = True
                    break

            if field_mismatch:
                bucket['mism'] += 1
                if matched_prefix is not None:
                    bucket['prefix_mism'][matched_prefix] += 1
                else:
                    if metric:
                        hint = metric.split('_', 1)[0].split(':', 1)[0]
                        bucket['prefix_mism'][hint] += 1
                    else:
                        bucket['prefix_mism']['?'] += 1

                if metric:
                    bucket['metric_mism'][metric] += 1

                va = sa.get('mean', None)
                vb = sb.get('mean', None)
                try:
                    absd = abs(float(va) - float(vb))
                    reld = absd / (abs(float(vb)) + 1e-12)
                except Exception:
                    absd = None
                    reld = None

                bucket['rows'].append(
                    {
                        'metric_class': metric_class,
                        'matched_prefix': matched_prefix,
                        'split': split,
                        'pert_name': pn,
                        'metric': metric,
                        'name': name_obj,
                        'mean_a': va,
                        'mean_b': vb,
                        'abs': absd,
                        'rel': reld,
                    }
                )

        # Emit summary
        lines.append('6) Value check on intersecting stats (compare mean only):')
        lines.append('   split by perturbation × (core / bookkeeping / untracked)')
        for pg in ['unperturbed', 'perturbed']:
            lines.append(f'   {pg}:')
            lines.extend(_summarize_bucket('core', groups[pg]['core'], topk=6))
            lines.extend(
                _summarize_bucket('bookkeeping', groups[pg]['bookkeeping'], topk=6)
            )
            lines.extend(
                _summarize_bucket('untracked', groups[pg]['untracked'], topk=6)
            )
        lines.append('')

        text = '\n'.join(lines)
        print(text)

        return text


def _pretty_path(path):
    # IndexableWalker paths are tuples; make them readable
    if isinstance(path, (list, tuple)):
        return '.'.join(map(str, path))
    return str(path)


def _format_diff(diff, *, label_a='A', label_b='B', max_items=12, indent='   '):
    """
    Format ub.IndexableWalker.diff output into a short, readable list.

    diff should be a dict-like with at least 'faillist' / 'unique1' / 'unique2'.
    We treat:
      - 'faillist' as mismatched values at shared paths
      - 'unique1' as present only in B (depending on diff direction)
      - 'unique2' as present only in A
    (Direction depends on how you call diff; we’ll label consistently below.)
    """
    lines = []
    if diff is None:
        return lines

    faillist = diff.get('faillist', []) or []
    unique_b = diff.get('unique1', []) or []  # see note below on direction
    unique_a = diff.get('unique2', []) or []

    # Mismatched values
    if faillist:
        lines.append(f'{indent}Value mismatches ({len(faillist)}):')
        for item in faillist[:max_items]:
            # item is often (path, v1, v2) or dict-ish; handle both
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

    # Coverage-only differences (paths only; values are sometimes in the item)
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


def _isclose(a, b, rel_tol=1e-4, abs_tol=1e-8):
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    except Exception:
        return a == b


def _stat_family(metric_name: str) -> str:
    # lightweight family heuristic; you already have a better one — feel free to swap
    if not metric_name:
        return '?'
    return metric_name.split('_', 1)[0].split(':', 1)[0]


def _stat_is_perturbed(name_obj: dict) -> bool:
    return bool(isinstance(name_obj, dict) and name_obj.get('perturbation', None))


def _stat_disp(name_obj: dict) -> str:
    # compact label for report lines
    metric = name_obj.get('name', None) if isinstance(name_obj, dict) else None
    split = name_obj.get('split', None) if isinstance(name_obj, dict) else None
    pert = None
    if isinstance(name_obj, dict) and name_obj.get('perturbation', None):
        pert = name_obj['perturbation'].get('name', 'pert')
    if pert:
        return f'{metric} split={split} pert={pert}'
    return f'{metric} split={split}'


def _pert_name(name_obj):
    if isinstance(name_obj, dict) and name_obj.get('perturbation', None):
        return name_obj['perturbation'].get('name', 'pert')
    return None
