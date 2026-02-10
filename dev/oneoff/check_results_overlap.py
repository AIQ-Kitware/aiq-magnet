"""
!uv pip install kaleido plotly
"""

from collections import Counter
import re
import math
import pandas as pd
import kwutil
import ubelt as ub
from magnet.helm_outputs import HelmOutputs, HelmRun
import networkx as nx
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union  # NOQA

"""
!python ~/code/aiq-magnet/dev/poc/inspect_historic_helm_runs.py /data/crfm-helm-public --out_fpath run_specs.yaml --out_detail_fpath run_details.yaml
"""
helm_rows = kwutil.Yaml.load('run_details.yaml')

finished_jobs = list(
    ub.Path('/home/local/KHQ/jon.crall/code/aiq-magnet/results/helm').glob('*/DONE')
)
kwdagger_rows = []
for fpath in finished_jobs:
    config = kwutil.Json.coerce(fpath.parent / 'job_config.json')
    run_spec_name = config['helm.run_entry']
    dpath = fpath.parent
    runs = HelmOutputs.coerce(dpath / 'benchmark_output').suites()[0].runs()
    assert len(runs) == 1
    run = runs[0]
    kwdagger_rows.append(
        {
            'dpath': dpath,
            'run_spec_name': run_spec_name,
            'run': run,
        }
    )

print(f'{len(helm_rows)=}')
print(f'{len(kwdagger_rows)=}')
# Writes:
# len(helm_rows)=427
# len(kwdagger_rows)=145

CORE_PREFIXES = (
    # common HELM headline metrics
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
    'math_equiv',
    'math_equiv_chain_of_thought',
    'safety_score',
    'safety_gpt_score',
    'safety_llama_score',
    'air_score',
    'air_category_',  # optional (can be a lot)
)

CORE_EXACT_EXCLUDE = {
    # e.g. "exact_match_indicator",
}


def stable_name_key(name_obj, *, ignore_perturbation=True, ignore_name_file_path=True):
    """
    Build a stable, comparable key for a HELM stat 'name' dict.

    By default we ignore perturbation (core metrics should be unperturbed anyway),
    but this makes the function safer. Also optionally drop name_file_path, which
    can differ across envs.
    """
    name_obj = ub.udict(name_obj).copy()
    if ignore_perturbation:
        name_obj.pop('perturbation', None)
    if ignore_name_file_path:
        p = name_obj.get('perturbation', None)
        if isinstance(p, dict):
            p = ub.udict(p).copy()
            p.pop('name_file_path', None)
            name_obj['perturbation'] = p
    name_obj = deep_sort_keys(name_obj)
    return ub.hash_data(name_obj, base=36)


def is_core_metric_name(metric_name: str) -> bool:
    if metric_name in CORE_EXACT_EXCLUDE:
        return False
    return any(metric_name.startswith(p) for p in CORE_PREFIXES)


def extract_core_stats(stats_list, *, require_unperturbed=True, require_count_gt0=True):
    """
    Returns dict: key -> record
    record includes: metric, split, mean, count, full_name_obj
    """
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

        key = stable_name_key(
            name, ignore_perturbation=True, ignore_name_file_path=True
        )
        out[key] = {
            'metric': metric,
            'split': name.get('split', None),
            'mean': None if s.get('mean', None) is None else float(s['mean']),
            'count': int(s.get('count', 0)),
            'name_obj': name,
            'stat_obj': s,
        }
    return out


def compare_core_stats(
    helm_stats, kwdg_stats, *, rel_tol=1e-4, abs_tol=1e-8, topn=30, verbose=0
):
    """
    Prints a concise diff for core metrics.
    Returns a dict you can store in helm_row if desired.
    """
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
        # if mean missing, skip (rare)
        if a is None or b is None:
            continue
        ok = math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        close += int(ok)
        if not ok:
            absd = abs(a - b)
            reld = absd / (abs(b) + 1e-12)
            mism.append((absd, reld, k))

    mism.sort(reverse=True)

    # Summaries
    cov = len(isect) / max(1, len(keysA | keysB))
    agree = close / max(1, len(isect))

    def _summ_missing(keys, src):
        c = Counter()
        for k in keys:
            r = src[k]
            c[(r['metric'], r['split'])] += 1
        return c

    missingA = _summ_missing(onlyA, A)
    missingB = _summ_missing(onlyB, B)

    if verbose:
        print('')
        print('=' * 80)
        print('CORE METRIC DIFF')
        print(f'nA={len(A)} nB={len(B)}  isect={len(isect)} union={len(keysA | keysB)}')
        print(
            f'coverage={cov:.3f}  agreement(isclose)={agree:.3f}  mismatches={len(mism)}'
        )
        print('')

        if onlyA:
            print(f'Only in A (HELM public): {len(onlyA)}')
            for (metric, split), n in missingA.most_common(20):
                print(f'  {metric} split={split}  (x{n})')
            print('')
        if onlyB:
            print(f'Only in B (your run): {len(onlyB)}')
            for (metric, split), n in missingB.most_common(20):
                print(f'  {metric} split={split}  (x{n})')
            print('')

        if mism:
            print(f'Top {min(topn, len(mism))} mean deltas:')
            for absd, reld, k in mism[:topn]:
                ra = A[k]
                rb = B[k]
                print(
                    f'  abs={absd:.6g} rel={reld:.6g} | '
                    f'{ra["metric"]} split={ra["split"]} | A={ra["mean"]:.6g} B={rb["mean"]:.6g}'
                )
            print('')
        else:
            print('No core metric mismatches (within tolerance).')
            print('')

    # Return a compact dict for downstream (e.g. sankey / triage)
    out = {
        'core_nA': len(A),
        'core_nB': len(B),
        'core_isect': len(isect),
        'core_union': len(keysA | keysB),
        'core_coverage': cov,
        'core_agreement': agree,
        'core_mismatches': len(mism),
        'core_onlyA': len(onlyA),
        'core_onlyB': len(onlyB),
        # optionally keep a small sample for debugging
        'core_top_mismatches': [
            {
                'metric': A[k]['metric'],
                'split': A[k]['split'],
                'A_mean': A[k]['mean'],
                'B_mean': B[k]['mean'],
                'absdiff': absd,
                'reldiff': reld,
            }
            for absd, reld, k in mism[:10]
        ],
    }
    return out


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
    # common “hierarchical” metrics
    if name.startswith('air_'):
        return 'air'
    if name.startswith('bias_metric:'):
        return 'bias_metric'
    if name.startswith('safety_'):
        return 'safety'
    if name.startswith('bbq_'):
        return 'bbq'
    # “operational” metrics you might want to ignore for score agreement
    if name in {
        'num_prompt_tokens',
        'num_completion_tokens',
        'num_output_tokens',
        'inference_runtime',
        'training_co2_cost',
        'training_energy_cost',
        'batch_size',
        'num_bytes',
        'num_perplexity_tokens',
        'max_prob',
        'logprob',
        'perplexity',
        'bits_per_byte',
        'logprob_per_byte',
    }:
        return 'ops'
    # default: take prefix before '@' or before first '_' if it looks like a family
    if '@' in name:
        return name.split('@', 1)[0]
    m = re.match(r'^[a-z]+', name)
    return m.group(0) if m else name


def count_stat_information(stat_list):
    from collections import Counter

    histo1 = Counter()
    histo2 = Counter()

    histograms = {
        'counts': Counter([s['count'] for s in stat_list]),
        'perturbed': Counter(),
        'splits': Counter(),
        'metric_family': Counter(),
    }

    for stat in stat_list:
        if stat['count'] > 0:
            statname = ub.udict(stat['name'])
            perterbation = statname.pop('perturbation', None)
            has_perturbation = bool(perterbation)
            if perterbation:
                peterb_name = perterbation['name']
                peterb_hash = ub.hash_data(perterbation, base=36)
                perturbation_id = peterb_name + '_' + peterb_hash[0:12]
                statname['perturbation_id'] = perturbation_id
            else:
                statname['perturbation_id'] = None
            statname2 = statname.copy()
            statname2['is_perturbed'] = statname2.pop('perturbation_id') is not None
            key1 = ub.urepr(statname, compact=1, nobr=1, nl=0)
            key2 = ub.urepr(statname2, compact=1, nobr=1, nl=0)
            histo1[key1] += 1
            histo2[key2] += 1
            histograms['splits'][stat['name']['split']] += 1
            histograms['perturbed'][has_perturbation] += 1
            metric_name = stat['name']['name']
            metric_fam = metric_family(metric_name)
            histograms['metric_family'][metric_fam] += 1

    # print(f'histo2 = {ub.urepr(histo2, nl=1)}')
    print(f'histograms = {ub.urepr(histograms, nl=2)}')


def explore_helm_stats():
    all_stats_lists = []
    for helm_row in ub.ProgIter(helm_rows, desc='explore helm stats'):
        run_dir = ub.Path(helm_row['run_dir'])
        helm_run = HelmRun.coerce(run_dir)
        helm_stats = helm_run.json.stats()
        stats_list = helm_stats
        all_stats_lists += stats_list
    count_stat_information(all_stats_lists)


OPS_FAMILIES = {'ops', 'finish', 'num', 'prompt'}  # from your histogram work


def stat_family(metric_name: str) -> str:
    # Use *your* metric_family function if you already have it; otherwise:
    if metric_name.startswith('air_'):
        return 'air'
    if metric_name.startswith('bias_metric:'):
        return 'bias_metric'
    if metric_name.startswith('safety_'):
        return 'safety'
    if metric_name.startswith('bbq_'):
        return 'bbq'
    # token/runtime-ish
    if metric_name.startswith('num_'):
        return 'num'
    if metric_name.startswith('finish_reason_'):
        return 'finish'
    if metric_name.startswith('prompt_'):
        return 'prompt'
    # fall back to first chunk before '_' or ':'
    return metric_name.split('_', 1)[0].split(':', 1)[0]


def stat_kind_from_family(fam: str) -> str:
    return 'ops' if fam in OPS_FAMILIES else 'task'


def stat_bucket(is_perturbed: bool, kind: str) -> str:
    return ('pert' if is_perturbed else 'base') + '_' + kind


def stable_stat_id(stat) -> str:
    # stable hash of stat["name"] only (not values)
    name_obj = deep_sort_keys(stat['name'])
    return ub.hash_data(name_obj, base=36)


def stat_meta_and_value(stat):
    """
    Returns:
      key, bucket, split, family, kind, mean
    """
    name = stat['name']
    metric_name = name['name']
    split = name.get('split', None)
    is_pert = 'perturbation' in name
    fam = stat_family(metric_name)
    kind = stat_kind_from_family(fam)
    bucket = stat_bucket(is_pert, kind)
    key = stable_stat_id(stat)
    mean = stat.get('mean', None)
    return key, bucket, split, fam, kind, mean


def build_stat_index(stats_list, *, require_mean=True, drop_zero_count=True):
    """
    Build an index:
      idx[bucket][split][key] = (mean, family)
    """
    idx = ub.ddict(lambda: ub.ddict(dict))
    for s in stats_list:
        if drop_zero_count and s.get('count', 0) == 0:
            continue
        key, bucket, split, fam, kind, mean = stat_meta_and_value(s)
        if require_mean and mean is None:
            continue
        idx[bucket][split][key] = (float(mean) if mean is not None else None, fam)
    return idx


def compare_indices(A, B, *, rel_tol=1e-4, abs_tol=1e-8):
    """
    Compare two indices from build_stat_index.
    Returns a nested summary dict suitable for Sankey labels.
    """
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
            # optional: see which families contribute most to mismatches
            fam_mismatch = ub.ddict(int)

            for k in isect:
                va, fa = a[k]
                vb, fb = b[k]
                if va is None or vb is None:
                    continue
                ok = math.isclose(va, vb, rel_tol=rel_tol, abs_tol=abs_tol)
                n_close += int(ok)
                if not ok:
                    # prefer family from A; fall back to B
                    fam = fa if fa is not None else fb
                    fam_mismatch[fam] += 1

            agree = (n_close / n_isect) if n_isect else 0.0

            # store a compact per-split record
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


def summarize_for_sankey(comp_summary):
    """
    Turn compare_indices output into a few scalar features + one bucket label.
    You can tune thresholds here.
    """

    def get(bucket, split, key, default=None):
        return comp_summary.get(bucket, {}).get(split, {}).get(key, default)

    # Aggregate over splits (valid/test) for each bucket:
    def agg(bucket, key):
        # weighted by n_isect so tiny splits don't dominate
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

    feats = {}
    for bucket in ['base_task', 'base_ops', 'pert_task', 'pert_ops']:
        feats[f'{bucket}_coverage'] = agg(bucket, 'coverage')
        feats[f'{bucket}_agreement'] = agg(bucket, 'agreement')

    # A single label you can use for Sankey; tune to taste
    # Priority: base_task first (usually what you care about)
    bt_cov = feats['base_task_coverage'] or 0.0
    bt_ag = feats['base_task_agreement'] or 0.0
    pt_cov = feats['pert_task_coverage'] or 0.0
    pt_ag = feats['pert_task_agreement'] or 0.0

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

    # optional secondary label for perturbed task
    if pt_cov < 0.98:
        label2 = 'pert_task: coverage mismatch'
    elif pt_ag >= 0.99:
        label2 = 'pert_task: near match'
    elif pt_ag >= 0.95:
        label2 = 'pert_task: close'
    elif pt_ag >= 0.80:
        label2 = 'pert_task: partial'
    else:
        label2 = 'pert_task: low'

    feats['agreement_bucket_base_task'] = label
    feats['agreement_bucket_pert_task'] = label2
    return feats


# sankey_flow = nx.DiGraph()

Row = Dict[str, Any]
By = Union[str, Callable[[Row], Any]]

# ----------------------------
# DSL step definitions
# ----------------------------


@dataclass(frozen=True)
class Root:
    label: str


@dataclass(frozen=True)
class Group:
    name: str
    by: By


@dataclass(frozen=True)
class Bucket:
    name: str
    by: By


@dataclass(frozen=True)
class Split:
    name: str
    by: By
    branches: Dict[Any, 'Plan']
    default: Optional['Plan'] = None


def _eval_by(by: By, row: Row):
    return by(row) if callable(by) else row.get(by)


def _label(stage: str, value: Any, fmt: str):
    return fmt.format(name=stage, value=value)


def _is_root_step(step) -> bool:
    return isinstance(step, Root)


# ----------------------------
# Plan with methods
# ----------------------------


@dataclass
class Plan:
    steps: List[Any] = field(default_factory=list)

    def __init__(self, *steps):
        self.steps = list(steps)

    # ---- Pretty printing the plan ----

    def to_text(self) -> str:
        """Human-friendly plan pretty-print."""
        lines: List[str] = []

        def rec(plan: Plan, indent: str = ''):
            for st in plan.steps:
                if isinstance(st, Root):
                    lines.append(f'{indent}ROOT {st.label!r}')
                elif isinstance(st, Group):
                    lines.append(f'{indent}GROUP {st.name!r} by={_by_repr(st.by)}')
                elif isinstance(st, Bucket):
                    lines.append(f'{indent}BUCKET {st.name!r} by={_by_repr(st.by)}')
                elif isinstance(st, Split):
                    lines.append(f'{indent}SPLIT {st.name!r} by={_by_repr(st.by)}')
                    for k, sub in st.branches.items():
                        lines.append(f'{indent}  BRANCH {k!r}:')
                        rec(sub, indent + '    ')
                    if st.default is not None:
                        lines.append(f'{indent}  DEFAULT:')
                        rec(st.default, indent + '    ')
                else:
                    lines.append(f'{indent}{type(st).__name__} (?)')

        rec(self, '')
        return '\n'.join(lines)

    # ---- Trace a single row through the plan ----

    def trace(self, row: Row, *, label_fmt='{name}: {value}') -> List[str]:
        """Return the node sequence (path) this row takes."""
        root = self._find_root_label(default='All HELM')
        path = [root]

        def run(plan: Plan, cur: str):
            node = cur
            for st in plan.steps:
                if isinstance(st, Root):
                    continue
                elif isinstance(st, Group):
                    val = _eval_by(st.by, row)
                    nxt = _label(st.name, val, label_fmt)
                    path.append(nxt)
                    node = nxt
                elif isinstance(st, Bucket):
                    val = _eval_by(st.by, row)
                    nxt = _label(st.name, val, label_fmt)
                    path.append(nxt)
                    node = nxt
                elif isinstance(st, Split):
                    key = _eval_by(st.by, row)
                    split_node = _label(st.name, key, label_fmt)
                    path.append(split_node)
                    node = split_node
                    branch = st.branches.get(key) or st.default
                    if branch is None:
                        return node
                    node = run(branch, node)
                else:
                    raise TypeError(f'Unknown step type: {type(st)}')
            return node

        run(self, root)
        # path includes nodes *after* each step; edges are zip(path, path[1:])
        return path

    # ---- Build the sankey DiGraph ----

    def build_sankey(
        self,
        rows: Iterable[Row],
        *,
        weight: Union[float, Callable[[Row], float]] = 1.0,
        label_fmt: str = '{name}: {value}',
        edge_attr: str = 'value',
    ) -> nx.DiGraph:
        """
        Build a DiGraph where edges carry aggregated flow in edge_attr.
        """
        G = nx.DiGraph()
        weight_fn = weight if callable(weight) else (lambda r: float(weight))
        # root = self._find_root_label(default='All HELM')

        def add_edge(u, v, w):
            if G.has_edge(u, v):
                G[u][v][edge_attr] += w
            else:
                G.add_edge(u, v, **{edge_attr: w})

        for row in rows:
            w = weight_fn(row)
            path = self.trace(row, label_fmt=label_fmt)
            for a, b in zip(path, path[1:]):
                add_edge(a, b, w)

        return G

    # ---- Pretty print the graph ----

    def graph_to_text(
        self,
        G: nx.DiGraph,
        *,
        edge_attr: str = 'value',
        max_edges: Optional[int] = 200,
        sort: str = 'value_desc',
    ) -> str:
        """
        Summarize nodes and weighted edges in a readable way.
        """
        lines: List[str] = []
        lines.append(f'Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}')
        lines.append('')

        # Node degrees / flow totals can be handy
        def outflow(n):
            return sum(G[n][v].get(edge_attr, 0) for v in G.successors(n))

        def inflow(n):
            return sum(G[u][n].get(edge_attr, 0) for u in G.predecessors(n))

        nodes_sorted = sorted(
            G.nodes, key=lambda n: (outflow(n), inflow(n)), reverse=True
        )
        lines.append('Top nodes by outflow/inflow:')
        for n in nodes_sorted[:20]:
            lines.append(f'  {n}  out={outflow(n):g} in={inflow(n):g}')
        lines.append('')

        edges = [(u, v, G[u][v].get(edge_attr, 0)) for u, v in G.edges]
        if sort == 'value_desc':
            edges.sort(key=lambda t: t[2], reverse=True)
        elif sort == 'lex':
            edges.sort(key=lambda t: (str(t[0]), str(t[1])))

        lines.append('Edges:')
        shown = edges if max_edges is None else edges[:max_edges]
        for u, v, val in shown:
            lines.append(f'  {u}  ->  {v}   {edge_attr}={val:g}')
        if max_edges is not None and len(edges) > max_edges:
            lines.append(f'... ({len(edges) - max_edges} more edges)')
        return '\n'.join(lines)

    # ---- internal ----

    def _find_root_label(self, default='All HELM') -> str:
        for st in self.steps:
            if isinstance(st, Root):
                return st.label
        return default


def _by_repr(by: By) -> str:
    if isinstance(by, str):
        return f'{by!r}'
    # try to keep it readable (named function is nicest)
    name = getattr(by, '__name__', None)
    if name:
        return f'<fn {name}>'
    return f'<callable {by!r}>'


# def stable_stat_id(name_obj):
#     return ub.hash_data(deep_sort_keys(name_obj), base=36)


# def metric_family(metric_name: str) -> str:
#     # Use your improved family mapping if you have one already
#     if metric_name.startswith('air_'):
#         return 'air'
#     if metric_name.startswith('bias_metric:'):
#         return 'bias_metric'
#     if metric_name.startswith('safety_'):
#         return 'safety'
#     if metric_name.startswith('bbq_'):
#         return 'bbq'
#     if metric_name.startswith('finish_reason_'):
#         return 'finish'
#     if metric_name.startswith('prompt_'):
#         return 'prompt'
#     if metric_name.startswith('num_'):
#         return 'num'
#     if metric_name in {
#         'inference_runtime',
#         'training_co2_cost',
#         'training_energy_cost',
#         'batch_size',
#         'num_bytes',
#         'num_perplexity_tokens',
#         'max_prob',
#         'logprob',
#         'perplexity',
#         'bits_per_byte',
#         'logprob_per_byte',
#     }:
#         return 'ops'
#     return metric_name.split('_', 1)[0].split(':', 1)[0]


def stat_meta(stat):
    """
    Extract metadata needed for grouping diffs.
    """
    name = stat['name']
    metric = name['name']
    split = name.get('split', None)
    perturb = name.get('perturbation', None)
    pert_name = perturb.get('name', None) if perturb else None
    is_pert = perturb is not None
    fam = metric_family(metric)
    kind = 'ops' if fam in OPS_FAMILIES else 'task'
    key = stable_stat_id(name)
    mean = stat.get('mean', None)
    count = stat.get('count', 0)
    return {
        'key': key,
        'metric': metric,
        'family': fam,
        'kind': kind,
        'split': split,
        'is_pert': is_pert,
        'pert_name': pert_name,
        'count': count,
        'mean': None if mean is None else float(mean),
        'name_obj': name,  # keep if you want to print/debug
    }


def index_stats(stats_list):
    """
    key -> meta dict
    (filters out count==0 by default, because those are usually N/A)
    """
    idx = {}
    for s in stats_list:
        if s.get('count', 0) == 0:
            continue
        m = stat_meta(s)
        idx[m['key']] = m
    return idx


def diff_report(statsA, statsB, *, rel_tol=1e-4, abs_tol=1e-8, topn=3, title=None):
    A = index_stats(statsA)
    B = index_stats(statsB)

    keysA = set(A)
    keysB = set(B)
    isect = keysA & keysB
    onlyA = keysA - keysB
    onlyB = keysB - keysA

    # Categorize differences among intersected keys by value mismatch
    mism = []
    for k in isect:
        a = A[k]['mean']
        b = B[k]['mean']
        if a is None or b is None:
            continue
        if not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
            absd = abs(a - b)
            reld = absd / (abs(b) + 1e-12)
            mism.append((absd, reld, k))

    mism.sort(reverse=True)

    def summarize_keys(keys, src):
        c = Counter()
        for k in keys:
            m = src[k]
            c[('is_pert', m['is_pert'])] += 1
            c[('family', m['family'])] += 1
            c[('kind', m['kind'])] += 1
            c[('split', m['split'])] += 1
            if m['is_pert']:
                c[('pert_name', m['pert_name'])] += 1
        return c

    cov = len(isect) / max(1, len(keysA | keysB))

    print('')
    print('=' * 80)
    print(title or 'RUN DIFF REPORT')
    print(f'coverage(isect/union) = {len(isect)}/{len(keysA | keysB)} = {cov:.3f}')
    print(f'onlyA={len(onlyA)}  onlyB={len(onlyB)}  value_mismatches={len(mism)}')
    print('')

    # Coverage diffs: what's missing/extra?
    if onlyA:
        cA = summarize_keys(onlyA, A)
        print('[Coverage] Present only in A (HELM public) summary:')
        _print_counter_block(cA, topn=8)
        print('')
    if onlyB:
        cB = summarize_keys(onlyB, B)
        print('[Coverage] Present only in B (your run) summary:')
        _print_counter_block(cB, topn=8)
        print('')

    # Value mismatches: among shared keys, what differs?
    if mism:
        famC = Counter()
        pertC = Counter()
        for _, _, k in mism:
            m = A.get(k) or B.get(k)
            famC[m['family']] += 1
            pertC[m['is_pert']] += 1
        print('[Value] Mismatch breakdown:')
        print(f'  is_perturbed: {dict(pertC)}')
        print(f'  top families: {famC.most_common(10)}')
        print('')

        if 1:
            print(f'[Value] Top {topn} absolute mean deltas:')
            for absd, reld, k in mism[:topn]:
                a = A[k]
                b = B[k]
                print(
                    f'  abs={absd:.3g} rel={reld:.3g} '
                    f'| fam={a["family"]} kind={a["kind"]} split={a["split"]} '
                    f'pert={a["pert_name"] if a["is_pert"] else "-"} '
                    f'| metric={a["metric"]}'
                )
    print('=' * 80)
    print('')


def _print_counter_block(C: Counter, topn=8):
    # show small curated views
    def show(prefix):
        items = [(k[1], v) for k, v in C.items() if k[0] == prefix]
        items.sort(key=lambda x: x[1], reverse=True)
        print(f'  {prefix}: {items[:topn]}')

    show('is_pert')
    show('kind')
    show('split')
    show('family')
    show('pert_name')


kwdagger_lut = {r['run_spec_name']: r for r in kwdagger_rows}
for helm_row in ub.ProgIter(helm_rows, desc='compare runs'):
    run_dir = ub.Path(helm_row['run_dir'])
    run_spec_name = helm_row['run_spec_name']

    suite_name = run_dir.parent.name
    benchmark_name = run_dir.parent.parent.parent.parent.name
    assert run_dir.parent.parent.parent.name == 'benchmark_output'
    assert run_dir.parent.parent.name == 'runs'

    helm_row['suite_name'] = suite_name
    helm_row['benchmark_name'] = benchmark_name

    kwdagger_row = kwdagger_lut.get(run_spec_name)
    helm_row['reproduced_step1'] = kwdagger_row is not None

    if kwdagger_row is None:
        # useful for Sankey
        helm_row['agreement_bucket_base_task'] = 'not attempted'
        helm_row['agreement_bucket_pert_task'] = 'not attempted'
        continue

    # load stats
    helm_run = HelmRun.coerce(run_dir)
    kwdg_run = kwdagger_row['run']

    helm_stats = helm_run.json.stats()
    kwdg_stats = kwdg_run.json.stats()

    # Build indices (bucket -> split -> key -> (mean, family))
    A = build_stat_index(helm_stats, require_mean=True, drop_zero_count=True)
    B = build_stat_index(kwdg_stats, require_mean=True, drop_zero_count=True)

    comp = compare_indices(A, B, rel_tol=1e-4, abs_tol=1e-8)
    feats = summarize_for_sankey(comp)

    # Store compact stuff for Sankey
    helm_row.update(
        {
            # base task is usually the headline
            'base_task_cov': feats['base_task_coverage'],
            'base_task_agree': feats['base_task_agreement'],
            'pert_task_cov': feats['pert_task_coverage'],
            'pert_task_agree': feats['pert_task_agreement'],
            'agreement_bucket_base_task': feats['agreement_bucket_base_task'],
            'agreement_bucket_pert_task': feats['agreement_bucket_pert_task'],
            # keep raw comp if you want drill-down later (can be large)
            # "comp_summary": comp,
        }
    )

    core_info = compare_core_stats(helm_stats, kwdg_stats, rel_tol=1e-4, abs_tol=1e-8, verbose=0)
    helm_row['core_info'] = core_info

    # If you want to print only when core looks "bad":
    if (
        core_info['core_mismatches'] > 0
        or core_info['core_onlyA']
        or core_info['core_onlyB']
    ):
        print(run_spec_name)
        compare_core_stats(helm_stats, kwdg_stats, rel_tol=1e-4, abs_tol=1e-8, verbose=1)

    if helm_row.get('agreement_bucket_base_task') == 'base_task: coverage mismatch':
        diff_report(helm_stats, kwdg_stats, title=run_spec_name)

    # Optional: keep top mismatch families (helps debugging and possibly Sankey)
    # e.g. families most often mismatching within base_task (aggregated across splits)
    def top_fams(bucket):
        fam_counts = ub.ddict(int)
        for split, rec in comp.get(bucket, {}).items():
            for fam, n in rec.get('fam_mismatch_top', []):
                fam_counts[fam] += n
        return sorted(fam_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    helm_row['base_task_mismatch_fams'] = top_fams('base_task')
    helm_row['pert_task_mismatch_fams'] = top_fams('pert_task')


df = pd.DataFrame(helm_rows)
df.value_counts(['benchmark_name', 'reproduced_step1'])


def attempt_status(r):
    return 'attempted' if r.get('reproduced_step1', False) else 'not_attempted'


def agreement_label(r):
    return r.get('agreement_bucket_base_task', 'unknown')


plan = Plan(
    Root('All HELM'),
    Group('benchmark', by='benchmark_name'),
    Bucket('attempt', by=attempt_status),
    Bucket('agreement', by=agreement_label),
)


print(plan.to_text())

sankey_flow = plan.build_sankey(
    helm_rows, label_fmt='{name}: {value}'
)  # or "{value}" if safe
print(plan.graph_to_text(sankey_flow, max_edges=150))

print(nx.write_network_text(sankey_flow))


def nx_to_sankey(G):
    # stable node order: topological if possible, else insertion
    try:
        nodes = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        nodes = list(G.nodes)

    idx = {n: i for i, n in enumerate(nodes)}

    source = []
    target = []
    value = []
    for u, v, data in G.edges(data=True):
        source.append(idx[u])
        target.append(idx[v])
        value.append(data.get('value', 0))

    return nodes, source, target, value


nodes, source, target, value = nx_to_sankey(sankey_flow)

# nodes = ["All HELM", "Found locally", "Missing", "Stats match", "Stats mismatch"]

# links = [
#     {"source": "All HELM", "target": "Found locally", "value": 145},
#     {"source": "All HELM", "target": "Missing", "value": 282},
#     {"source": "Found locally", "target": "Stats match", "value": 80},
#     {"source": "Found locally", "target": "Stats mismatch", "value": 65},
# ]

# Map labels -> indices
# idx = {name: i for i, name in enumerate(nodes)}
# source = [idx[e["source"]] for e in links]
# target = [idx[e["target"]] for e in links]
# value  = [e["value"] for e in links]

fig = go.Figure(
    go.Sankey(
        node=dict(label=nodes, pad=15, thickness=18),
        link=dict(source=source, target=target, value=value),
    )
)

fig.update_layout(
    title_text='HELM Reproduction Funnel',
    font_size=14,
)

fig.write_image('helm_repro_sankey.png', scale=2)  # higher scale = sharper
fig.write_image('helm_repro_sankey.jpg', scale=2)
# For papers, SVG/PDF is often best:
fig.write_image('helm_repro_sankey.svg')
fig.write_image('helm_repro_sankey.pdf')

# import wormhole
ub.cmd('wormhole send helm_repro_sankey.jpg', verbose=3)
"""
!wormhole send helm_repro_sankey.jpg
"""
