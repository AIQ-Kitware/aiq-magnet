from __future__ import annotations

import argparse
import datetime as datetime_mod
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import kwutil
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from magnet.backends.helm.helm_outputs import HelmRun
from magnet.backends.helm.helm_run_analysis import HelmRunAnalysis
from magnet.backends.helm.helm_run_diff import HelmRunDiff
from magnet.backends.helm.util import helm_metrics


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    alpha = pos - lo
    return values[lo] * (1 - alpha) + values[hi] * alpha


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _run_level_core_rows(diff: HelmRunDiff) -> list[dict[str, Any]]:
    idx_a = diff.a.stat_index(drop_zero_count=True, require_mean=True, short_hash=diff.short_hash)
    idx_b = diff.b.stat_index(drop_zero_count=True, require_mean=True, short_hash=diff.short_hash)
    rows = []
    for k in set(idx_a) & set(idx_b):
        a = idx_a[k]
        b = idx_b[k]
        if a.mean is None or b.mean is None:
            continue
        if a.metric_class != 'core':
            continue
        abs_delta = abs(a.mean - b.mean)
        denom = max(abs(a.mean), abs(b.mean), 1e-12)
        rel_delta = abs_delta / denom
        rows.append({
            'key': k,
            'metric': a.metric,
            'metric_class': a.metric_class,
            'a': float(a.mean),
            'b': float(b.mean),
            'abs_delta': abs_delta,
            'rel_delta': rel_delta,
        })
    return rows


def _iter_joined_rows(joined, row_by_key):
    if row_by_key is not None:
        return row_by_key.values()
    if isinstance(joined, dict):
        return joined.values()
    if hasattr(joined, '__iter__'):
        return joined
    return []


def _row_key(row: Any) -> Any:
    return (
        getattr(row, 'key', None)
        or getattr(row, 'stat_key', None)
        or getattr(row, 'row_key', None)
        or row
    )


def _instance_level_core_rows(diff: HelmRunDiff) -> list[dict[str, Any]]:
    joined_a = diff.a.joined_instance_stat_table(assert_assumptions=False, short_hash=diff.short_hash)
    joined_b = diff.b.joined_instance_stat_table(assert_assumptions=False, short_hash=diff.short_hash)
    map_a = getattr(joined_a, 'row_by_key', None)
    map_b = getattr(joined_b, 'row_by_key', None)
    if map_a is None:
        map_a = {_row_key(r): r for r in _iter_joined_rows(joined_a, map_a)}
    if map_b is None:
        map_b = {_row_key(r): r for r in _iter_joined_rows(joined_b, map_b)}

    rows = []
    for k in set(map_a) & set(map_b):
        ra = map_a[k]
        rb = map_b[k]
        sa = getattr(ra, 'stat', None) if hasattr(ra, 'stat') else (ra.get('stat') if isinstance(ra, dict) else None)
        sb = getattr(rb, 'stat', None) if hasattr(rb, 'stat') else (rb.get('stat') if isinstance(rb, dict) else None)
        ma = _safe_float((sa or {}).get('mean') if isinstance(sa, dict) else getattr(sa, 'mean', None))
        mb = _safe_float((sb or {}).get('mean') if isinstance(sb, dict) else getattr(sb, 'mean', None))
        ca = int((sa or {}).get('count', 0) or 0) if isinstance(sa, dict) else int(getattr(sa, 'count', 0) or 0)
        cb = int((sb or {}).get('count', 0) or 0) if isinstance(sb, dict) else int(getattr(sb, 'count', 0) or 0)
        if ma is None or mb is None or ca == 0 or cb == 0:
            continue
        name_obj = (sa or {}).get('name') if isinstance(sa, dict) else getattr(sa, 'name_obj', None)
        metric = name_obj.get('name') if isinstance(name_obj, dict) else getattr(sa, 'metric', None)
        metric_class, _ = helm_metrics.classify_metric(metric)
        if metric_class != 'core':
            continue
        abs_delta = abs(ma - mb)
        denom = max(abs(ma), abs(mb), 1e-12)
        rel_delta = abs_delta / denom
        rows.append({
            'key': k,
            'metric': metric,
            'metric_class': metric_class,
            'a': ma,
            'b': mb,
            'abs_delta': abs_delta,
            'rel_delta': rel_delta,
        })
    return rows


def _group_quantiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(float(r['abs_delta']) for r in rows)
    return {
        'count': len(values),
        'abs_delta': {
            'min': _quantile(values, 0.0),
            'p50': _quantile(values, 0.5),
            'p90': _quantile(values, 0.9),
            'p95': _quantile(values, 0.95),
            'p99': _quantile(values, 0.99),
            'max': _quantile(values, 1.0),
        },
    }


def _metric_quantiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_metric.setdefault(str(row['metric']), []).append(row)
    out = []
    for metric, items in sorted(by_metric.items()):
        info = _group_quantiles(items)
        info['metric'] = metric
        out.append(info)
    return out


def _metric_descriptor(metric: str) -> dict[str, str]:
    if metric in {
        'exact_match',
        'prefix_exact_match',
        'quasi_exact_match',
        'quasi_prefix_exact_match',
        'exact_match@5',
        'prefix_exact_match@5',
        'quasi_exact_match@5',
        'quasi_prefix_exact_match@5',
    }:
        return {
            'kind': 'binary',
            'range': '0 to 1',
            'direction': 'higher is better',
        }
    if metric in {'bleu_1', 'bleu_4', 'f1_score', 'rouge_l'}:
        return {
            'kind': 'bounded overlap score',
            'range': '0 to 1',
            'direction': 'higher is better',
        }
    return {
        'kind': 'score',
        'range': 'metric-dependent',
        'direction': 'higher is better unless documented otherwise',
    }


def _should_treat_as_discrete(values) -> bool:
    values = [float(v) for v in values if v is not None]
    unique_values = sorted(set(values))
    if not unique_values:
        return False
    return len(unique_values) <= 8 and all(v in {0.0, 1.0} for v in unique_values)


def _agreement_curve(rows: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    if not rows:
        return []
    vals = [float(r['abs_delta']) for r in rows]
    out = []
    for t in thresholds:
        matched = sum(v <= t for v in vals)
        out.append({
            'abs_tol': t,
            'agree_ratio': matched / len(vals),
            'matched': matched,
            'count': len(vals),
        })
    return out


def _infer_run_spec_name(*run_paths: str) -> str:
    names = [Path(p).name for p in run_paths if p]
    names = [n for n in names if n]
    if not names:
        return 'unknown_run_spec'
    unique = sorted(set(names))
    if len(unique) == 1:
        return unique[0]
    return unique[0]


def _build_pair(run_a: str, run_b: str, label: str, thresholds: list[float]) -> dict[str, Any]:
    diff = HelmRunDiff(HelmRun.coerce(run_a), HelmRun.coerce(run_b), a_name=f'{label}:A', b_name=f'{label}:B')
    run_rows = _run_level_core_rows(diff)
    inst_rows = _instance_level_core_rows(diff)
    return {
        'label': label,
        'inputs': {
            'run_a': str(Path(run_a).expanduser().resolve()),
            'run_b': str(Path(run_b).expanduser().resolve()),
        },
        'diagnosis': diff.summary_dict(level=20).get('diagnosis', {}),
        'core_metrics': sorted({str(r['metric']) for r in inst_rows}),
        'run_level': {
            'n_rows': len(run_rows),
            'overall_quantiles': _group_quantiles(run_rows),
            'by_metric': _metric_quantiles(run_rows),
            'agreement_vs_abs_tol': _agreement_curve(run_rows, thresholds),
        },
        'instance_level': {
            'n_rows': len(inst_rows),
            'overall_quantiles': _group_quantiles(inst_rows),
            'by_metric': _metric_quantiles(inst_rows),
            'agreement_vs_abs_tol': _agreement_curve(inst_rows, thresholds),
        },
        '_instance_rows': inst_rows,
    }


def _agreement_curve_rows(pair_a: dict[str, Any], pair_b: dict[str, Any], level_key: str) -> list[dict[str, Any]]:
    rows = []
    for pair in [pair_a, pair_b]:
        for row in pair[level_key]['agreement_vs_abs_tol']:
            rows.append({
                'pair': pair['label'],
                'abs_tol': float(row['abs_tol']),
                'agree_ratio': float(row['agree_ratio']),
            })
    return rows


def _plot_distribution(ax, pair_a: dict[str, Any], pair_b: dict[str, Any], level_key: str) -> None:
    rows = pd.DataFrame(_agreement_curve_rows(pair_a, pair_b, level_key))
    sns.lineplot(
        ax=ax,
        data=rows,
        x='abs_tol',
        y='agree_ratio',
        hue='pair',
        style='pair',
        markers=True,
        dashes=False,
        linewidth=2,
    )
    ax.set_xscale('symlog', linthresh=1e-12)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Absolute Tolerance Threshold for Core Metric Difference')
    ax.set_ylabel('Fraction of Core Metric Comparisons in Agreement')
    ax.legend(title='')


def _plot_quantiles(ax, pair_a: dict[str, Any], pair_b: dict[str, Any], level_key: str, title: str) -> None:
    labels = ['p50', 'p90', 'p95', 'p99', 'max']
    x = list(range(len(labels)))
    a_vals = [pair_a[level_key]['overall_quantiles']['abs_delta'][k] for k in labels]
    b_vals = [pair_b[level_key]['overall_quantiles']['abs_delta'][k] for k in labels]
    ax.plot(x, a_vals, marker='o', label=pair_a['label'])
    ax.plot(x, b_vals, marker='o', label=pair_b['label'])
    ax.set_xticks(x, labels)
    ax.set_yscale('symlog', linthresh=1e-12)
    ax.set_title(title)
    ax.set_xlabel('Quantile')
    ax.set_ylabel('Absolute Difference in Core Metric Value')
    ax.legend()


def _distribution_rows(pair: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in pair.get('_instance_rows', []):
        rows.append({
            'pair': pair['label'],
            'metric': row['metric'],
            'side': 'A',
            'value': float(row['a']),
        })
        rows.append({
            'pair': pair['label'],
            'metric': row['metric'],
            'side': 'B',
            'value': float(row['b']),
        })
    return pd.DataFrame(rows)


def _plot_metric_distributions(fig_dpath: Path, stamp: str, left: dict[str, Any], right: dict[str, Any], run_spec_name: str) -> Path:
    df = pd.concat([
        _distribution_rows(left),
        _distribution_rows(right),
    ], ignore_index=True)
    metrics = sorted(df['metric'].dropna().unique().tolist())
    pair_order = [left['label'], right['label']]
    fig, axes = plt.subplots(
        len(pair_order),
        len(metrics),
        figsize=(5.2 * len(metrics), 4.2 * len(pair_order)),
        constrained_layout=True,
    )
    if len(pair_order) == 1 and len(metrics) == 1:
        axes = [[axes]]
    elif len(pair_order) == 1:
        axes = [axes]
    elif len(metrics) == 1:
        axes = [[ax] for ax in axes]
    for row_idx, pair_label in enumerate(pair_order):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            sub = df[(df['pair'] == pair_label) & (df['metric'] == metric)]
            discrete = _should_treat_as_discrete(sub['value'].tolist())
            sns.histplot(
                data=sub,
                x='value',
                hue='side',
                stat='probability',
                common_norm=False,
                discrete=discrete,
                multiple='dodge',
                shrink=0.8,
                bins=None if discrete else 20,
                ax=ax,
            )
            ax.set_title(f'{pair_label}\n{metric}')
            ax.set_xlabel('Core metric value')
            ax.set_ylabel('Probability')
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title('')
    fig.suptitle(
        'Core Metric Score Distributions Within Each Comparison Pair\n'
        f'Run Spec: {run_spec_name}\n'
        'Each panel shows the per-instance score distribution for side A vs side B.',
        fontsize=16,
    )
    out_fpath = fig_dpath / f'core_metric_distributions_{stamp}.png'
    fig.savefig(out_fpath, dpi=180)
    plt.close(fig)
    return out_fpath


def _single_run_instance_core_rows(run_path: str, label: str) -> pd.DataFrame:
    ana = HelmRunAnalysis(HelmRun.coerce(run_path), name=label)
    joined = ana.joined_instance_stat_table(assert_assumptions=False)
    row_by_key = getattr(joined, 'row_by_key', None) or {}
    rows = []
    for row in row_by_key.values():
        stat = row.stat
        mean = _safe_float(stat.get('mean'))
        count = int(stat.get('count', 0) or 0)
        if mean is None or count == 0:
            continue
        name_obj = stat.get('name', {})
        metric = name_obj.get('name')
        metric_class, _ = helm_metrics.classify_metric(metric)
        if metric_class != 'core':
            continue
        rows.append({
            'run': label,
            'metric': metric,
            'value': float(mean),
        })
    return pd.DataFrame(rows)


def _plot_three_run_metric_distributions(
    fig_dpath: Path,
    stamp: str,
    kwdagger_a_run: str,
    kwdagger_b_run: str,
    official_run: str,
    run_spec_name: str,
) -> Path:
    df = pd.concat([
        _single_run_instance_core_rows(kwdagger_a_run, 'kwdagger A'),
        _single_run_instance_core_rows(kwdagger_b_run, 'kwdagger B'),
        _single_run_instance_core_rows(official_run, 'official'),
    ], ignore_index=True)
    metrics = sorted(df['metric'].dropna().unique().tolist())
    run_order = ['kwdagger A', 'kwdagger B', 'official']
    fig, axes = plt.subplots(
        len(metrics),
        len(run_order),
        figsize=(5.0 * len(run_order), 3.2 * len(metrics)),
        constrained_layout=True,
    )
    if len(metrics) == 1 and len(run_order) == 1:
        axes = [[axes]]
    elif len(metrics) == 1:
        axes = [axes]
    elif len(run_order) == 1:
        axes = [[ax] for ax in axes]
    for row_idx, metric in enumerate(metrics):
        for col_idx, run_label in enumerate(run_order):
            ax = axes[row_idx][col_idx]
            sub = df[(df['metric'] == metric) & (df['run'] == run_label)]
            discrete = _should_treat_as_discrete(sub['value'].tolist())
            sns.histplot(
                data=sub,
                x='value',
                stat='probability',
                discrete=discrete,
                shrink=0.8,
                bins=None if discrete else 20,
                ax=ax,
                color='#4C72B0',
            )
            if row_idx == 0:
                ax.set_title(run_label)
            ax.set_xlabel('Core metric value')
            ax.set_ylabel(metric if col_idx == 0 else '')
    fig.suptitle(
        'Per-Run Instance-Level Core Metric Score Distributions\n'
        f'Run Spec: {run_spec_name}\n'
        'Columns are kwdagger repeat A, kwdagger repeat B, and the official HELM run.',
        fontsize=16,
    )
    out_fpath = fig_dpath / f'core_metric_three_run_distributions_{stamp}.png'
    fig.savefig(out_fpath, dpi=180)
    plt.close(fig)
    return out_fpath


def _plot_overlay_metric_distributions(
    fig_dpath: Path,
    stamp: str,
    kwdagger_a_run: str,
    kwdagger_b_run: str,
    official_run: str,
    run_spec_name: str,
) -> Path:
    df = pd.concat([
        _single_run_instance_core_rows(kwdagger_a_run, 'kwdagger A'),
        _single_run_instance_core_rows(kwdagger_b_run, 'kwdagger B'),
        _single_run_instance_core_rows(official_run, 'official'),
    ], ignore_index=True)
    metrics = sorted(df['metric'].dropna().unique().tolist())
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(10, 3.2 * len(metrics)),
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = df[df['metric'] == metric].copy()
        discrete = _should_treat_as_discrete(sub['value'].tolist())
        sns.histplot(
            data=sub,
            x='value',
            hue='run',
            stat='probability',
            common_norm=False,
            element='step',
            fill=False,
            multiple='layer',
            discrete=discrete,
            bins=None if discrete else 20,
            ax=ax,
        )
        desc = _metric_descriptor(metric)
        ax.set_title(
            f"{metric} ({desc['kind']}, {desc['range']}, {desc['direction']})"
        )
        ax.set_xlabel('Instance-level metric value')
        ax.set_ylabel('Probability')
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title('')
    fig.suptitle(
        'Overlay of Per-Instance Core Metric Score Distributions by Run\n'
        f'Run Spec: {run_spec_name}\n'
        'This shows the raw score distributions for each core metric across kwdagger repeats and the official HELM run.',
        fontsize=16,
    )
    out_fpath = fig_dpath / f'core_metric_overlay_distributions_{stamp}.png'
    fig.savefig(out_fpath, dpi=180)
    plt.close(fig)
    return out_fpath


def _plot_overlay_metric_ecdfs(
    fig_dpath: Path,
    stamp: str,
    kwdagger_a_run: str,
    kwdagger_b_run: str,
    official_run: str,
    run_spec_name: str,
) -> Path:
    df = pd.concat([
        _single_run_instance_core_rows(kwdagger_a_run, 'kwdagger A'),
        _single_run_instance_core_rows(kwdagger_b_run, 'kwdagger B'),
        _single_run_instance_core_rows(official_run, 'official'),
    ], ignore_index=True)
    metrics = sorted(df['metric'].dropna().unique().tolist())
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(10, 3.2 * len(metrics)),
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = df[df['metric'] == metric].copy()
        sns.ecdfplot(
            data=sub,
            x='value',
            hue='run',
            ax=ax,
        )
        desc = _metric_descriptor(metric)
        ax.set_title(
            f"{metric} ECDF ({desc['kind']}, {desc['range']}, {desc['direction']})"
        )
        ax.set_xlabel('Instance-level metric value')
        ax.set_ylabel('Cumulative fraction of instances')
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title('')
    fig.suptitle(
        'ECDF of Per-Instance Core Metric Scores by Run\n'
        f'Run Spec: {run_spec_name}\n'
        'This often communicates sparse or zero-heavy metric distributions more clearly than histograms.',
        fontsize=16,
    )
    out_fpath = fig_dpath / f'core_metric_ecdfs_{stamp}.png'
    fig.savefig(out_fpath, dpi=180)
    plt.close(fig)
    return out_fpath


def _single_run_core_stat_index(run_path: str) -> dict[str, Any]:
    ana = HelmRunAnalysis(HelmRun.coerce(run_path))
    idx = ana.stat_index(drop_zero_count=True, require_mean=True)
    return {k: v for k, v in idx.items() if v.metric_class == 'core'}


def _write_three_run_runlevel_table(
    out_dpath: Path,
    stamp: str,
    kwdagger_a_run: str,
    kwdagger_b_run: str,
    official_run: str,
) -> tuple[Path, Path]:
    idx_a = _single_run_core_stat_index(kwdagger_a_run)
    idx_b = _single_run_core_stat_index(kwdagger_b_run)
    idx_o = _single_run_core_stat_index(official_run)
    keys = sorted(set(idx_a) & set(idx_b) & set(idx_o))
    rows = []
    for key in keys:
        a = idx_a[key]
        b = idx_b[key]
        o = idx_o[key]
        rows.append({
            'stat_key': key,
            'metric': a.metric,
            'kwdagger_a': a.mean,
            'kwdagger_b': b.mean,
            'official': o.mean,
            'delta_official_vs_kwdagger_a': None if a.mean is None or o.mean is None else abs(o.mean - a.mean),
            'delta_kwdagger_a_vs_kwdagger_b': None if a.mean is None or b.mean is None else abs(a.mean - b.mean),
        })
    table = pd.DataFrame(rows)
    csv_fpath = out_dpath / f'core_runlevel_table_{stamp}.csv'
    md_fpath = out_dpath / f'core_runlevel_table_{stamp}.md'
    table.to_csv(csv_fpath, index=False)
    md_fpath.write_text(table.to_markdown(index=False) + '\n')
    return csv_fpath, md_fpath


def _strip_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_private(v)
            for k, v in obj.items()
            if not str(k).startswith('_')
        }
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def _write_text(report: dict[str, Any], out_fpath: Path) -> None:
    left, right = report['pairs']
    lines = []
    lines.append('Core Metric Report')
    lines.append('')
    lines.append(f"generated_utc: {report['generated_utc']}")
    lines.append(f"left_label: {left['label']}")
    lines.append(f"right_label: {right['label']}")
    lines.append('')
    lines.append('core_metrics:')
    for metric in left['core_metrics']:
        lines.append(f'  - {metric}')
    lines.append('')
    for pair in report['pairs']:
        lines.append(f"pair: {pair['label']}")
        lines.append(f"  diagnosis: {pair['diagnosis'].get('label')}")
        lines.append(f"  primary_reason_names: {pair['diagnosis'].get('primary_reason_names')}")
        lines.append(f"  run_level_n: {pair['run_level']['n_rows']}")
        lines.append(f"  instance_level_n: {pair['instance_level']['n_rows']}")
        lines.append(f"  run_level_quantiles: {json.dumps(pair['run_level']['overall_quantiles']['abs_delta'])}")
        lines.append(f"  instance_level_quantiles: {json.dumps(pair['instance_level']['overall_quantiles']['abs_delta'])}")
        lines.append('  by_metric:')
        for row in pair['instance_level']['by_metric']:
            lines.append(
                f"    - metric={row['metric']} count={row['count']} "
                f"p50={row['abs_delta']['p50']} p90={row['abs_delta']['p90']} "
                f"p95={row['abs_delta']['p95']} p99={row['abs_delta']['p99']} "
                f"max={row['abs_delta']['max']}"
            )
        lines.append('  agreement_vs_abs_tol:')
        for row in pair['instance_level']['agreement_vs_abs_tol']:
            lines.append(
                f"    - abs_tol={row['abs_tol']} agree_ratio={row['agree_ratio']}"
            )
        lines.append('')
    out_fpath.write_text('\n'.join(lines) + '\n')


def _find_curve_value(rows: list[dict[str, Any]], abs_tol: float) -> float | None:
    for row in rows:
        if float(row.get('abs_tol', float('nan'))) == float(abs_tol):
            return row.get('agree_ratio')
    return None


def _write_management_summary(report: dict[str, Any], out_fpath: Path) -> None:
    left, right = report['pairs']
    lines = []
    lines.append('Core Metric Executive Summary')
    lines.append('')
    lines.append(f"generated_utc: {report['generated_utc']}")
    lines.append(f"run_spec_name: {report['run_spec_name']}")
    lines.append(f"core_metrics: {', '.join(left.get('core_metrics', []))}")
    lines.append('')
    lines.append('metric_descriptions:')
    for metric in left.get('core_metrics', []):
        desc = _metric_descriptor(metric)
        lines.append(
            f"  - {metric}: {desc['kind']}; {desc['range']}; {desc['direction']}"
        )
    lines.append('')
    lines.append(f"{left['label']}:")
    lines.append(f"  diagnosis: {left['diagnosis'].get('label')}")
    lines.append(f"  run-level N: {left['run_level']['n_rows']}")
    lines.append(f"  instance-level N: {left['instance_level']['n_rows']}")
    lines.append(
        f"  instance agreement at abs_tol=0.0: {_find_curve_value(left['instance_level']['agreement_vs_abs_tol'], 0.0)}"
    )
    lines.append(
        f"  run-level abs delta max: {left['run_level']['overall_quantiles']['abs_delta']['max']}"
    )
    lines.append(
        f"  instance-level abs delta max: {left['instance_level']['overall_quantiles']['abs_delta']['max']}"
    )
    lines.append('')
    lines.append(f"{right['label']}:")
    lines.append(f"  diagnosis: {right['diagnosis'].get('label')}")
    lines.append(f"  run-level N: {right['run_level']['n_rows']}")
    lines.append(f"  instance-level N: {right['instance_level']['n_rows']}")
    for tol in [0.0, 1e-3, 1e-2, 1e-1, 2.5e-1, 5e-1, 1.0]:
        lines.append(
            f"  instance agreement at abs_tol={tol}: "
            f"{_find_curve_value(right['instance_level']['agreement_vs_abs_tol'], tol)}"
        )
    lines.append(
        f"  run-level abs delta p90/max: "
        f"{right['run_level']['overall_quantiles']['abs_delta']['p90']} / "
        f"{right['run_level']['overall_quantiles']['abs_delta']['max']}"
    )
    lines.append(
        f"  instance-level abs delta p99/max: "
        f"{right['instance_level']['overall_quantiles']['abs_delta']['p99']} / "
        f"{right['instance_level']['overall_quantiles']['abs_delta']['max']}"
    )
    out_fpath.write_text('\n'.join(lines) + '\n')


def _write_latest_alias(src: Path, latest_root: Path, latest_name: str) -> Path:
    latest_fpath = latest_root / latest_name
    if latest_fpath.exists() or latest_fpath.is_symlink():
        latest_fpath.unlink()
    rel_src = os.path.relpath(src, start=latest_fpath.parent)
    os.symlink(rel_src, latest_fpath)
    return latest_fpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--left-run-a', required=True)
    parser.add_argument('--left-run-b', required=True)
    parser.add_argument('--left-label', required=True)
    parser.add_argument('--right-run-a', required=True)
    parser.add_argument('--right-run-b', required=True)
    parser.add_argument('--right-label', required=True)
    parser.add_argument('--report-dpath', required=True)
    args = parser.parse_args()

    thresholds = [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1.0]
    report_dpath = Path(args.report_dpath).expanduser().resolve()
    report_dpath.mkdir(parents=True, exist_ok=True)
    stamp = datetime_mod.datetime.now(datetime_mod.UTC).strftime('%Y%m%dT%H%M%SZ')
    history_dpath = report_dpath / '.history' / stamp[:8]
    history_dpath.mkdir(parents=True, exist_ok=True)
    run_spec_name = _infer_run_spec_name(args.left_run_a, args.left_run_b, args.right_run_a)

    left = _build_pair(args.left_run_a, args.left_run_b, args.left_label, thresholds)
    right = _build_pair(args.right_run_a, args.right_run_b, args.right_label, thresholds)
    report = {
        'generated_utc': stamp,
        'run_spec_name': run_spec_name,
        'thresholds': thresholds,
        'pairs': [left, right],
    }

    json_fpath = history_dpath / f'core_metric_report_{stamp}.json'
    txt_fpath = history_dpath / f'core_metric_report_{stamp}.txt'
    mgmt_fpath = history_dpath / f'core_metric_management_summary_{stamp}.txt'
    fig_fpath = history_dpath / f'core_metric_report_{stamp}.png'
    dist_fig_fpath = _plot_metric_distributions(history_dpath, stamp, left, right, run_spec_name)
    three_run_dist_fpath = _plot_three_run_metric_distributions(
        history_dpath,
        stamp,
        args.left_run_a,
        args.left_run_b,
        args.right_run_a,
        run_spec_name,
    )
    overlay_dist_fpath = _plot_overlay_metric_distributions(
        history_dpath,
        stamp,
        args.left_run_a,
        args.left_run_b,
        args.right_run_a,
        run_spec_name,
    )
    ecdf_fig_fpath = _plot_overlay_metric_ecdfs(
        history_dpath,
        stamp,
        args.left_run_a,
        args.left_run_b,
        args.right_run_a,
        run_spec_name,
    )
    runlevel_csv_fpath, runlevel_md_fpath = _write_three_run_runlevel_table(
        history_dpath,
        stamp,
        args.left_run_a,
        args.left_run_b,
        args.right_run_a,
    )

    report = kwutil.Json.ensure_serializable(_strip_private(report))
    json_fpath.write_text(json.dumps(report, indent=2))
    _write_text(report, txt_fpath)
    _write_management_summary(report, mgmt_fpath)

    sns.set_theme(style='whitegrid', context='talk')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    _plot_quantiles(
        axes[0, 0],
        left,
        right,
        'run_level',
        f"Run-Level Core Metric Difference Quantiles\nRun Spec: {run_spec_name}\nN={left['run_level']['n_rows']} vs N={right['run_level']['n_rows']}"
    )
    _plot_quantiles(
        axes[0, 1],
        left,
        right,
        'instance_level',
        f"Instance-Level Core Metric Difference Quantiles\nRun Spec: {run_spec_name}\nN={left['instance_level']['n_rows']} vs N={right['instance_level']['n_rows']}"
    )
    _plot_distribution(axes[1, 0], left, right, 'run_level')
    axes[1, 0].set_title(
        f"Run-Level Core Metric Agreement vs Tolerance\n"
        f"Run Spec: {run_spec_name}\n"
        f"{left['label']} N={left['run_level']['n_rows']}, {right['label']} N={right['run_level']['n_rows']}"
    )
    _plot_distribution(axes[1, 1], left, right, 'instance_level')
    axes[1, 1].set_title(
        f"Instance-Level Core Metric Agreement vs Tolerance\n"
        f"Run Spec: {run_spec_name}\n"
        f"{left['label']} N={left['instance_level']['n_rows']}, {right['label']} N={right['instance_level']['n_rows']}"
    )
    fig.suptitle(
        'Core Metric Agreement and Difference Summary\n'
        f'Run Spec: {run_spec_name}\n'
        'Comparing kwdagger repeatability against official-vs-kwdagger divergence.',
        fontsize=18,
    )
    fig.savefig(fig_fpath, dpi=180)
    plt.close(fig)

    latest_map = {
        json_fpath: 'core_metric_report.latest.json',
        txt_fpath: 'core_metric_report.latest.txt',
        mgmt_fpath: 'core_metric_management_summary.latest.txt',
        fig_fpath: 'core_metric_report.latest.png',
        dist_fig_fpath: 'core_metric_distributions.latest.png',
        three_run_dist_fpath: 'core_metric_three_run_distributions.latest.png',
        overlay_dist_fpath: 'core_metric_overlay_distributions.latest.png',
        ecdf_fig_fpath: 'core_metric_ecdfs.latest.png',
        runlevel_csv_fpath: 'core_runlevel_table.latest.csv',
        runlevel_md_fpath: 'core_runlevel_table.latest.md',
    }
    for src, latest_name in latest_map.items():
        _write_latest_alias(src, report_dpath, latest_name)

    print(f'Wrote core metric report: {json_fpath}')
    print(f'Wrote core metric text: {txt_fpath}')
    print(f'Wrote core metric management summary: {mgmt_fpath}')
    print(f'Wrote core metric plot: {fig_fpath}')
    print(f'Wrote core metric distributions: {dist_fig_fpath}')
    print(f'Wrote core metric three-run distributions: {three_run_dist_fpath}')
    print(f'Wrote core metric overlay distributions: {overlay_dist_fpath}')
    print(f'Wrote core metric ecdfs: {ecdf_fig_fpath}')
    print(f'Wrote core run-level table csv: {runlevel_csv_fpath}')
    print(f'Wrote core run-level table md: {runlevel_md_fpath}')


if __name__ == '__main__':
    main()
