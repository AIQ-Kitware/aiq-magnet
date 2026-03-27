from __future__ import annotations

import argparse
import datetime as datetime_mod
import json
from pathlib import Path
from typing import Any

import kwutil

from magnet.backends.helm.helm_outputs import HelmRun
from magnet.backends.helm.helm_run_diff import HelmRunDiff


def _pair_report(run_a: str, run_b: str, label: str) -> dict[str, Any]:
    diff = HelmRunDiff(
        run_a=HelmRun.coerce(run_a),
        run_b=HelmRun.coerce(run_b),
        a_name=f'{label}:A',
        b_name=f'{label}:B',
    )
    return {
        'label': label,
        'inputs': {
            'run_a': str(Path(run_a).expanduser().resolve()),
            'run_b': str(Path(run_b).expanduser().resolve()),
        },
        'run_level': {
            'agreement': diff._value_agreement_summary(),
            'distance': diff.value_distance_profile(),
        },
        'instance_level': {
            'agreement': diff.instance_agreement_profile(),
            'distance': diff.instance_distance_profile(),
        },
        'diagnosis': diff.summary_dict(level=20).get('diagnosis', {}),
    }


def _instance_metric_rows(pair: dict[str, Any]) -> dict[tuple[str, str | None], dict[str, Any]]:
    distance_rows = pair['instance_level']['distance'].get('by_metric', [])
    agreement_rows = pair['instance_level']['agreement'].get('by_metric', [])
    agree_lut = {
        (row.get('metric_class'), row.get('metric')): row
        for row in agreement_rows
    }
    rows = {}
    for row in distance_rows:
        key = (row.get('metric_class'), row.get('metric'))
        rows[key] = {
            'metric_class': row.get('metric_class'),
            'metric': row.get('metric'),
            'count': row.get('summary', {}).get('count'),
            'agree_ratio': agree_lut.get(key, {}).get('agree_ratio'),
            'mismatched': agree_lut.get(key, {}).get('mismatched'),
            'abs_delta': row.get('summary', {}).get('abs_delta', {}),
            'rel_delta': row.get('summary', {}).get('rel_delta', {}),
        }
    return rows


def _run_class_rows(pair: dict[str, Any]) -> dict[str, dict[str, Any]]:
    agree = pair['run_level']['agreement'].get('by_class', {})
    dist = pair['run_level']['distance'].get('by_class', {})
    rows = {}
    for cls in sorted(set(agree) | set(dist)):
        rows[cls] = {
            'metric_class': cls,
            'comparable': (agree.get(cls) or {}).get('comparable'),
            'agree_ratio': (agree.get(cls) or {}).get('agree_ratio'),
            'mismatched': (agree.get(cls) or {}).get('mismatched'),
            'abs_delta': (dist.get(cls) or {}).get('abs_delta', {}),
            'rel_delta': (dist.get(cls) or {}).get('rel_delta', {}),
            'count': (dist.get(cls) or {}).get('count'),
        }
    return rows


def _fmt(x: Any) -> str:
    if x is None:
        return 'NA'
    if isinstance(x, float):
        return f'{x:.6g}'
    return str(x)


def _build_text(report: dict[str, Any]) -> str:
    left = report['pairs'][0]
    right = report['pairs'][1]
    left_run = _run_class_rows(left)
    right_run = _run_class_rows(right)
    left_inst = _instance_metric_rows(left)
    right_inst = _instance_metric_rows(right)

    lines = []
    lines.append('Metric Quantiles Comparison')
    lines.append('')
    lines.append(f"generated_utc: {report['generated_utc']}")
    lines.append(f"left_label: {left['label']}")
    lines.append(f"right_label: {right['label']}")
    lines.append('')
    lines.append('pair_diagnoses:')
    lines.append(f"  {left['label']}: {left['diagnosis'].get('label')}")
    lines.append(f"  {right['label']}: {right['diagnosis'].get('label')}")
    lines.append('')
    lines.append('run_level_by_class:')
    for cls in sorted(set(left_run) | set(right_run)):
        lrow = left_run.get(cls, {})
        rrow = right_run.get(cls, {})
        lines.append(f'  class={cls}')
        lines.append(
            f"    {left['label']}: agree={_fmt(lrow.get('agree_ratio'))} count={_fmt(lrow.get('count'))} "
            f"abs_p50={_fmt((lrow.get('abs_delta') or {}).get('p50'))} "
            f"abs_p90={_fmt((lrow.get('abs_delta') or {}).get('p90'))} "
            f"abs_p99={_fmt((lrow.get('abs_delta') or {}).get('p99'))} "
            f"abs_max={_fmt((lrow.get('abs_delta') or {}).get('max'))}"
        )
        lines.append(
            f"    {right['label']}: agree={_fmt(rrow.get('agree_ratio'))} count={_fmt(rrow.get('count'))} "
            f"abs_p50={_fmt((rrow.get('abs_delta') or {}).get('p50'))} "
            f"abs_p90={_fmt((rrow.get('abs_delta') or {}).get('p90'))} "
            f"abs_p99={_fmt((rrow.get('abs_delta') or {}).get('p99'))} "
            f"abs_max={_fmt((rrow.get('abs_delta') or {}).get('max'))}"
        )
    lines.append('')
    lines.append('instance_level_by_metric:')
    for key in sorted(set(left_inst) | set(right_inst), key=lambda k: (str(k[0]), str(k[1]))):
        lrow = left_inst.get(key, {})
        rrow = right_inst.get(key, {})
        lines.append(f'  metric_class={key[0]} metric={key[1]}')
        lines.append(
            f"    {left['label']}: agree={_fmt(lrow.get('agree_ratio'))} count={_fmt(lrow.get('count'))} "
            f"abs_p50={_fmt((lrow.get('abs_delta') or {}).get('p50'))} "
            f"abs_p90={_fmt((lrow.get('abs_delta') or {}).get('p90'))} "
            f"abs_p99={_fmt((lrow.get('abs_delta') or {}).get('p99'))} "
            f"abs_max={_fmt((lrow.get('abs_delta') or {}).get('max'))}"
        )
        lines.append(
            f"    {right['label']}: agree={_fmt(rrow.get('agree_ratio'))} count={_fmt(rrow.get('count'))} "
            f"abs_p50={_fmt((rrow.get('abs_delta') or {}).get('p50'))} "
            f"abs_p90={_fmt((rrow.get('abs_delta') or {}).get('p90'))} "
            f"abs_p99={_fmt((rrow.get('abs_delta') or {}).get('p99'))} "
            f"abs_max={_fmt((rrow.get('abs_delta') or {}).get('max'))}"
        )
    return '\n'.join(lines) + '\n'


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

    report_dpath = Path(args.report_dpath).expanduser().resolve()
    report_dpath.mkdir(parents=True, exist_ok=True)
    stamp = datetime_mod.datetime.now(datetime_mod.UTC).strftime('%Y%m%dT%H%M%SZ')

    report = {
        'generated_utc': stamp,
        'pairs': [
            _pair_report(args.left_run_a, args.left_run_b, args.left_label),
            _pair_report(args.right_run_a, args.right_run_b, args.right_label),
        ],
    }
    report = kwutil.Json.ensure_serializable(report)

    json_fpath = report_dpath / f'metric_quantiles_{stamp}.json'
    txt_fpath = report_dpath / f'metric_quantiles_{stamp}.txt'
    json_fpath.write_text(json.dumps(report, indent=2))
    txt_fpath.write_text(_build_text(report))
    print(f'Wrote metric quantiles report: {json_fpath}')
    print(f'Wrote metric quantiles text: {txt_fpath}')


if __name__ == '__main__':
    main()
