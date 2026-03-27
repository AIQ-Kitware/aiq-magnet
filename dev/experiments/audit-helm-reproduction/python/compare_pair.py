from __future__ import annotations

import argparse
import datetime as datetime_mod
import json
from pathlib import Path
from typing import Any

import kwutil

from magnet.backends.helm.helm_outputs import HelmRun
from magnet.backends.helm.helm_run_diff import HelmRunDiff


def load_yaml_or_default(text: str | None, default: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if text is None:
        return default
    data = kwutil.Yaml.coerce(text)
    if not isinstance(data, list):
        raise TypeError('Tolerance config must decode to a list of dictionaries')
    return data


def default_tolerances() -> list[dict[str, Any]]:
    return [
        {'name': 'strict', 'abs_tol': 0.0, 'rel_tol': 0.0},
        {'name': 'tiny', 'abs_tol': 1e-12, 'rel_tol': 1e-6},
        {'name': 'small', 'abs_tol': 1e-9, 'rel_tol': 1e-4},
        {'name': 'medium', 'abs_tol': 1e-6, 'rel_tol': 1e-3},
        {'name': 'loose', 'abs_tol': 1e-3, 'rel_tol': 1e-2},
        {'name': 'xloose', 'abs_tol': 1e-2, 'rel_tol': 1e-1},
        {'name': 'xxloose', 'abs_tol': 1e-1, 'rel_tol': 1.0},
        {'name': 'extreme', 'abs_tol': 1.0, 'rel_tol': 10.0},
    ]


def validate_run_dir(run_dpath: Path) -> None:
    required_files = [
        'run_spec.json',
        'scenario_state.json',
        'stats.json',
        'per_instance_stats.json',
    ]
    missing_files = [name for name in required_files if not (run_dpath / name).exists()]
    if missing_files:
        missing_text = ', '.join(missing_files)
        raise SystemExit(
            f'Run artifacts are incomplete for {run_dpath}. '
            f'Missing required files: {missing_text}'
        )


def summarize_tolerance_hits(sweep: dict[str, Any]) -> dict[str, Any]:
    out = {'run_level': [], 'instance_level': []}
    for level_key, target in [('run_level', 'overall'), ('instance_level', 'means')]:
        for row in sweep.get(level_key, []):
            summary = row.get('summary', {}) or {}
            if level_key == 'run_level':
                agree = ((summary.get('overall', {}) or {}).get('agree_ratio', None))
            else:
                agree = ((summary.get('means', {}) or {}).get('agree_ratio', None))
            out[level_key].append({
                'name': row.get('name'),
                'abs_tol': row.get('abs_tol'),
                'rel_tol': row.get('rel_tol'),
                'agree_ratio': agree,
            })
    return out


def write_text_report(report: dict[str, Any], out_fpath: Path) -> None:
    strict = report.get('strict_summary', {}) or {}
    diag = strict.get('diagnosis', {}) or {}
    run_dist = report.get('distance_summary', {}).get('run_level', {}) or {}
    inst_dist = report.get('distance_summary', {}).get('instance_level', {}) or {}
    sweep_hits = report.get('tolerance_highlights', {}) or {}

    lines = []
    lines.append('Audit Pair Comparison')
    lines.append('')
    lines.append(f"generated_utc: {report.get('generated_utc')}")
    lines.append(f"run_a: {report.get('inputs', {}).get('run_a')}")
    lines.append(f"run_b: {report.get('inputs', {}).get('run_b')}")
    lines.append('')
    lines.append(f"diagnosis_label: {diag.get('label')}")
    lines.append(f"primary_reason_names: {diag.get('primary_reason_names')}")
    lines.append('')

    lines.append('strict_agreement:')
    overall = (strict.get('value_agreement', {}) or {}).get('overall', {}) or {}
    lines.append(f"  run_level_agree_ratio: {overall.get('agree_ratio')}")
    means = (strict.get('instance_value_agreement', {}) or {}).get('means', {}) or {}
    lines.append(f"  instance_level_agree_ratio: {means.get('agree_ratio')}")
    lines.append('')

    lines.append('distance_summary:')
    lines.append(f"  run_level_count: {(run_dist.get('overall', {}) or {}).get('count')}")
    lines.append(f"  run_level_abs_p50: {((run_dist.get('overall', {}) or {}).get('abs_delta', {}) or {}).get('p50')}")
    lines.append(f"  run_level_abs_p90: {((run_dist.get('overall', {}) or {}).get('abs_delta', {}) or {}).get('p90')}")
    lines.append(f"  run_level_abs_max: {((run_dist.get('overall', {}) or {}).get('abs_delta', {}) or {}).get('max')}")
    lines.append(f"  instance_level_count: {(inst_dist.get('overall', {}) or {}).get('count')}")
    lines.append(f"  instance_level_abs_p50: {((inst_dist.get('overall', {}) or {}).get('abs_delta', {}) or {}).get('p50')}")
    lines.append(f"  instance_level_abs_p90: {((inst_dist.get('overall', {}) or {}).get('abs_delta', {}) or {}).get('p90')}")
    lines.append(f"  instance_level_abs_max: {((inst_dist.get('overall', {}) or {}).get('abs_delta', {}) or {}).get('max')}")
    lines.append('')

    lines.append('tolerance_sweep_run_level:')
    for row in sweep_hits.get('run_level', []):
        lines.append(
            f"  {row.get('name')}: abs_tol={row.get('abs_tol')} rel_tol={row.get('rel_tol')} agree_ratio={row.get('agree_ratio')}"
        )
    lines.append('')
    lines.append('tolerance_sweep_instance_level:')
    for row in sweep_hits.get('instance_level', []):
        lines.append(
            f"  {row.get('name')}: abs_tol={row.get('abs_tol')} rel_tol={row.get('rel_tol')} agree_ratio={row.get('agree_ratio')}"
        )
    out_fpath.write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-a', required=True)
    parser.add_argument('--run-b', required=True)
    parser.add_argument('--label-a', default='A')
    parser.add_argument('--label-b', default='B')
    parser.add_argument('--report-dpath', required=True)
    parser.add_argument('--run-tolerances-yaml', default=None)
    parser.add_argument('--instance-tolerances-yaml', default=None)
    args = parser.parse_args()

    report_dpath = Path(args.report_dpath).expanduser().resolve()
    report_dpath.mkdir(parents=True, exist_ok=True)
    stamp = datetime_mod.datetime.now(datetime_mod.UTC).strftime('%Y%m%dT%H%M%SZ')

    run_a_dpath = Path(args.run_a).expanduser().resolve()
    run_b_dpath = Path(args.run_b).expanduser().resolve()
    validate_run_dir(run_a_dpath)
    validate_run_dir(run_b_dpath)

    run_a = HelmRun.coerce(run_a_dpath)
    run_b = HelmRun.coerce(run_b_dpath)
    diff = HelmRunDiff(run_a=run_a, run_b=run_b, a_name=args.label_a, b_name=args.label_b)

    run_tolerances = load_yaml_or_default(args.run_tolerances_yaml, default_tolerances())
    instance_tolerances = load_yaml_or_default(args.instance_tolerances_yaml, default_tolerances())

    strict_summary = diff.summary_dict(level=20)
    distance_summary = {
        'run_level': diff.value_distance_profile(),
        'instance_level': diff.instance_distance_profile(),
    }
    tolerance_sweep = diff.tolerance_sweep_summary(
        run_tolerances=run_tolerances,
        instance_tolerances=instance_tolerances,
    )
    report = {
        'generated_utc': stamp,
        'inputs': {
            'run_a': str(run_a_dpath),
            'run_b': str(run_b_dpath),
            'label_a': args.label_a,
            'label_b': args.label_b,
        },
        'strict_summary': strict_summary,
        'distance_summary': distance_summary,
        'tolerance_sweep': tolerance_sweep,
        'tolerance_highlights': summarize_tolerance_hits(tolerance_sweep),
    }

    json_fpath = report_dpath / f'pair_report_{stamp}.json'
    txt_fpath = report_dpath / f'pair_report_{stamp}.txt'
    report = kwutil.Json.ensure_serializable(report)
    json_fpath.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_text_report(report, txt_fpath)
    print(f'Wrote pair report: {json_fpath}')
    print(f'Wrote pair text: {txt_fpath}')


if __name__ == '__main__':
    main()
