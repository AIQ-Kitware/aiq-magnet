from __future__ import annotations

import argparse
import datetime as datetime_mod
import os
from pathlib import Path

from magnet.backends.helm.helm_outputs import HelmRun
from magnet.backends.helm.helm_run_diff import HelmRunDiff


def _safe_unlink(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def _write_latest_alias(src: Path, latest_root: Path, latest_name: str) -> Path:
    latest_fpath = latest_root / latest_name
    _safe_unlink(latest_fpath)
    rel_src = os.path.relpath(src, start=latest_fpath.parent)
    os.symlink(rel_src, latest_fpath)
    return latest_fpath


def _slugify(text: str) -> str:
    return (
        str(text)
        .replace('/', '-')
        .replace(':', '-')
        .replace(',', '-')
        .replace('=', '-')
        .replace('@', '-')
        .replace(' ', '-')
    )


def _infer_run_spec_name(*run_paths: str) -> str:
    names = [Path(p).name for p in run_paths if p]
    names = [n for n in names if n]
    if not names:
        return 'unknown_run_spec'
    unique = sorted(set(names))
    if len(unique) == 1:
        return unique[0]
    return unique[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-a', required=True)
    parser.add_argument('--run-b', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--report-dpath', required=True)
    parser.add_argument('--top-n', type=int, default=8)
    parser.add_argument('--show-details', type=int, default=6)
    parser.add_argument('--level', type=int, default=30)
    args = parser.parse_args()

    report_dpath = Path(args.report_dpath).expanduser().resolve()
    report_dpath.mkdir(parents=True, exist_ok=True)
    stamp = datetime_mod.datetime.now(datetime_mod.UTC).strftime('%Y%m%dT%H%M%SZ')
    history_dpath = report_dpath / '.history' / stamp[:8]
    history_dpath.mkdir(parents=True, exist_ok=True)

    diff = HelmRunDiff(
        HelmRun.coerce(args.run_a),
        HelmRun.coerce(args.run_b),
        a_name=f'{args.label}:A',
        b_name=f'{args.label}:B',
    )
    run_spec_name = _infer_run_spec_name(args.run_a, args.run_b)
    lines: list[str] = []
    lines.append(f'Instance Sample Inspection')
    lines.append(f'label: {args.label}')
    lines.append(f'run_spec_name: {run_spec_name}')
    lines.append(f'run_a: {Path(args.run_a).expanduser().resolve()}')
    lines.append(f'run_b: {Path(args.run_b).expanduser().resolve()}')
    lines.append('')
    diff.summarize_instances(
        level=args.level,
        top_n=args.top_n,
        show_details=args.show_details,
        writer=lines.append,
    )
    out_fpath = history_dpath / f'instance_samples_{_slugify(args.label)}_{stamp}.txt'
    out_fpath.write_text('\n'.join(lines) + '\n')
    latest_name = f'instance_samples_{_slugify(args.label)}.latest.txt'
    latest_fpath = _write_latest_alias(out_fpath, report_dpath, latest_name)
    print(f'Wrote instance sample report: {out_fpath}')
    print(f'Updated latest link: {latest_fpath}')


if __name__ == '__main__':
    main()
