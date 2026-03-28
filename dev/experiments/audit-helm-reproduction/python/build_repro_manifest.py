from __future__ import annotations

import argparse
import fnmatch
import math
from pathlib import Path
from typing import Any

import kwutil

from common import dump_yaml, env_defaults, repo_run_details_fpath, repo_run_specs_fpath


VICUNA_NOCHAT_OVERRIDE = (
    "dev/experiments/audit-helm-reproduction/configs/debug/"
    "vicuna_no_chat_template.yaml"
)


def _load_run_specs(fpath: str | None) -> list[str]:
    path = Path(fpath) if fpath else repo_run_specs_fpath()
    data = kwutil.Yaml.load(path)
    if not isinstance(data, list):
        raise TypeError(f"run specs at {path} must decode to a list")
    run_specs = [str(x) for x in data]
    return list(dict.fromkeys(run_specs))


def _load_run_details(fpath: str | None) -> list[dict[str, Any]]:
    path = Path(fpath) if fpath else repo_run_details_fpath()
    if not path.exists():
        return []
    data = kwutil.Yaml.load(path)
    if not isinstance(data, list):
        raise TypeError(f"run details at {path} must decode to a list")
    rows = [row for row in data if isinstance(row, dict)]
    return rows


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(text, pat) for pat in patterns)


def _infer_benchmark(run_entry: str) -> str:
    left = run_entry.split(":", 1)[0]
    return left.split(",", 1)[0]


def _infer_model(run_entry: str) -> str | None:
    for part in run_entry.replace(":", ",").split(","):
        if part.startswith("model="):
            return part.split("=", 1)[1]
    return None


def _sort_key(run_entry: str) -> tuple[str, str, str]:
    model = _infer_model(run_entry) or ""
    benchmark = _infer_benchmark(run_entry)
    return (model, benchmark, run_entry)


def _filter_run_entries(
    run_entries: list[str],
    *,
    include_patterns: list[str],
    exclude_patterns: list[str],
    models: list[str],
    benchmarks: list[str],
) -> list[str]:
    filtered = []
    for run_entry in run_entries:
        if include_patterns and not _matches_any(run_entry, include_patterns):
            continue
        if exclude_patterns and _matches_any(run_entry, exclude_patterns):
            continue
        model = _infer_model(run_entry)
        benchmark = _infer_benchmark(run_entry)
        if models and model not in set(models):
            continue
        if benchmarks and benchmark not in set(benchmarks):
            continue
        filtered.append(run_entry)
    return filtered


def _shard_entries(
    run_entries: list[str],
    *,
    num_shards: int | None,
    shard_index: int | None,
) -> list[str]:
    if num_shards is None and shard_index is None:
        return run_entries
    if num_shards is None or shard_index is None:
        raise SystemExit("--num-shards and --shard-index must be provided together")
    if num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")
    return [entry for idx, entry in enumerate(run_entries) if idx % num_shards == shard_index]


def _choose_model_override(run_entries: list[str], force_nochat: bool) -> str | None:
    models = {_infer_model(entry) for entry in run_entries}
    needs_vicuna = "lmsys/vicuna-7b-v1.3" in models
    if force_nochat or needs_vicuna:
        return VICUNA_NOCHAT_OVERRIDE
    return None


def _detail_lut(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lut: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("run_spec_name")
        if isinstance(key, str) and key not in lut:
            lut[key] = row
    return lut


def _build_manifest(
    *,
    experiment_name: str,
    description: str,
    suite: str,
    run_entries: list[str],
    max_eval_instances: int,
    tmux_workers: int,
    devices: str,
    model_deployments_fpath: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "description": description,
        "run_entries": run_entries,
        "max_eval_instances": max_eval_instances,
        "suite": suite,
        "mode": "compute_if_missing",
        "materialize": "symlink",
        "backend": "tmux",
        "devices": devices,
        "tmux_workers": tmux_workers,
        "local_path": "prod_env",
        "precomputed_root": None,
        "require_per_instance_stats": True,
        "model_deployments_fpath": model_deployments_fpath,
        "enable_huggingface_models": [],
        "enable_local_huggingface_models": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-output", default=None)
    parser.add_argument("--run-specs-fpath", default=None)
    parser.add_argument("--run-details-fpath", default=None)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--description", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--tmux-workers", type=int, default=None)
    parser.add_argument("--max-eval-instances", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--sort", default="model_benchmark", choices=["model_benchmark", "input"])
    parser.add_argument("--force-vicuna-nochat", action="store_true")
    parser.add_argument("--include-pattern", action="append", default=[])
    parser.add_argument("--exclude-pattern", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--benchmark", action="append", default=[])
    args = parser.parse_args()

    defaults = env_defaults()
    run_entries = _load_run_specs(args.run_specs_fpath)
    run_details = _load_run_details(args.run_details_fpath)
    detail_lut = _detail_lut(run_details)

    run_entries = _filter_run_entries(
        run_entries,
        include_patterns=args.include_pattern,
        exclude_patterns=args.exclude_pattern,
        models=args.model,
        benchmarks=args.benchmark,
    )
    if args.sort == "model_benchmark":
        run_entries = sorted(run_entries, key=_sort_key)
    run_entries = _shard_entries(
        run_entries,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if args.limit is not None:
        run_entries = run_entries[: args.limit]
    if not run_entries:
        raise SystemExit("No run entries matched the requested filters")

    max_eval_instances = (
        args.max_eval_instances
        if args.max_eval_instances is not None
        else 1000
    )
    if args.single_gpu:
        devices = args.devices if args.devices is not None else "0"
        tmux_workers = args.tmux_workers if args.tmux_workers is not None else 1
    else:
        devices = args.devices if args.devices is not None else "0,1"
        tmux_workers = (
            args.tmux_workers
            if args.tmux_workers is not None
            else int(defaults["AUDIT_DEFAULT_TMUX_WORKERS"])
        )

    model_override = _choose_model_override(run_entries, args.force_vicuna_nochat)
    description = args.description or (
        f"Historic reproducibility batch with {len(run_entries)} run entries"
    )
    manifest = _build_manifest(
        experiment_name=args.experiment_name,
        description=description,
        suite=args.suite,
        run_entries=run_entries,
        max_eval_instances=max_eval_instances,
        tmux_workers=tmux_workers,
        devices=devices,
        model_deployments_fpath=model_override,
    )

    out_fpath = Path(args.output)
    out_fpath.parent.mkdir(parents=True, exist_ok=True)
    out_fpath.write_text(dump_yaml(manifest))

    selection_rows = []
    for idx, run_entry in enumerate(run_entries):
        row = {
            "index": idx,
            "run_entry": run_entry,
            "benchmark": _infer_benchmark(run_entry),
            "model": _infer_model(run_entry),
            "detail": detail_lut.get(run_entry),
        }
        selection_rows.append(row)

    selection = {
        "experiment_name": args.experiment_name,
        "suite": args.suite,
        "manifest_fpath": str(out_fpath),
        "selection_count": len(run_entries),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "limit": args.limit,
        "devices": devices,
        "tmux_workers": tmux_workers,
        "max_eval_instances": max_eval_instances,
        "model_deployments_fpath": model_override,
        "include_patterns": args.include_pattern,
        "exclude_patterns": args.exclude_pattern,
        "models": args.model,
        "benchmarks": args.benchmark,
        "entries": selection_rows,
    }
    selection_fpath = (
        Path(args.selection_output)
        if args.selection_output
        else out_fpath.with_suffix(out_fpath.suffix + ".selection.yaml")
    )
    selection_fpath.write_text(dump_yaml(selection))
    print(out_fpath)
    print(selection_fpath)


if __name__ == "__main__":
    main()
