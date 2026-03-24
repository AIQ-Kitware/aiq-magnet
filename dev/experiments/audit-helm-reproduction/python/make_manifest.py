from __future__ import annotations

import argparse
import os
from pathlib import Path

import kwutil

from common import dump_yaml, env_defaults, repo_run_specs_fpath


SMOKE_RUN_ENTRIES = [
    "mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model=eleutherai/pythia-6.9b,data_augmentation=canonical",
    "boolq:model=eleutherai/pythia-6.9b,data_augmentation=canonical",
    "narrative_qa:model=eleutherai/pythia-6.9b,data_augmentation=canonical",
    "mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model=lmsys/vicuna-7b-v1.3,data_augmentation=canonical",
    "boolq:model=lmsys/vicuna-7b-v1.3,data_augmentation=canonical",
    "narrative_qa:model=lmsys/vicuna-7b-v1.3,data_augmentation=canonical",
]


def _validate_entries_exist(run_entries: list[str]) -> list[str]:
    fpath = repo_run_specs_fpath()
    all_run_specs = set(kwutil.Yaml.load(fpath))
    missing = [entry for entry in run_entries if entry not in all_run_specs]
    return missing


def build_smoke_manifest(args: argparse.Namespace) -> dict:
    defaults = env_defaults()
    max_eval_instances = (
        args.max_eval_instances
        if args.max_eval_instances is not None
        else int(defaults["AUDIT_DEFAULT_MAX_EVAL_INSTANCES"])
    )
    tmux_workers = (
        args.tmux_workers
        if args.tmux_workers is not None
        else int(defaults["AUDIT_DEFAULT_TMUX_WORKERS"])
    )
    devices = args.devices if args.devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")

    missing = _validate_entries_exist(SMOKE_RUN_ENTRIES)
    if missing:
        raise RuntimeError(
            "Smoke manifest entries were not found in run_specs.yaml: "
            + kwutil.Json.dumps(missing)
        )

    return {
        "schema_version": 1,
        "experiment_name": args.experiment_name,
        "description": "Small smoke-test batch for HELM reproduction auditing.",
        "run_entries": SMOKE_RUN_ENTRIES,
        "max_eval_instances": max_eval_instances,
        "suite": args.suite,
        "mode": "compute_if_missing",
        "materialize": "symlink",
        "backend": "tmux",
        "devices": devices,
        "tmux_workers": tmux_workers,
        "local_path": "prod_env",
        "precomputed_root": None,
        "require_per_instance_stats": True,
        "model_deployments_fpath": None,
        "enable_huggingface_models": [],
        "enable_local_huggingface_models": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-type", default="smoke", choices=["smoke"]
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment-name", default="audit-smoke")
    parser.add_argument("--suite", default="audit-smoke")
    parser.add_argument("--max-eval-instances", type=int, default=None)
    parser.add_argument("--tmux-workers", type=int, default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    if args.manifest_type != "smoke":
        raise NotImplementedError(args.manifest_type)

    manifest = build_smoke_manifest(args)
    out_fpath = Path(args.output)
    out_fpath.parent.mkdir(parents=True, exist_ok=True)
    out_fpath.write_text(dump_yaml(manifest))
    print(out_fpath)


if __name__ == "__main__":
    main()
