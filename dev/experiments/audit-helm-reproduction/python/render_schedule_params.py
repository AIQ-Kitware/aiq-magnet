from __future__ import annotations

import argparse
import json

from common import dump_yaml, experiment_result_dpath, load_manifest


def build_schedule_params(manifest: dict) -> dict:
    matrix = {
        "helm.run_entry": list(manifest["run_entries"]),
        "helm.max_eval_instances": [manifest["max_eval_instances"]],
        "helm.precomputed_root": [manifest.get("precomputed_root", None)],
        "helm.suite": [manifest.get("suite", "audit-smoke")],
        "helm.require_per_instance_stats": [
            manifest.get("require_per_instance_stats", True)
        ],
        "helm.mode": [manifest.get("mode", "compute_if_missing")],
        "helm.materialize": [manifest.get("materialize", "symlink")],
        "helm.local_path": [manifest.get("local_path", "prod_env")],
    }
    model_deployments_fpath = manifest.get("model_deployments_fpath", None)
    if model_deployments_fpath is not None:
        matrix["helm.model_deployments_fpath"] = [model_deployments_fpath]
    enable_hf = manifest.get("enable_huggingface_models", [])
    if enable_hf:
        matrix["helm.enable_huggingface_models"] = [json.dumps(enable_hf)]
    enable_local_hf = manifest.get("enable_local_huggingface_models", [])
    if enable_local_hf:
        matrix["helm.enable_local_huggingface_models"] = [json.dumps(enable_local_hf)]
    return {
        "pipeline": "magnet.backends.helm.pipeline.helm_single_run_pipeline()",
        "matrix": matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "params",
            "experiment_name",
            "result_dpath",
            "backend",
            "tmux_workers",
            "devices",
        ],
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)

    if args.mode == "params":
        print(dump_yaml(build_schedule_params(manifest)), end="")
    elif args.mode == "experiment_name":
        print(manifest["experiment_name"])
    elif args.mode == "result_dpath":
        print(experiment_result_dpath(manifest))
    elif args.mode == "backend":
        print(manifest.get("backend", "tmux"))
    elif args.mode == "tmux_workers":
        print(manifest.get("tmux_workers", 2))
    elif args.mode == "devices":
        print(manifest.get("devices", "0,1"))
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
