from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import kwutil


def env_defaults() -> dict[str, str]:
    return {
        "AIQ_MAGNET_ROOT": os.environ.get(
            "AIQ_MAGNET_ROOT", "/home/joncrall/code/aiq-magnet"
        ),
        "AIQ_PYTHON": os.environ.get("AIQ_PYTHON", "python"),
        "HELM_PRECOMPUTED_ROOT": os.environ.get(
            "HELM_PRECOMPUTED_ROOT", "/data/crfm-helm-public"
        ),
        "AUDIT_RESULTS_ROOT": os.environ.get(
            "AUDIT_RESULTS_ROOT", "/data/crfm-helm-audit"
        ),
        "AUDIT_DEFAULT_MAX_EVAL_INSTANCES": os.environ.get(
            "AUDIT_DEFAULT_MAX_EVAL_INSTANCES", "100"
        ),
        "AUDIT_DEFAULT_TMUX_WORKERS": os.environ.get(
            "AUDIT_DEFAULT_TMUX_WORKERS", "2"
        ),
    }


def audit_root() -> Path:
    return Path(__file__).resolve().parent.parent


def aiq_root() -> Path:
    return Path(env_defaults()["AIQ_MAGNET_ROOT"]).expanduser().resolve()


def repo_run_specs_fpath() -> Path:
    return aiq_root() / "run_specs.yaml"


def repo_run_details_fpath() -> Path:
    return aiq_root() / "run_details.yaml"


def default_report_root() -> Path:
    return audit_root() / "reports"


def load_manifest(manifest_fpath: str | os.PathLike[str]) -> dict[str, Any]:
    data = kwutil.Yaml.load(Path(manifest_fpath))
    if not isinstance(data, dict):
        raise TypeError("Manifest must decode to a dictionary")
    return data


def dump_yaml(data: Any) -> str:
    return kwutil.Yaml.dumps(data)


def experiment_result_dpath(manifest: dict[str, Any]) -> Path:
    root = Path(env_defaults()["AUDIT_RESULTS_ROOT"]).expanduser().resolve()
    return root / str(manifest["experiment_name"])


def experiment_report_dpath(manifest: dict[str, Any]) -> Path:
    return default_report_root() / str(manifest["experiment_name"])
