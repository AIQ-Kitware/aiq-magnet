"""
IO / discovery for HELM reproduction analysis.
"""
from __future__ import annotations

from typing import Any, Dict, List
import ubelt as ub
import kwutil

from magnet.helm_outputs import HelmOutputs

def load_public_helm_rows(run_details_yaml: str) -> List[Dict[str, Any]]:
    return kwutil.Yaml.load(run_details_yaml)

def discover_kwdagger_runs(results_root: str) -> List[Dict[str, Any]]:
    """
    Discover local completed runs based on DONE markers.
    Returns list of dicts: {dpath, run_spec_name, run}
    """
    finished_jobs = list(ub.Path(results_root).glob("*/DONE"))
    rows = []
    for fpath in finished_jobs:
        config = kwutil.Json.coerce(fpath.parent / "job_config.json")
        run_spec_name = config["helm.run_entry"]
        dpath = fpath.parent
        runs = HelmOutputs.coerce(dpath / "benchmark_output").suites()[0].runs()
        assert len(runs) == 1
        run = runs[0]
        rows.append({"dpath": dpath, "run_spec_name": run_spec_name, "run": run})
    return rows

def annotate_public_row_paths(helm_row: Dict[str, Any]):
    """
    Adds suite_name and benchmark_name extracted from run_dir.
    """
    run_dir = ub.Path(helm_row["run_dir"])
    suite_name = run_dir.parent.name
    benchmark_name = run_dir.parent.parent.parent.parent.name
    assert run_dir.parent.parent.parent.name == "benchmark_output"
    assert run_dir.parent.parent.name == "runs"
    helm_row["suite_name"] = suite_name
    helm_row["benchmark_name"] = benchmark_name
    return helm_row
