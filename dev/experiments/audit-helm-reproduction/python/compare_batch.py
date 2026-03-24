from __future__ import annotations

import argparse
import datetime as datetime_mod
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import kwutil
import ubelt as ub

from common import (
    env_defaults,
    experiment_report_dpath,
    experiment_result_dpath,
    load_manifest,
)
from magnet.backends.helm.cli.materialize_helm_run import find_best_precomputed_run
from magnet.backends.helm.helm_outputs import HelmOutputs, HelmRun
from magnet.backends.helm.helm_run_diff import HelmRunDiff


def parse_helm_run_dir(run_dir: str) -> dict[str, str]:
    p = ub.Path(run_dir)
    parts = list(p.parts)
    out = {
        "helm_suite_name": "unknown",
        "helm_version": "unknown",
        "run_leaf": p.name,
    }
    try:
        idx = parts.index("benchmark_output")
    except ValueError:
        idx = -1
    if idx >= 1:
        out["helm_suite_name"] = str(parts[idx - 1])
    if idx >= 0 and (idx + 2) < len(parts):
        out["helm_version"] = str(parts[idx + 2])
    else:
        out["helm_version"] = str(p.parent.name)
    return out


def infer_benchmark_group(run_spec_name: str | None) -> str:
    text = (run_spec_name or "").strip()
    if not text:
        return "unknown"
    idxs = [i for i in [text.find(":"), text.find(",")] if i >= 0]
    if idxs:
        return text[: min(idxs)].strip()
    return text


def load_kwdg_rows(results_dpath: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    finished_jobs = sorted(
        fpath
        for fpath in results_dpath.rglob("DONE")
        if (fpath.parent / "job_config.json").exists()
    )
    rows = []
    for fpath in ub.ProgIter(finished_jobs, desc="load kwdg runs"):
        dpath = fpath.parent
        try:
            config = kwutil.Json.load(dpath / "job_config.json")
            run_spec_name = config.get("helm.run_entry", None)
            if run_spec_name is None:
                continue
            suites = HelmOutputs.coerce(dpath / "benchmark_output").suites()
            runs = []
            for suite in suites:
                runs.extend(list(suite.runs()))
            if len(runs) != 1:
                continue
            run = HelmRun.coerce(runs[0])
            rows.append(
                {
                    "dpath": str(dpath),
                    "run_spec_name": run_spec_name,
                    "run": run,
                }
            )
        except Exception:
            continue

    lut = {}
    for row in rows:
        lut[row["run_spec_name"]] = row
    return rows, lut


def aggregate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counter = Counter()
    diagnosis_counter = Counter()
    reason_counter = Counter()
    for row in rows:
        status = row.get("status", "unknown")
        status_counter[status] += 1
        if status != "compared":
            continue
        diag = row.get("diagnosis", {}) or {}
        diagnosis_counter[diag.get("label", "unknown")] += 1
        for reason in diag.get("reasons", []) or []:
            name = reason.get("name", "unknown")
            reason_counter[name] += 1
    return {
        "n_rows": len(rows),
        "status_counts": dict(status_counter),
        "diagnosis_label_counts": dict(diagnosis_counter),
        "reason_counts": dict(reason_counter),
    }


def maybe_write_sankey_report(
    case_rows: list[dict[str, Any]], report_dpath: Path, stamp: str
) -> dict[str, Any]:
    source_fpath = (
        Path(__file__).resolve().parents[3]
        / "oneoff"
        / "diagnose_reproducibility.py"
    )
    if not source_fpath.exists():
        return {"plotly_error": "sankey helper script not found"}

    spec = importlib.util.spec_from_file_location(
        "audit_diag_helper", source_fpath
    )
    if spec is None or spec.loader is None:
        return {"plotly_error": "unable to import sankey helper"}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_sankey_report(
        case_rows, report_dpath=ub.Path(report_dpath), stamp=stamp
    )


def build_historic_rows(
    manifest: dict[str, Any], precomputed_root: str
) -> list[dict[str, Any]]:
    rows = []
    for run_entry in manifest["run_entries"]:
        match = find_best_precomputed_run(
            precomputed_root=precomputed_root,
            requested_desc=run_entry,
            max_eval_instances=manifest.get("max_eval_instances", None),
            require_per_instance_stats=manifest.get(
                "require_per_instance_stats", True
            ),
        )
        row = {
            "run_spec_name": run_entry,
            "run_dir": None,
            "model": None,
            "benchmark_group": infer_benchmark_group(run_entry),
            "benchmark_name": "unknown",
            "suite_name": "unknown",
            "helm_version": "unknown",
        }
        if match is not None:
            parsed = parse_helm_run_dir(str(match.run_dir))
            row["run_dir"] = str(match.run_dir)
            row["benchmark_name"] = parsed["helm_suite_name"]
            row["suite_name"] = parsed["helm_suite_name"]
            row["helm_version"] = parsed["helm_version"]
            if "model=" in run_entry:
                model_text = run_entry.split("model=", 1)[1].split(",", 1)[0]
                row["model"] = model_text
        rows.append(row)
    return rows


def write_summary_text(
    summary_report: dict[str, Any], out_fpath: Path
) -> None:
    inputs = summary_report.get("inputs", {}) or {}
    lines = []
    lines.append(f"generated_utc: {summary_report['generated_utc']}")
    lines.append(f"case_jsonl: {summary_report['report_case_jsonl']}")
    lines.append(f"summary_json: {summary_report['report_summary_json']}")
    if inputs:
        lines.append("")
        lines.append("inputs:")
        for key, value in sorted(inputs.items()):
            lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("status_counts:")
    for key, value in sorted(
        summary_report["aggregate"]["status_counts"].items()
    ):
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("diagnosis_label_counts:")
    for key, value in sorted(
        summary_report["aggregate"]["diagnosis_label_counts"].items()
    ):
        lines.append(f"  {key}: {value}")
    out_fpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-dpath", default=None)
    parser.add_argument("--report-dpath", default=None)
    parser.add_argument("--precomputed-root", default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    defaults = env_defaults()
    results_dpath = (
        Path(args.results_dpath).expanduser().resolve()
        if args.results_dpath
        else experiment_result_dpath(manifest)
    )
    report_dpath = (
        Path(args.report_dpath).expanduser().resolve()
        if args.report_dpath
        else experiment_report_dpath(manifest)
    )
    precomputed_root = args.precomputed_root or defaults["HELM_PRECOMPUTED_ROOT"]
    report_dpath.mkdir(parents=True, exist_ok=True)

    historic_rows = build_historic_rows(manifest, precomputed_root)
    kwdg_rows, kwdg_lut = load_kwdg_rows(results_dpath)

    stamp = datetime_mod.datetime.now(datetime_mod.UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    case_jsonl_fpath = report_dpath / f"compare_cases_{stamp}.jsonl"
    summary_json_fpath = report_dpath / f"compare_summary_{stamp}.json"
    summary_txt_fpath = report_dpath / f"compare_summary_{stamp}.txt"

    all_case_rows = []
    with case_jsonl_fpath.open("w", encoding="utf8") as file:
        for idx, helm_row in enumerate(historic_rows, start=1):
            run_spec_name = helm_row["run_spec_name"]
            kwrow = kwdg_lut.get(run_spec_name, None)
            case_row = {
                "index": idx,
                "run_spec_name": run_spec_name,
                "benchmark_name": helm_row.get("benchmark_name", "unknown"),
                "benchmark_group": helm_row.get("benchmark_group", "unknown"),
                "suite_name": helm_row.get("suite_name", "unknown"),
                "model_name": helm_row.get("model", None),
                "helm_version": helm_row.get("helm_version", None),
                "helm_run_dir": helm_row["run_dir"],
                "kwdg_run_dir": None if kwrow is None else kwrow["dpath"],
            }

            if helm_row["run_dir"] is None:
                case_row.update(
                    {
                        "status": "missing_historic_match",
                        "diagnosis": {
                            "label": "missing_historic_match",
                            "primary_priority": 0,
                            "primary_reason_names": [
                                "missing_historic_match"
                            ],
                            "reasons": [
                                {
                                    "name": "missing_historic_match",
                                    "priority": 0,
                                    "details": {},
                                }
                            ],
                        },
                    }
                )
            elif kwrow is None:
                case_row.update(
                    {
                        "status": "missing_kwdg_match",
                        "diagnosis": {
                            "label": "missing_kwdg_match",
                            "primary_priority": 0,
                            "primary_reason_names": ["missing_kwdg_match"],
                            "reasons": [
                                {
                                    "name": "missing_kwdg_match",
                                    "priority": 0,
                                    "details": {},
                                }
                            ],
                        },
                    }
                )
            else:
                try:
                    helm_run = HelmRun.coerce(helm_row["run_dir"])
                    kwdg_run = kwrow["run"]
                    rd = HelmRunDiff(
                        run_a=helm_run,
                        run_b=kwdg_run,
                        a_name="HELM",
                        b_name="KWDG",
                    )
                    summary = rd.summary_dict(level=20)
                    diag = summary.get("diagnosis", {}) or {}
                    case_row.update(
                        {
                            "status": "compared",
                            "diagnosis": diag,
                            "run_spec_semantic": summary.get(
                                "run_spec_semantic", None
                            ),
                            "scenario_semantic": summary.get(
                                "scenario_semantic", None
                            ),
                            "dataset_overlap": summary.get(
                                "dataset_overlap", None
                            ),
                            "stats_coverage_by_name": summary.get(
                                "stats_coverage_by_name", None
                            ),
                            "stats_coverage_by_name_count": summary.get(
                                "stats_coverage_by_name_count", None
                            ),
                            "value_agreement": summary.get(
                                "value_agreement", None
                            ),
                            "instance_value_agreement": summary.get(
                                "instance_value_agreement", None
                            ),
                        }
                    )
                except Exception as ex:
                    case_row.update(
                        {
                            "status": "error",
                            "error": repr(ex),
                            "diagnosis": {
                                "label": "comparison_error",
                                "primary_priority": 0,
                                "primary_reason_names": ["comparison_error"],
                                "reasons": [
                                    {
                                        "name": "comparison_error",
                                        "priority": 0,
                                        "details": {"error": repr(ex)},
                                    }
                                ],
                            },
                        }
                    )

            case_row = kwutil.Json.ensure_serializable(case_row)
            file.write(json.dumps(case_row, ensure_ascii=False) + "\n")
            file.flush()
            all_case_rows.append(case_row)

    summary_report = {
        "report_case_jsonl": str(case_jsonl_fpath),
        "report_summary_json": str(summary_json_fpath),
        "report_summary_txt": str(summary_txt_fpath),
        "generated_utc": stamp,
        "inputs": {
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "kwdg_results_dpath": str(results_dpath),
            "precomputed_root": str(precomputed_root),
            "n_manifest_run_entries": len(manifest["run_entries"]),
            "n_kwdg_rows": len(kwdg_rows),
            "n_historic_rows": len(historic_rows),
        },
        "aggregate": aggregate_report(all_case_rows),
    }
    try:
        sankey_artifacts = maybe_write_sankey_report(
            all_case_rows, report_dpath, stamp
        )
    except Exception as ex:
        sankey_artifacts = {
            "plotly_error": f"failed to build sankey report: {ex!r}"
        }
    summary_report["artifacts"] = sankey_artifacts
    summary_report = kwutil.Json.ensure_serializable(summary_report)
    summary_json_fpath.write_text(
        json.dumps(summary_report, indent=2, ensure_ascii=False)
    )
    write_summary_text(summary_report, summary_txt_fpath)

    print(f"Wrote case report: {case_jsonl_fpath}")
    print(f"Wrote summary report: {summary_json_fpath}")
    print(f"Wrote summary text: {summary_txt_fpath}")
    if sankey_artifacts.get("plotly_error", None):
        print(f"Sankey note: {sankey_artifacts['plotly_error']}")


if __name__ == "__main__":
    main()
