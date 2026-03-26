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
from magnet.backends.helm.cli.materialize_helm_run import discover_benchmark_output_dirs
from magnet.backends.helm.cli.materialize_helm_run import run_dir_matches_requested
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


def load_run_spec_json(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    fpath = run_dir / "run_spec.json"
    if not fpath.exists():
        return {}
    return json.loads(fpath.read_text())


def infer_benchmark_group(run_spec_name: str | None) -> str:
    text = (run_spec_name or "").strip()
    if not text:
        return "unknown"
    idxs = [i for i in [text.find(":"), text.find(",")] if i >= 0]
    if idxs:
        return text[: min(idxs)].strip()
    return text


def collect_historic_candidates(
    precomputed_root: str | Path,
    run_entry: str,
) -> list[dict[str, Any]]:
    candidates = []
    for bo in discover_benchmark_output_dirs([precomputed_root]):
        try:
            outputs = HelmOutputs.coerce(bo)
        except Exception:
            continue
        for suite in outputs.suites(pattern="*"):
            for run in suite.runs(pattern="*"):
                run_dir = Path(run.path)
                if not run_dir_matches_requested(run.name, run_entry):
                    continue
                run_spec = load_run_spec_json(run_dir)
                adapter_spec = run_spec.get("adapter_spec", {}) or {}
                metric_specs = run_spec.get("metric_specs", []) or []
                candidates.append(
                    {
                        "run_dir": run_dir,
                        "run_name": run.name,
                        "source_root": bo,
                        "helm_version": run_dir.parent.name,
                        "requested_max_eval_instances": adapter_spec.get(
                            "max_eval_instances", None
                        ),
                        "model_deployment": adapter_spec.get(
                            "model_deployment", None
                        ),
                        "metric_class_names": [
                            m.get("class_name", None) for m in metric_specs
                        ],
                    }
                )
    return candidates


def choose_historic_candidate(
    candidates: list[dict[str, Any]],
    desired_max_eval_instances: int | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not candidates:
        return None, {
            "candidate_count": 0,
            "exact_requested_max_eval_match": False,
            "candidate_requested_max_eval_instances": [],
        }
    exact_matches = []
    if desired_max_eval_instances is not None:
        exact_matches = [
            c
            for c in candidates
            if c.get("requested_max_eval_instances", None)
            == desired_max_eval_instances
        ]
    ranked_pool = exact_matches if exact_matches else candidates

    def sort_key(c: dict[str, Any]):
        req = c.get("requested_max_eval_instances", None)
        req_dist = (
            abs(req - desired_max_eval_instances)
            if req is not None and desired_max_eval_instances is not None
            else float("inf")
        )
        return (req_dist, str(c.get("helm_version", "")), str(c["run_dir"]))

    chosen = sorted(ranked_pool, key=sort_key)[0]
    info = {
        "candidate_count": len(candidates),
        "exact_requested_max_eval_match": bool(exact_matches),
        "candidate_requested_max_eval_instances": sorted(
            {c.get("requested_max_eval_instances", None) for c in candidates},
            key=lambda x: (x is None, x),
        ),
        "chosen_requested_max_eval_instances": chosen.get(
            "requested_max_eval_instances", None
        ),
    }
    return chosen, info


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
    primary_reason_counter = Counter()
    for row in rows:
        status = row.get("status", "unknown")
        status_counter[status] += 1
        if status != "compared":
            continue
        diag = row.get("diagnosis", {}) or {}
        diagnosis_counter[diag.get("label", "unknown")] += 1
        for reason_name in diag.get("primary_reason_names", []) or []:
            primary_reason_counter[reason_name] += 1
        for reason in diag.get("reasons", []) or []:
            name = reason.get("name", "unknown")
            reason_counter[name] += 1
    return {
        "n_rows": len(rows),
        "status_counts": dict(status_counter),
        "diagnosis_label_counts": dict(diagnosis_counter),
        "primary_reason_name_counts": dict(primary_reason_counter),
        "reason_counts": dict(reason_counter),
    }


def build_high_level_findings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    n_rows = len(rows)
    if n_rows == 0:
        return findings

    compared = sum(1 for row in rows if row.get("status") == "compared")
    if compared == n_rows:
        findings.append(
            {
                "label": "comparison_pipeline_working",
                "severity": "info",
                "summary": f"All {compared} requested runs were paired and compared successfully.",
            }
        )

    requested_max_mismatch = sum(
        1
        for row in rows
        if row.get("historic_requested_max_eval_instances", None)
        != row.get("kwdg_requested_max_eval_instances", None)
    )
    if requested_max_mismatch:
        findings.append(
            {
                "label": "requested_max_eval_mismatch",
                "severity": "high",
                "summary": (
                    f"{requested_max_mismatch}/{n_rows} cases use a historic run with "
                    "a different requested max_eval_instances value."
                ),
            }
        )

    no_exact_match = sum(
        1
        for row in rows
        if not row.get("historic_exact_requested_max_eval_match", False)
    )
    if no_exact_match:
        findings.append(
            {
                "label": "no_exact_historic_eval_size_match",
                "severity": "high",
                "summary": (
                    f"{no_exact_match}/{n_rows} cases did not have an exact historic match "
                    "for requested max_eval_instances in the public bundle."
                ),
            }
        )

    for reason_name, severity in [
        ("deployment_drift", "high"),
        ("execution_spec_drift", "high"),
        ("evaluation_spec_drift", "medium"),
        ("dataset_variant_drift", "medium"),
        ("dataset_instance_drift", "medium"),
        ("core_metric_drift", "medium"),
        ("completion_content_drift", "medium"),
    ]:
        count = sum(
            1
            for row in rows
            if any(
                reason.get("name") == reason_name
                for reason in ((row.get("diagnosis", {}) or {}).get("reasons", []) or [])
            )
        )
        if count:
            findings.append(
                {
                    "label": reason_name,
                    "severity": severity,
                    "summary": f"{reason_name} appears in {count}/{n_rows} compared cases.",
                }
            )
    return findings


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
        desired_max_eval_instances = manifest.get("max_eval_instances", None)
        candidates = collect_historic_candidates(
            precomputed_root=precomputed_root,
            run_entry=run_entry,
        )
        match, match_info = choose_historic_candidate(
            candidates, desired_max_eval_instances
        )
        row = {
            "run_spec_name": run_entry,
            "run_dir": None,
            "model": None,
            "benchmark_group": infer_benchmark_group(run_entry),
            "benchmark_name": "unknown",
            "suite_name": "unknown",
            "helm_version": "unknown",
            "requested_max_eval_instances": None,
            "model_deployment": None,
            "metric_class_names": [],
            "match_info": match_info,
        }
        if match is not None:
            parsed = parse_helm_run_dir(str(match["run_dir"]))
            row["run_dir"] = str(match["run_dir"])
            row["benchmark_name"] = parsed["helm_suite_name"]
            row["suite_name"] = parsed["helm_suite_name"]
            row["helm_version"] = str(match.get("helm_version", parsed["helm_version"]))
            row["requested_max_eval_instances"] = match.get(
                "requested_max_eval_instances", None
            )
            row["model_deployment"] = match.get("model_deployment", None)
            row["metric_class_names"] = match.get("metric_class_names", [])
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
    lines.append("")
    lines.append("primary_reason_name_counts:")
    for key, value in sorted(
        summary_report["aggregate"].get("primary_reason_name_counts", {}).items()
    ):
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("reason_counts:")
    for key, value in sorted(
        summary_report["aggregate"].get("reason_counts", {}).items()
    ):
        lines.append(f"  {key}: {value}")
    findings = summary_report.get("high_level_findings", []) or []
    if findings:
        lines.append("")
        lines.append("high_level_findings:")
        for item in findings:
            lines.append(
                f"  [{item.get('severity', 'info')}] {item.get('label')}: {item.get('summary')}"
            )
    out_fpath.write_text("\n".join(lines) + "\n")


def write_management_summary(
    summary_report: dict[str, Any], out_fpath: Path
) -> None:
    inputs = summary_report.get("inputs", {}) or {}
    findings = summary_report.get("high_level_findings", []) or []
    aggregate = summary_report.get("aggregate", {}) or {}
    status_counts = aggregate.get("status_counts", {}) or {}
    compared = status_counts.get("compared", 0)
    total = sum(status_counts.values())
    lines = []
    lines.append("Audit HELM Reproduction: Executive Summary")
    lines.append("")
    lines.append(
        f"Compared {inputs.get('n_manifest_run_entries', '?')} requested runs against "
        f"{inputs.get('n_historic_rows', '?')} historic matches and "
        f"{inputs.get('n_kwdg_rows', '?')} reproduced runs."
    )
    lines.append(f"{compared}/{total} runs completed comparison successfully.")
    lines.append("")
    lines.append("Key findings:")
    for item in findings:
        lines.append(f"- [{item.get('severity', 'info').upper()}] {item.get('summary')}")
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
    management_txt_fpath = report_dpath / f"management_summary_{stamp}.txt"

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
                "historic_requested_max_eval_instances": helm_row.get(
                    "requested_max_eval_instances", None
                ),
                "historic_model_deployment": helm_row.get(
                    "model_deployment", None
                ),
                "historic_metric_class_names": helm_row.get(
                    "metric_class_names", []
                ),
                "historic_match_info": helm_row.get("match_info", {}),
                "historic_exact_requested_max_eval_match": (
                    helm_row.get("match_info", {}) or {}
                ).get("exact_requested_max_eval_match", False),
                "kwdg_requested_max_eval_instances": None,
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
                    kwdg_run_spec = load_run_spec_json(kwdg_run.path)
                    case_row["kwdg_requested_max_eval_instances"] = (
                        kwdg_run_spec.get("adapter_spec", {}) or {}
                    ).get(
                        "max_eval_instances", None
                    )
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
        "report_management_txt": str(management_txt_fpath),
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
        "high_level_findings": build_high_level_findings(all_case_rows),
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
    write_management_summary(summary_report, management_txt_fpath)

    print(f"Wrote case report: {case_jsonl_fpath}")
    print(f"Wrote summary report: {summary_json_fpath}")
    print(f"Wrote summary text: {summary_txt_fpath}")
    print(f"Wrote management summary: {management_txt_fpath}")
    if sankey_artifacts.get("plotly_error", None):
        print(f"Sankey note: {sankey_artifacts['plotly_error']}")


if __name__ == "__main__":
    main()
