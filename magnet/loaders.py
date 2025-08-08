import os
import json
from typing import Dict, Any, List

import dacite
import pandas as pd
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.run_spec import RunSpec


def load_run_spec(run_spec_file_path):
    with open(run_spec_file_path) as f:
        json_run_spec = json.load(f)
        run_spec = dacite.from_dict(RunSpec, json_run_spec)

    return run_spec


def load_scenario_state(suite, run_spec, root_dir="benchmark_output"):
    state_file_path = os.path.join(
        root_dir, "runs", suite, run_spec, "scenario_state.json"
    )
    with open(state_file_path) as f:
        json_state: Dict[str, Any] = json.load(f)
        scenario_state = dacite.from_dict(ScenarioState, json_state)

    return scenario_state


def load_all_scenario_states_as_dataframe(
    suite, run_specs, root_dir="benchmark_output"
):
    scenario_states_records = []
    for run_spec in run_specs:
        state: ScenarioState = load_scenario_state(suite, run_spec, root_dir=root_dir)
        model = state.adapter_spec.model.replace(os.path.sep, "_")

        for request_state in state.request_states:
            input = request_state.instance.input.text
            output = request_state.result.completions[0].text

            scenario_states_records.append((run_spec, model, input, output))

    scenario_states_df = pd.DataFrame(
        scenario_states_records, columns=["run_spec", "model", "input", "output"]
    )

    return scenario_states_df


def load_stats(suite, run_spec, root_dir="benchmark_output"):
    stats_file_path = os.path.join(root_dir, "runs", suite, run_spec, "stats.json")
    with open(stats_file_path) as f:
        json_stats: List[Dict[str, Any]] = json.load(f)
        stats = [dacite.from_dict(Stat, json_stat) for json_stat in json_stats]

    return stats


def load_all_stats_as_dataframe(suite, run_specs, root_dir="benchmark_output"):
    stats_records = []
    for run_spec in run_specs:
        stats = load_stats(suite, run_spec, root_dir=root_dir)

        for stat in stats:
            if stat.name.perturbation is None:
                stats_records.append(
                    (
                        run_spec,
                        stat.name.name,
                        None,
                        None,
                        None,
                        None,
                        None,  # perturbation name, robustness, fairness, computed_on, seed
                        stat.count,
                        stat.sum,
                        stat.sum_squared,
                        stat.min,
                        stat.max,
                        stat.mean,
                        stat.variance,
                        stat.stddev,
                    )
                )
            else:
                stats_records.append(
                    (
                        run_spec,
                        stat.name.name,
                        stat.name.perturbation.name,
                        stat.name.perturbation.robustness,
                        stat.name.perturbation.fairness,
                        stat.name.perturbation.computed_on,
                        stat.name.perturbation.seed,
                        stat.count,
                        stat.sum,
                        stat.sum_squared,
                        stat.min,
                        stat.max,
                        stat.mean,
                        stat.variance,
                        stat.stddev,
                    )
                )

    stats_df = pd.DataFrame(
        stats_records,
        columns=[
            "run_spec",
            "name",
            "perturbation_name",
            "perturbation_robustness",
            "perturbation_fairness",
            "perturbation_computed_on",
            "perturbation_seed",
            "count",
            "sum",
            "sum_squared",
            "min",
            "max",
            "mean",
            "variance",
            "stddev",
        ],
    )

    return stats_df
