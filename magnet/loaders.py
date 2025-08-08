import os
import json
from typing import Dict, Any, List
import copy

import dacite
import pandas as pd
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.run_spec import RunSpec
import ubelt as ub


def json_to_dataframe(json_data, paths):
    """Convert nested JSON to DataFrame using dot-separated paths."""

    # Handle single dict case
    if isinstance(json_data, dict):
        json_data = [json_data]

    result_data = {path: [] for path in paths}

    for item in json_data:
        walker = ub.IndexableWalker(item)

        for path in paths:
            # Convert dot-separated path to list
            path_list = []
            for p in path.split('.'):
                try:
                    # We want to convert list indicies in the path to
                    # ints if possible; this does break if we have
                    # dictionaries with integer keys
                    path_list.append(int(p))
                except ValueError:
                    path_list.append(p)

            try:
                value = walker[path_list]
                result_data[path].append(value)
            except (KeyError, IndexError):
                result_data[path].append(None)

    return pd.DataFrame(result_data)


def load_run_spec(run_spec_file_path):
    with open(run_spec_file_path) as f:
        json_run_spec = json.load(f)
        run_spec = dacite.from_dict(RunSpec, json_run_spec)

    return run_spec


def load_stats(suite, run_spec, root_dir="benchmark_output"):
    """
    Example:
        >>> from magnet.loaders import *  # NOQA
        >>> import magnet
        >>> dpath = magnet.demo.ensure_helm_demo_outputs()
        >>> root_dir = (dpath / 'benchmark_output')
        >>> suite = 'latest'
        >>> run_spec = 'mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2'
        >>> stats = load_stats(suite, run_spec, root_dir)
    """
    stats_file_path = os.path.join(root_dir, 'runs', suite, run_spec, "stats.json")
    with open(stats_file_path) as f:
        json_stats: List[Dict[str, Any]] = json.load(f)
        stats = [dacite.from_dict(Stat, json_stat) for json_stat in json_stats]

    return stats


def load_scenario_state(suite, run_spec, root_dir="benchmark_output"):
    """
    Example:
        >>> from magnet.loaders import *  # NOQA
        >>> import magnet
        >>> dpath = magnet.demo.ensure_helm_demo_outputs()
        >>> root_dir = (dpath / 'benchmark_output')
        >>> suite = 'latest'
        >>> run_spec = 'mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2'
        >>> scenario_state = load_scenario_state(suite, run_spec, root_dir)
    """
    state_file_path = os.path.join(root_dir, 'runs', suite, run_spec, "scenario_state.json")
    with open(state_file_path) as f:
        json_state: Dict[str, Any] = json.load(f)
        scenario_state = dacite.from_dict(ScenarioState, json_state)

    return scenario_state


def load_all_run_specs_as_dataframe(suite,
                                    run_specs,
                                    run_spec_fields,
                                    root_dir="benchmark_output"):
    run_spec_fields = copy.deepcopy(run_spec_fields)
    if 'name' not in run_spec_fields:
        run_spec_fields.insert(0, 'name')

    run_specs_dfs = []
    for run_spec in run_specs:
        state_file_path = os.path.join(root_dir, 'runs', suite, run_spec, "run_spec.json")
        with open(state_file_path) as f:
            run_spec_data = json.load(f)

        run_spec_df = json_to_dataframe(run_spec_data,
                                        run_spec_fields)

        run_specs_dfs.append(run_spec_df)

    return pd.concat(run_specs_dfs)


def load_all_scenario_states_as_dataframe(suite,
                                          run_specs,
                                          scenario_state_fields,
                                          root_dir="benchmark_output"):
    """
    Example:
        >>> from magnet.loaders import *  # NOQA
        >>> import magnet
        >>> dpath = magnet.demo.ensure_helm_demo_outputs()
        >>> root_dir = (dpath / 'benchmark_output')
        >>> suite = 'latest'
        >>> run_specs = ['mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2']
        >>> scenario_states_df = load_all_scenario_states_as_dataframe(suite, run_specs, root_dir)
    """
    scenario_states_dfs = []
    for run_spec in run_specs:
        state_file_path = os.path.join(root_dir, 'runs', suite, run_spec, "scenario_state.json")
        with open(state_file_path) as f:
            scenario_state_data = json.load(f)

        scenario_state_df = json_to_dataframe(scenario_state_data['request_states'],
                                              scenario_state_fields)

        # Add the run_spec name as a "foreign-key" to our scenario_state dataframe
        run_spec_name_column_df = pd.DataFrame(
            {'run_spec.name': [run_spec] * len(scenario_state_df)})
        scenario_state_df = pd.concat([run_spec_name_column_df, scenario_state_df], axis=1)

        scenario_states_dfs.append(scenario_state_df)

    return pd.concat(scenario_states_dfs)

def load_all_stats_as_dataframe(suite,
                                run_specs,
                                stats_fields,
                                root_dir="benchmark_output"):
    """
    Example:
        >>> from magnet.loaders import *  # NOQA
        >>> import magnet
        >>> dpath = magnet.demo.ensure_helm_demo_outputs()
        >>> root_dir = (dpath / 'benchmark_output')
        >>> suite = 'latest'
        >>> run_specs = ['mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2']
        >>> stats = load_all_stats_as_dataframe(suite, run_specs, root_dir)
    """
    stats_dfs = []
    for run_spec in run_specs:
        stats_file_path = os.path.join(root_dir, 'runs', suite, run_spec, "stats.json")
        with open(stats_file_path) as f:
            stats_data = json.load(f)

        stats_df = json_to_dataframe(stats_data,
                                     stats_fields)

        # Add the run_spec name as a "foreign-key" to our stats dataframe
        run_spec_name_column_df = pd.DataFrame(
            {'run_spec.name': [run_spec] * len(stats_df)})
        stats_df = pd.concat([run_spec_name_column_df, stats_df], axis=1)

        stats_dfs.append(stats_df)

    return pd.concat(stats_dfs)
