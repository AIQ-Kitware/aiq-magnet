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
    """
    Convert nested JSON to DataFrame using dot-separated paths.

    Example:
        >>> from magnet.loaders import *
        >>> json_data = {'nested': {
        >>> 'type': 'measure',
        >>> 'results': [
        >>>     {'score': 1},
        >>>     {'score': 2},
        >>>     {'score': 3},
        >>> ]}}
        >>> paths = ['nested.type', 'nested.results', 'nested.results.0.score']
        >>> df = json_to_dataframe(json_data, paths)
        >>> print(df.to_string())
          nested.type                              nested.results  nested.results.0.score
        0     measure  [{'score': 1}, {'score': 2}, {'score': 3}]                       1
    """

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
    Returns:
        List[helm.benchmark.metrics.statistic.Stat]

    Example:
        >>> from magnet.loaders import *
        >>> import magnet
        >>> outputs = magnet.helm_outputs.HelmOutputs.demo()
        >>> dpath = magnet.demo.ensure_helm_demo_outputs()
        >>> root_dir = outputs.root_dir
        >>> suite = outputs.list_suites()[0]
        >>> run_spec = outputs.list_run_specs(suite=suite)[0]
        >>> stats = load_stats(suite, run_spec, root_dir)
        >>> assert len(stats) > 80
    """
    stats_file_path = os.path.join(root_dir, 'runs', suite, run_spec, "stats.json")
    with open(stats_file_path) as f:
        json_stats: List[Dict[str, Any]] = json.load(f)
        stats = [dacite.from_dict(Stat, json_stat) for json_stat in json_stats]

    return stats


def load_scenario_state(suite, run_spec, root_dir="benchmark_output"):
    """
    Returns:
        helm.benchmark.adaptation.scenario_state.ScenarioState

    Example:
        >>> from magnet.loaders import *
        >>> import magnet
        >>> outputs = magnet.helm_outputs.HelmOutputs.demo()
        >>> root_dir = outputs.root_dir
        >>> suite = outputs.list_suites()[0]
        >>> run_spec = outputs.list_run_specs(suite=suite)[0]
        >>> scenario_state = load_scenario_state(suite, run_spec, root_dir)
        >>> assert len(scenario_state.instances) >= 1
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
    """
    Returns:
        pandas.DataFrame

    Example:
        >>> from magnet.loaders import *
        >>> import magnet
        >>> outputs = magnet.helm_outputs.HelmOutputs.demo()
        >>> root_dir = outputs.root_dir
        >>> suite = outputs.list_suites()[0]
        >>> run_specs = [outputs.list_run_specs(suite=suite)[0]]
        >>> run_spec_fields = [
        >>>     "name",
        >>>     "adapter_spec.model",
        >>>     "adapter_spec",
        >>>     "metric_specs",
        >>>     "data_augmenter_spec",
        >>>     "groups",
        >>> ]
        >>> run_specs_df = load_all_run_specs_as_dataframe(suite, run_specs, run_spec_fields, root_dir)
        >>> assert len(run_specs_df.columns) == len(run_spec_fields)
        >>> assert len(run_specs_df) == 1
        >>> assert not any([x is None for x in run_specs_df.values.ravel()])
    """
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
    Returns:
        pandas.DataFrame

    Example:
        >>> from magnet.loaders import *
        >>> import magnet
        >>> outputs = magnet.helm_outputs.HelmOutputs.demo()
        >>> root_dir = outputs.root_dir
        >>> suite = outputs.list_suites()[0]
        >>> run_specs = [outputs.list_run_specs(suite=suite)[0]]
        >>> scenario_state_fields = [
        >>>     "instance.input.text",
        >>>     "instance.id",
        >>>     "instance.split",
        >>>     "train_trial_index",
        >>>     "result.completions.0.text",
        >>>     "result.completions",
        >>>     "instance",
        >>>     "request",
        >>>     "result",
        >>> ]
        >>> scenario_states_df = load_all_scenario_states_as_dataframe(suite, run_specs, scenario_state_fields, root_dir)
        >>> assert len(scenario_states_df.columns) == len(scenario_state_fields) + 1
        >>> assert len(scenario_states_df) == 1
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
    Returns:
        pandas.DataFrame

    Example:
        >>> from magnet.loaders import *
        >>> import magnet
        >>> outputs = magnet.helm_outputs.HelmOutputs.demo()
        >>> root_dir = outputs.root_dir
        >>> suite = outputs.list_suites()[0]
        >>> run_specs = [outputs.list_run_specs(suite=suite)[0]]
        >>> stats_fields = [
        >>>     "name.name", "name.split", "name",
        >>>     "count", "sum", "sum_squared", "min",
        >>>     "max", "mean", "variance", "stddev",
        >>> ]
        >>> stats = load_all_stats_as_dataframe(suite, run_specs, stats_fields, root_dir)
        >>> assert len(stats) > 80
        >>> assert len(stats.columns) == len(stats_fields) + 1
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
