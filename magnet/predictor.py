import os
from glob import glob
import random
from typing import List

from helm.benchmark.metrics.statistic import Stat

from magnet.loaders import (
    load_run_spec,
    load_all_run_specs_as_dataframe,
    load_all_scenario_states_as_dataframe,
    load_all_stats_as_dataframe)


class Predictor:
    def __init__(self,
                 num_example_runs=3,
                 num_eval_samples=20,
                 random_seed=1):
        self.num_example_runs = num_example_runs
        self.num_eval_samples = num_eval_samples
        self.random_seed = random_seed

    def run_spec_filter(self, run_spec):
        # To be overridden; likely only want to use this filter *OR*
        # `run_spec_dataframe_filter` depending on if you want to
        # filter on the HELM data model of a run_spec, or the
        # dataframe row, both will be applied
        return True

    def run_spec_dataframe_filter(self, row):
        # To be overridden; likely only want to use this filter *OR*
        # `run_spec_filter` depending on if you want to
        # filter on the HELM data model of a run_spec, or the
        # dataframe row, both will be applied
        return True

    def run_spec_dataframe_fields(self):
        # The "name" field is required, and will be added if not
        # specified
        return [
            "name",
            "adapter_spec.model",
            "adapter_spec",
            "metric_specs",
            "data_augmenter_spec",
            "groups",
        ]

    def scenario_state_dataframe_fields(self):
        return [
            "instance.input.text",
            "instance.id",
            "instance.split",
            "train_trial_index",
            "result.completions.0.text",
            "result.completions",
            "instance",
            "request",
            "result",
        ]

    def stats_dataframe_fields(self):
        return [
            "name.name",
            "name.split",
            "name",
            "count",
            "sum",
            "sum_squared",
            "min",
            "max",
            "mean",
            "variance",
            "stddev",
        ]

    def predict(self,
                train_run_specs_df,
                train_scenario_states_df,
                train_stats_df,
                eval_run_specs_df,
                eval_scenario_states_df) -> List[Stat]:
        raise NotImplementedError

    def __call__(self, root_dir, suite):
        if self.random_seed is not None:
            random.seed(self.random_seed)

        selected_run_specs = []
        for run_spec_fp in glob(os.path.join(root_dir, 'runs', suite, '*', 'run_spec.json')):
            run_spec = load_run_spec(run_spec_fp)

            if self.run_spec_filter(run_spec):
                selected_run_specs.append(run_spec)

        selected_run_specs_df = load_all_run_specs_as_dataframe(
            suite,
            [r.name for r in selected_run_specs],
            self.run_spec_dataframe_fields(),
            root_dir=root_dir)

        selected_run_specs_df = selected_run_specs_df[selected_run_specs_df.apply(
            self.run_spec_dataframe_filter, axis=1)]

        selected_run_specs_names = list(selected_run_specs_df['name'])

        *train_runs, eval_run = random.sample(
            selected_run_specs_names, self.num_example_runs + 1)

        train_run_specs_df = selected_run_specs_df[
            selected_run_specs_df['name'].isin(train_runs)]

        eval_run_specs_df = selected_run_specs_df[
            selected_run_specs_df['name'] == eval_run]

        train_scenario_states_df = load_all_scenario_states_as_dataframe(
            suite,
            train_runs,
            self.scenario_state_dataframe_fields(),
            root_dir=root_dir)

        train_stats_df = load_all_stats_as_dataframe(
            suite,
            train_runs,
            self.stats_dataframe_fields(),
            root_dir=root_dir)

        _full_eval_scenario_states_df = load_all_scenario_states_as_dataframe(
            suite,
            [eval_run],
            self.scenario_state_dataframe_fields(),
            root_dir=root_dir)

        random_eval_indices = random.sample(
            range(len(_full_eval_scenario_states_df)), self.num_eval_samples)
        eval_scenario_states_df = _full_eval_scenario_states_df.iloc[random_eval_indices]

        predicted_stats = self.predict(train_run_specs_df,
                                       train_scenario_states_df,
                                       train_stats_df,
                                       eval_run_specs_df,
                                       eval_scenario_states_df)

        # TODO: Do something meaningful with the predictions
        print(predicted_stats)
