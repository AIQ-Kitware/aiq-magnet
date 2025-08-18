import os
from glob import glob
import random
from typing import List

from helm.benchmark.metrics.statistic import Stat
import ubelt as ub
import pandas as pd

from magnet.loaders import (
    load_run_spec,
    load_all_run_specs_as_dataframe,
    load_all_scenario_states_as_dataframe,
    load_all_stats_as_dataframe)
from magnet import HelmOutputs


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

        outputs = HelmOutputs(ub.Path(root_dir))
        suites_output = outputs.suites(suite)
        assert len(suites_output) == 1
        suite_output = suites_output[0]

        selected_runs = []
        for run in suite_output.runs():
            if self.run_spec_filter(run.raw.run_spec()):
                selected_runs.append(run)

        selected_run_specs_df = pd.concat([r.run_spec() for r in selected_runs])

        selected_run_specs_df = selected_run_specs_df[selected_run_specs_df.apply(
            self.run_spec_dataframe_filter, axis=1)]

        selected_run_specs_names = list(selected_run_specs_df['run_spec.name'])

        *train_runs, eval_run = random.sample(
            selected_run_specs_names, self.num_example_runs + 1)

        train_run_specs_df = selected_run_specs_df[
            selected_run_specs_df['run_spec.name'].isin(train_runs)]

        eval_run_specs_df = selected_run_specs_df[
            selected_run_specs_df['run_spec.name'] == eval_run]

        train_request_states_df = pd.concat([
            r.request_states() for r in selected_runs
            if r.raw.run_spec().name in train_runs])

        train_stats_df = pd.concat([
            r.stats() for r in selected_runs
            if r.raw.run_spec().name in train_runs])

        _full_eval_request_states_df = pd.concat([
            r.request_states() for r in selected_runs
            if r.raw.run_spec().name == eval_run])

        random_eval_indices = random.sample(
            range(len(_full_eval_request_states_df)), self.num_eval_samples)
        eval_request_states_df = _full_eval_request_states_df.iloc[random_eval_indices]

        predicted_stats = self.predict(train_run_specs_df,
                                       train_request_states_df,
                                       train_stats_df,
                                       eval_run_specs_df,
                                       eval_request_states_df)

        # TODO: Do something meaningful with the predictions
        print(predicted_stats)
