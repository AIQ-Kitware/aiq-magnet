import os
from glob import glob
import random
from typing import List
import json

from helm.benchmark.metrics.statistic import Stat

from magnet.loaders import (
    load_run_spec,
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
        # To be overridden
        return True

    def model_filter(self, model):
        # To be overridden
        return True

    def predict(self,
                train_scenario_states_df,
                train_stats_df,
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

        *train_runs, eval_run = random.sample(selected_run_specs, self.num_example_runs + 1)

        train_scenario_states_df = load_all_scenario_states_as_dataframe(
            suite, [r.name for r in train_runs], root_dir=root_dir)

        train_stats_df = load_all_stats_as_dataframe(
            suite, [r.name for r in train_runs], root_dir=root_dir)

        _full_eval_scenario_states_df = load_all_scenario_states_as_dataframe(
            suite, [eval_run.name], root_dir=root_dir)

        random_eval_indices = random.sample(
            range(len(_full_eval_scenario_states_df)), self.num_eval_samples)
        eval_scenario_states_df = _full_eval_scenario_states_df.iloc[random_eval_indices]

        predicted_stats = self.predict(train_scenario_states_df,
                                       train_stats_df,
                                       eval_scenario_states_df)

        # TODO: Do something meaningful with the predictions
        print(predicted_stats)
