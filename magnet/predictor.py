from typing import List

from helm.benchmark.metrics.statistic import Stat
import ubelt as ub
import pandas as pd
import kwarray

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
        random = kwarray.ensure_rng(self.random_seed, api='python')

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

        _all_stats_df = pd.concat([r.stats() for r in selected_runs])
        train_stats_df = _all_stats_df[_all_stats_df['run_spec.name'].isin(train_runs)]

        _all_scenario_state_df = pd.concat([r.scenario_state() for r in selected_runs])
        train_scenario_state_df = _all_scenario_state_df[_all_scenario_state_df['run_spec.name'].isin(train_runs)]
        _full_eval_scenario_state_df = _all_scenario_state_df[_all_scenario_state_df['run_spec.name'] == eval_run]

        if self.num_eval_samples > len(_full_eval_scenario_state_df):
            raise RuntimeError("Not enough rows in eval scenario_state to sample")

        random_eval_indices = random.sample(
            range(len(_full_eval_scenario_state_df)), min(len(_full_eval_scenario_state_df), self.num_eval_samples))
        eval_scenario_state_df = _full_eval_scenario_state_df.iloc[random_eval_indices]

        predicted_stats = self.predict(train_run_specs_df,
                                       train_scenario_state_df,
                                       train_stats_df,
                                       eval_run_specs_df,
                                       eval_scenario_state_df)

        # TODO: Do something meaningful with the predictions
        print(predicted_stats)
