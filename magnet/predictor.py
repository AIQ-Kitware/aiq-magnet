from typing import Any

import rich
from rich.markup import escape
import ubelt as ub
import pandas as pd
import kwarray

from magnet.helm_outputs import HelmRuns, HelmSuites
from magnet.data_splits import TestSplit, TrainSplit


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
                train_split, sequestered_test_split) -> Any:
        raise NotImplementedError

    def prepare_predict_inputs(self, helm_runs):
        # TODO: Is this unused? Can we remove it?
        train_split, test_split = self.prepare_all_dataframes(helm_runs)
        sequestered_test_split = test_split.sequester()

        # predict method doesn't get `eval_stats_df`
        return train_split, sequestered_test_split

    def prepare_all_dataframes(self, helm_runs):
        rng = kwarray.ensure_rng(self.random_seed, api='python')

        coerced_runs = HelmRuns.coerce(helm_runs)

        selected_runs = []
        for run in coerced_runs:
            if self.run_spec_filter(run.json.run_spec()):
                selected_runs.append(run)

        selected_run_specs_df = pd.concat([r.run_spec() for r in selected_runs])

        selected_run_specs_df = selected_run_specs_df[selected_run_specs_df.apply(
            self.run_spec_dataframe_filter, axis=1)]

        selected_run_specs_names = list(selected_run_specs_df['run_spec.name'])

        *train_runs, eval_run = rng.sample(
            selected_run_specs_names, self.num_example_runs + 1)

        train_run_specs_df = selected_run_specs_df[
            selected_run_specs_df['run_spec.name'].isin(train_runs)]

        eval_run_specs_df = selected_run_specs_df[
            selected_run_specs_df['run_spec.name'] == eval_run]

        _all_stats_df = pd.concat([r.stats() for r in selected_runs])
        train_stats_df = _all_stats_df[_all_stats_df['run_spec.name'].isin(train_runs)]
        eval_stats_df = _all_stats_df[_all_stats_df['run_spec.name'] == eval_run]

        _all_scenario_state_df = pd.concat([r.scenario_state() for r in selected_runs])
        train_scenario_state_df = _all_scenario_state_df[_all_scenario_state_df['run_spec.name'].isin(train_runs)]
        _full_eval_scenario_state_df = _all_scenario_state_df[_all_scenario_state_df['run_spec.name'] == eval_run]

        _all_per_instance_stats_df = pd.concat([r.per_instance_stats() for r in selected_runs])
        train_per_instance_stats_df = _all_per_instance_stats_df[_all_per_instance_stats_df['run_spec.name'].isin(train_runs)]
        _full_eval_per_instance_stats_df = _all_per_instance_stats_df[_all_per_instance_stats_df['run_spec.name'] == eval_run]

        if self.num_eval_samples > len(_full_eval_scenario_state_df):
            raise RuntimeError("Not enough rows in eval scenario_state to sample")

        # Sample up to `self.num_eval_samples` random instances from
        # scenario_states
        unique_instance_ids = _full_eval_scenario_state_df['scenario_state.request_states.instance.id'].unique()
        random_instance_indices = rng.sample(
            range(len(unique_instance_ids)), min(len(unique_instance_ids), self.num_eval_samples))
        random_instance_ids = unique_instance_ids[random_instance_indices]

        # Filter scenario_states down to selected random instances
        _flags = _full_eval_scenario_state_df['scenario_state.request_states.instance.id'].isin(random_instance_ids)
        eval_scenario_state_df = _full_eval_scenario_state_df[_flags]

        # Filter per_instance_stats down to selected random instances
        # May not need to filter this dataframe as it's not provided
        # to predictors
        _flags = _full_eval_per_instance_stats_df['per_instance_stats.instance_id'].isin(random_instance_ids)
        eval_per_instance_stats_df = _full_eval_per_instance_stats_df[_flags]

        train_split = TrainSplit(
            run_specs=train_run_specs_df,
            scenario_state=train_scenario_state_df,
            stats=train_stats_df,
            per_instance_stats=train_per_instance_stats_df
        )

        test_split = TestSplit(
            run_specs=eval_run_specs_df,
            scenario_state=eval_scenario_state_df,
            stats=eval_stats_df,
            per_instance_stats=eval_per_instance_stats_df
        )
        return train_split, test_split

    def _coerce_helm_runs(self, *args, helm_runs=None, helm_suites=None, **kwargs):
        """
        Gaurds the inputs to _evaluate to provide notifications about API
        changes.
        """
        num_input_specified = sum(_ is not None for _ in [helm_runs, helm_suites])
        if len(args) > 0 or len(kwargs) > 0 or num_input_specified == 0:
            raise ValueError(ub.paragraph(
                '''
                Usage of evaluate has changed.  To specify which HELM results
                to work on, pass keyword arguments explicitly. Specifically,
                pass ``helm_runs=<glob>`` as a glob pattern (or list of glob
                patterns) that matches the runs of interest.
                '''))

        if num_input_specified > 1:
            raise ValueError('Specify only one of helm_runs or helm_suites')

        if helm_runs is not None:
            helm_runs = HelmRuns.coerce(helm_runs)
        elif helm_suites is not None:
            helm_suites = HelmSuites.coerce(helm_suites)
            helm_runs = helm_suites.runs()
        return helm_runs

    def _evaluate(self, helm_runs=None):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        Execute the predictor evaluation

        Args:
            helm_runs (str | PathLike | List[str | PathLike]):
                A path path to the underlying directory or pattern
                matching multiple run paths.

            helm_suites (str | PathLike | List[str | PathLike]):
                A path path to the underlying directory or pattern
                matching multiple suite paths. All runs will be included.
        """
        helm_runs = self._coerce_helm_runs(*args, **kwargs)
        return self._evaluate(helm_runs=helm_runs)


class RunPrediction:
    def __init__(self,
                 run_spec_name,
                 split,
                 stat_name,
                 mean,
                 computed_on=None,
                 perturbation_parameters=None,
                 count=1,
                 sum=None,
                 sum_squared=None,
                 min=None,
                 max=None,
                 variance=0.0,
                 stddev=0.0):
        self.run_spec_name = run_spec_name
        self.split = split
        self.stat_name = stat_name
        self.mean = mean

        self.computed_on = computed_on

        if perturbation_parameters is None:
            perturbation_parameters = {}
        self.perturbation_parameters = {
               **{"name": None, "fairness": None, "robustness": None},
               **perturbation_parameters}

        self.count = count

        self.sum = sum
        self.sum_squared = sum_squared

        if self.sum_squared is None and self.sum is not None:
            self.sum_squared = self.sum ** 2

        self.min = min
        self.max = max

        self.variance = variance
        self.stddev = stddev

    @classmethod
    def to_df(cls, instance_predictions):
        df = pd.DataFrame([
            {
                'run_spec.name': p.run_spec_name,
                'stats.name.split': p.split,
                'stats.count': p.count,
                'stats.max': p.max,
                'stats.mean': p.mean,
                'stats.min': p.min,
                'stats.name.name': p.stat_name,
                'stats.stddev': p.stddev,
                'stats.sum': p.sum,
                'stats.sum_squared': p.sum_squared,
                'stats.variance': p.variance,
                'stats.name.perturbation.computed_on': p.computed_on,
                **{'stats.name.perturbation.{}'.format(k): v for k, v in p.perturbation_parameters.items()}
            }
            for p in instance_predictions])

        return df


class RunPredictor(Predictor):
    def compare_predicted_to_actual(self, predicted_stats_df, eval_stats_df):
        perturbation_cols = [c for c in predicted_stats_df.columns
                             if c.startswith('stats.name.perturbation')]
        join_cols = ['run_spec.name',
                     'stats.name.split',
                     'stats.name.name',
                     *perturbation_cols]

        merged = pd.merge(predicted_stats_df, eval_stats_df,
                          on=join_cols)

        human_mapping = {'run_spec.name': 'run_spec',
                         'stats.name.split': 'split',
                         'stats.name.name': 'stat_name',
                         'stats.mean_x': 'predicted_mean',
                         'stats.mean_y': 'actual_mean',
                         **{c: c.replace('stats.name.perturbation.', 'perturbation_')
                            for c in perturbation_cols}}

        vantage = ['run_spec.name',
                   'stats.name.split',
                   'stats.name.name',
                   'stats.mean_x',
                   'stats.mean_y',
                   *perturbation_cols]

        selected = merged[vantage]
        human_table = selected.rename(human_mapping, axis=1)
        # More human readable float
        human_table.round(3)

        rich.print(escape(human_table.to_string()))

    def predict(self,
                train_split,
                sequestered_test_split) -> list[RunPrediction]:
        raise NotImplementedError

    def _evaluate(self, helm_runs=None):
        """
        Execute the predictor evaluation

        Args:
            helm_runs (str | PathLike | List[str | PathLike]):
                A path path to the underlying suite directory or pattern
                matching multiple run paths.
        """
        train_split, test_split = self.prepare_all_dataframes(helm_runs=helm_runs)
        sequestered_test_split = test_split.sequester()
        eval_stats_df = test_split.stats

        run_predictions = self.predict(train_split, sequestered_test_split)
        predicted_stats_df = RunPrediction.to_df(run_predictions)

        self.compare_predicted_to_actual(predicted_stats_df, eval_stats_df)
