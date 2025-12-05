import rich
from rich.markup import escape
import pandas as pd
from helm.benchmark.metrics.statistic import Stat

from magnet.predictor import Predictor


class InstancePrediction:
    def __init__(self,
                 run_spec_name,
                 instance_predict_id,
                 stat_name,
                 mean,
                 count=1,
                 sum=None,
                 sum_squared=None,
                 min=None,
                 max=None,
                 variance=0.0,
                 stddev=0.0):
        self.run_spec_name = run_spec_name
        self.instance_predict_id = instance_predict_id
        self.stat_name = stat_name
        self.mean = mean

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
                'magnet.instance_predict_id': p.instance_predict_id,
                'run_spec.name': p.run_spec_name,
                'per_instance_stats.stats.count': p.count,
                'per_instance_stats.stats.max': p.max,
                'per_instance_stats.stats.mean': p.mean,
                'per_instance_stats.stats.min': p.min,
                'per_instance_stats.stats.name.name': p.stat_name,
                'per_instance_stats.stats.stddev': p.stddev,
                'per_instance_stats.stats.sum': p.sum,
                'per_instance_stats.stats.sum_squared': p.sum_squared,
                'per_instance_stats.stats.variance': p.variance
            }
            for p in instance_predictions])

        return df


class InstancePredictor(Predictor):
    def compare_predicted_to_actual(self,
                                    predicted_instance_stats_df,
                                    eval_instance_stats_df):
        merged = pd.merge(predicted_instance_stats_df, eval_instance_stats_df,
                 on=['run_spec.name',
                     'magnet.instance_predict_id',
                     'per_instance_stats.stats.name.name'])

        human_mapping = {'run_spec.name': 'run_spec',
                         'per_instance_stats.instance_id': 'instance_id',
                         'magnet.instance_predict_id': 'prediction_id',
                         'per_instance_stats.stats.name.name': 'stat_name',
                         'per_instance_stats.stats.mean_x': 'predicted_mean',
                         'per_instance_stats.stats.mean_y': 'actual_mean'}

        vantage = ['run_spec.name',
                   'per_instance_stats.instance_id',
                   'magnet.instance_predict_id',
                   'per_instance_stats.stats.name.name',
                   'per_instance_stats.stats.mean_x',
                   'per_instance_stats.stats.mean_y']

        selected = merged[vantage]
        human_table = selected.rename(human_mapping, axis=1)
        # More human readable float
        rounded_human_table = human_table.round(3)

        rich.print(escape(rounded_human_table.to_string()))

        return human_table

    def predict(self,
                train_split,
                sequestered_test_split) -> list[InstancePrediction]:
        raise NotImplementedError

    def prepare_all_dataframes(self, helm_runs):
        train_split, test_split = super(InstancePredictor, self).prepare_all_dataframes(helm_runs)
        for split in [train_split, test_split]:
            _reindex_split(split)
        return train_split, test_split

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
        eval_instance_stats_df = test_split.per_instance_stats

        # TODO: Move the encapsulated splits
        predicted_instances = self.predict(train_split, sequestered_test_split)
        predicted_instance_stats_df = InstancePrediction.to_df(predicted_instances)

        return self.compare_predicted_to_actual(predicted_instance_stats_df, eval_instance_stats_df)


def _reindex_split(split):
    # Add a new dedicated column for easy merging downstream with
    # "actual" per-instance stats
    predict_id_col_name = "magnet.instance_predict_id"

    instance_stats_df = split.per_instance_stats
    scenario_state_df = split.scenario_state

    new_scenario_state_df = scenario_state_df.reset_index(drop=True)
    new_scenario_state_df = new_scenario_state_df.reset_index(names=predict_id_col_name)

    scenario_state_columns = new_scenario_state_df.columns.intersection(
        ['run_spec.name',
         'scenario_state.request_states.instance.id',
         'scenario_state.request_states.instance.perturbation.computed_on',
         'scenario_state.request_states.instance.perturbation.fairness',
         'scenario_state.request_states.instance.perturbation.name',
         'scenario_state.request_states.instance.perturbation.prob',
         'scenario_state.request_states.instance.perturbation.robustness',
         'scenario_state.request_states.instance.split'])

    per_instance_columns = instance_stats_df.columns.intersection(
        ['run_spec.name',
         'per_instance_stats.instance_id',
         'per_instance_stats.stats.name.perturbation.computed_on',
         'per_instance_stats.stats.name.perturbation.fairness',
         'per_instance_stats.stats.name.perturbation.name',
         'per_instance_stats.stats.name.perturbation.prob',
         'per_instance_stats.stats.name.perturbation.robustness',
         'per_instance_stats.stats.name.split'])

    assert len(per_instance_columns) == len(scenario_state_columns)

    merged = pd.merge(instance_stats_df, new_scenario_state_df,
                      left_on=list(per_instance_columns),
                      right_on=list(scenario_state_columns))
    new_instance_stats_df = merged[[predict_id_col_name, *instance_stats_df.columns]]

    split.scenario_state = new_scenario_state_df
    split.per_instance_stats = new_instance_stats_df
