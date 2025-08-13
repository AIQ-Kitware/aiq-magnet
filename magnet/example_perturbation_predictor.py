from typing import List
import argparse

from sklearn.linear_model import LinearRegression
from helm.benchmark.metrics.statistic import Stat
import pandas as pd

from magnet.predictor import Predictor

class ExamplePerturbatioPredictor(Predictor):
    def run_spec_dataframe_fields(self):
        # The "name" field is required, and will be added if not
        # specified
        return [
            "name",
            "adapter_spec.model",
            "data_augmenter_spec.perturbation_specs.0.args.prob",
            "adapter_spec",
            "metric_specs",
            "data_augmenter_spec",
            "groups",
        ]

    def stats_dataframe_fields(self):
        return [
            "name.name",
            "name.split",
            "name.perturbation.name",
            "name.perturbation.computed_on",
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
        predicted_stats = []

        perturbed_exact_match_stats_df = train_stats_df[
            (train_stats_df['name.name'] == 'exact_match') &
            (train_stats_df['name.perturbation.name'] == "misspellings") &
            (train_stats_df['name.perturbation.computed_on'] == "perturbed")]

        train_run_spec_and_stats_df = pd.merge(
            train_run_specs_df, perturbed_exact_match_stats_df,
            left_on='name', right_on='run_spec.name')

        # Create simple linear model for strength of perturbation to
        # exact_match performance
        model = LinearRegression()
        model.fit(train_run_spec_and_stats_df['data_augmenter_spec.perturbation_specs.0.args.prob'].values.reshape(-1, 1),
                  train_run_spec_and_stats_df['mean'].values.reshape(-1, 1))

        for _, row in eval_run_specs_df.iterrows():
            misspelling_perturbation_prob = row['data_augmenter_spec.perturbation_specs.0.args.prob']
            prediction = model.predict([[misspelling_perturbation_prob]])
            # `model.predict` outputs a 2d numpy array, need to unpack the single value
            prediction = prediction[0][0]

            predicted_stats.append(
                Stat(**{'name':
                        {'name': 'predicted_exact_match',
                         'split': 'valid',
                         'perturbation': {'name': 'misspellings',
                                          'robustness': True,
                                          'fairness': False,
                                          'computed_on': 'perturbed',
                                          'prob': misspelling_perturbation_prob}},
                        'count': 1,
                        'sum': prediction,
                        'sum_squared': prediction ** 2,
                        'min': prediction,
                        'max': prediction,
                        'mean': prediction,
                        'variance': 0.0,
                        'stddev': 0.0}))

        return predicted_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run example perturbation predictor")

    parser.add_argument('-r', '--root-dir',
                        type=str,
                        required=True,
                        help="Root directory for HELM outputs (usually 'something/something/benchmark_output')")
    parser.add_argument('-s', '--suite',
                        type=str,
                        required=True,
                        help="Suite name")

    args = parser.parse_args()

    predictor_instance = ExamplePerturbatioPredictor()
    predictor_instance(args.root_dir, args.suite)
