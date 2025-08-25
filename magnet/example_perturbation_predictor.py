from typing import List
import argparse

from sklearn.linear_model import LinearRegression
from helm.benchmark.metrics.statistic import Stat
import pandas as pd

from magnet.predictor import Predictor

class ExamplePerturbationPredictor(Predictor):
    """
    Class to demonstrate a stat prediction algorithm based on strength of perturbation

    Example:
        >>> import magnet
        >>> outputs = magnet.HelmOutputs.demo(run_entries=["boolq:data_augmentation=misspelling_sweep,model=openai/gpt2"], max_eval_instances=20)
        >>> suite = outputs.suites()[0].name
        >>> root_dir = outputs.root_dir
        >>> predictor_instance = ExamplePerturbationPredictor(num_eval_samples=5)
        >>> predictor_instance(root_dir, suite)
    """

    def predict(self,
                train_run_specs_df,
                train_scenario_states_df,
                train_stats_df,
                eval_run_specs_df,
                eval_scenario_states_df) -> dict[str, list[Stat]]:
        predicted_stats = {}

        perturbed_exact_match_stats_df = train_stats_df[
            (train_stats_df['stats.name.name'] == 'exact_match') &
            (train_stats_df['stats.name.perturbation.name'] == "misspellings") &
            (train_stats_df['stats.name.perturbation.computed_on'] == "perturbed")]

        train_run_spec_and_stats_df = pd.merge(
            train_run_specs_df, perturbed_exact_match_stats_df,
            left_on='run_spec.name', right_on='run_spec.name')

        # Create simple linear model for strength of perturbation to
        # exact_match performance
        model = LinearRegression()
        model.fit(train_run_spec_and_stats_df['stats.name.perturbation.prob'].values.reshape(-1, 1),
                  train_run_spec_and_stats_df['stats.mean'].values.reshape(-1, 1))

        for _, row in eval_run_specs_df.iterrows():
            perturbations = row['run_spec.data_augmenter_spec.perturbation_specs']

            assert len(perturbations) > 0
            misspelling_perturbation_prob = perturbations[0]['args']['prob']

            prediction = model.predict([[misspelling_perturbation_prob]])
            # `model.predict` outputs a 2d numpy array, need to unpack the single value
            prediction = prediction[0][0]

            predicted_stats.setdefault(row['run_spec.name'], []).append(
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

    predictor_instance = ExamplePerturbationPredictor()
    predictor_instance(args.root_dir, args.suite)
