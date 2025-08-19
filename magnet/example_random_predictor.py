import random
from typing import List
import argparse

from helm.benchmark.metrics.statistic import Stat

from magnet.predictor import Predictor

class ExampleRandomPredictor(Predictor):
    """
    Class to demonstrate a random stat prediction algorithm

    Example:
        >>> import magnet
        >>> outputs = magnet.HelmOutputs.demo()
        >>> suite = outputs.suites()[0].name
        >>> root_dir = outputs.root_dir
        >>> predictor_instance = ExampleRandomPredictor(num_eval_samples=5)
        >>> predictor_instance(root_dir, suite)
        [{'name': 'predicted_exact_match', 'split': 'valid'}[min=0.12, mean=0.12, max=0.12, sum=0.12 (1)]]
    """
    def predict(self,
                train_run_specs_df,
                train_scenario_state_df,
                train_stats_df,
                eval_run_specs_df,
                eval_scenario_state_df) -> List[Stat]:
        predicted_stats = []

        for run_spec in eval_scenario_state_df.groupby(['run_spec.name']):
            prediction = (random.choice(range(0,101)) / 100)
            predicted_stats.append(
                Stat(**{'name':
                        {'name': 'predicted_exact_match',
                         'split': 'valid'},
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
        description="Run example random predictor")

    parser.add_argument('-r', '--root-dir',
                        type=str,
                        required=True,
                        help="Root directory for HELM outputs (usually 'something/something/benchmark_output')")
    parser.add_argument('-s', '--suite',
                        type=str,
                        required=True,
                        help="Suite name")

    args = parser.parse_args()

    predictor_instance = ExampleRandomPredictor()
    predictor_instance(args.root_dir, args.suite)
