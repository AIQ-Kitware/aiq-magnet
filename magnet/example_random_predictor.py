import random
import argparse

from helm.benchmark.metrics.statistic import Stat

from magnet.predictor import Predictor, TrainSplit, SequesteredTestSplit


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
    """
    def predict(self,
                train_split: TrainSplit,
                sequestered_test_split: SequesteredTestSplit
                ) -> dict[str, list[Stat]]:
        predicted_stats = {}

        # Unpack split classes into dataframes
        train_run_specs_df = train_split.run_specs  # NOQA
        train_scenario_states_df = train_split.scenario_state  # NOQA
        train_stats_df = train_split.stats  # NOQA

        eval_run_specs_df = sequestered_test_split.run_specs  # NOQA
        eval_scenario_state_df = sequestered_test_split.scenario_state

        for key, _ in eval_scenario_state_df.groupby(['run_spec.name']):
            run_spec_name, = key
            prediction = (random.choice(range(0, 101)) / 100)
            predicted_stats.setdefault(run_spec_name, []).append(
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
