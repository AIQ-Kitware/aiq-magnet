import random
import argparse

from magnet.instance_predictor import InstancePredictor, InstancePrediction
from magnet.data_splits import TrainSplit, SequesteredTestSplit


class ExampleRandomInstancePredictor(InstancePredictor):
    """
    Class to demonstrate a random per-instance stat prediction algorithm

    Example:
        >>> import magnet
        >>> outputs = magnet.HelmOutputs.demo()
        >>> suite = outputs.suites()[0].name
        >>> root_dir = outputs.root_dir
        >>> predictor_instance = ExampleRandomInstancePredictor(num_eval_samples=5)
        >>> predictor_instance(root_dir, suite)
    """
    def predict(self,
                train_split: TrainSplit,
                sequestered_test_split: SequesteredTestSplit
                ) -> list[InstancePrediction]:
        # Unpack split classes into dataframes
        train_run_specs_df = train_split.run_specs  # NOQA
        train_scenario_states_df = train_split.scenario_state  # NOQA
        train_stats_df = train_split.stats  # NOQA

        eval_run_specs_df = sequestered_test_split.run_specs  # NOQA
        eval_scenario_state_df = sequestered_test_split.scenario_state

        predictions = []
        for _, row in eval_scenario_state_df.iterrows():
            run_spec_name = row['run_spec.name']
            instance_predict_id = row['magnet.instance_predict_id']

            prediction = random.choice([0.0, 1.0])

            predictions.append(
                InstancePrediction(
                    run_spec_name=run_spec_name,
                    instance_predict_id=instance_predict_id,
                    stat_name="exact_match",
                    mean=prediction))

        return predictions


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

    predictor_instance = ExampleRandomInstancePredictor()
    predictor_instance(args.root_dir, args.suite)
