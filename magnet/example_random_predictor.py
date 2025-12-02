import random
import argparse

from magnet.predictor import RunPredictor, RunPrediction
from magnet.data_splits import TrainSplit, SequesteredTestSplit


class ExampleRandomPredictor(RunPredictor):
    """
    Class to demonstrate a random stat prediction algorithm

    Example:
        >>> from magnet.example_random_predictor import *  # NOQA
        >>> import magnet
        >>> outputs = magnet.HelmOutputs.demo()
        >>> helm_runs = outputs.suites()[0].runs()
        >>> predictor_instance = ExampleRandomPredictor(num_eval_samples=5)
        >>> predictor_instance(helm_runs=helm_runs)
    """
    def predict(self,
                train_split: TrainSplit,
                sequestered_test_split: SequesteredTestSplit
                ) -> list[RunPrediction]:
        # Unpack split classes into dataframes
        train_run_specs_df = train_split.run_specs  # NOQA
        train_scenario_states_df = train_split.scenario_state  # NOQA
        train_stats_df = train_split.stats  # NOQA

        eval_run_specs_df = sequestered_test_split.run_specs  # NOQA
        eval_scenario_state_df = sequestered_test_split.scenario_state

        predictions = []

        for key, _ in eval_scenario_state_df.groupby(['run_spec.name']):
            run_spec_name, = key
            prediction = (random.choice(range(0, 101)) / 100)

            predictions.append(
                RunPrediction(
                    run_spec_name=run_spec_name,
                    split="valid",
                    stat_name="exact_match",
                    mean=prediction))

        return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run example random predictor")

    parser.add_argument('helm_runs',
                        type=str,
                        help="Pattern matching the set of HELM runs to evaluate on")

    args = parser.parse_args()

    predictor_instance = ExampleRandomPredictor()
    predictor_instance(helm_runs=args.helm_runs)


if __name__ == "__main__":
    main()
