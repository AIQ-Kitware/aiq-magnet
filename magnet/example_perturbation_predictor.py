import argparse

from sklearn.linear_model import LinearRegression
import pandas as pd

from magnet.predictor import RunPredictor, RunPrediction
from magnet.data_splits import TrainSplit, SequesteredTestSplit


class ExamplePerturbationPredictor(RunPredictor):
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
                train_split: TrainSplit,
                sequestered_test_split: SequesteredTestSplit
                ) -> list[RunPrediction]:
        # Unpack split classes into dataframes
        train_run_specs_df = train_split.run_specs
        train_scenario_states_df = train_split.scenario_state  # NOQA
        train_stats_df = train_split.stats

        eval_run_specs_df = sequestered_test_split.run_specs
        eval_scenario_states_df = sequestered_test_split.scenario_state  # NOQA

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

        predictions = []

        for _, row in eval_run_specs_df.iterrows():
            run_spec_name = row['run_spec.name']
            perturbations = row['run_spec.data_augmenter_spec.perturbation_specs']

            assert len(perturbations) > 0
            misspelling_perturbation_prob = perturbations[0]['args']['prob']

            prediction = model.predict([[misspelling_perturbation_prob]])
            # `model.predict` outputs a 2d numpy array, need to unpack the single value
            prediction = prediction[0][0]

            predictions.append(
                RunPrediction(
                    run_spec_name=run_spec_name,
                    split='valid',
                    stat_name='exact_match',
                    mean=prediction,
                    perturbation_parameters={
                        'name': 'misspellings',
                        'robustness': True,
                        'fairness': False,
                        'computed_on': 'perturbed',
                        'prob': misspelling_perturbation_prob}))

        return predictions

def main():
    parser = argparse.ArgumentParser(
        description="Run example perturbation predictor")

    parser.add_argument('helm_suite_path',
                        type=str,
                        help="Path to HELM run outputs for a suite (usually 'something/something/benchmark_output/runs/suite_name')")

    args = parser.parse_args()

    predictor_instance = ExamplePerturbationPredictor()
    predictor_instance(args.helm_suite_path)


if __name__ == "__main__":
    main()
