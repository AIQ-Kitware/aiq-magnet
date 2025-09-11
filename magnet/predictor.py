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
                eval_scenario_states_df) -> dict[str, list[Stat]]:
        raise NotImplementedError

    def prepare_predict_inputs(self, root_dir, suite):
        *predict_inputs, eval_stats_df = self.prepare_all_dataframes(root_dir, suite)
        # predict method doesn't get `eval_stats_df`
        return predict_inputs

    def prepare_all_dataframes(self, root_dir, suite):
        random = kwarray.ensure_rng(self.random_seed, api='python')

        outputs = HelmOutputs(ub.Path(root_dir))
        suites_output = outputs.suites(suite)
        assert len(suites_output) == 1
        suite_output = suites_output[0]

        selected_runs = []
        for run in suite_output.runs():
            if self.run_spec_filter(run.json.run_spec()):
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
        eval_stats_df = _all_stats_df[_all_stats_df['run_spec.name'] == eval_run]

        _all_scenario_state_df = pd.concat([r.scenario_state() for r in selected_runs])
        train_scenario_state_df = _all_scenario_state_df[_all_scenario_state_df['run_spec.name'].isin(train_runs)]
        _full_eval_scenario_state_df = _all_scenario_state_df[_all_scenario_state_df['run_spec.name'] == eval_run]

        if self.num_eval_samples > len(_full_eval_scenario_state_df):
            raise RuntimeError("Not enough rows in eval scenario_state to sample")

        random_eval_indices = random.sample(
            range(len(_full_eval_scenario_state_df)), min(len(_full_eval_scenario_state_df), self.num_eval_samples))
        eval_scenario_state_df = _full_eval_scenario_state_df.iloc[random_eval_indices]

        return (train_run_specs_df,
                train_scenario_state_df,
                train_stats_df,
                eval_run_specs_df,
                eval_scenario_state_df,
                eval_stats_df)

    def compare_predicted_to_actual(self, predicted_stats, eval_stats_df):
        import kwutil
        from rich.console import Console
        from rich.table import Table
        from magnet.utils import util_pandas

        console = Console()

        for key, run_eval_stats_df in eval_stats_df.groupby(['run_spec.name']):
            run_spec_name, = key

            predicted_stats_for_run = predicted_stats.get(run_spec_name, [])

            stats_flat = [kwutil.DotDict.from_nested(stats.__dict__)
                          for stats in predicted_stats_for_run]
            flat_table = util_pandas.DotDictDataFrame(stats_flat)
            # Add a prefix to enable joins for join keys
            flat_table = flat_table.insert_prefix('predicted_stats')
            # Enrich with contextual metadata (primary key for run_spec joins)
            flat_table['run_spec.name'] = run_spec_name
            flat_table = flat_table.reorder(head=['run_spec.name'], axis=1)

            predicted_stats_df = flat_table
            # Remove 'predicted' from predicted Stats name to try and
            # match with original metric
            predicted_stats_df['predicted_stats.name.name'] =\
                predicted_stats_df['predicted_stats.name.name'].apply(
                    lambda n: n.replace('predicted_', ''))

            # If not populated already, these are required for merging
            predicted_stats_missing_columns =\
                (set(eval_stats_df.columns)
                 - (set(map(lambda x: x.replace('predicted_', ''), predicted_stats_df.columns))))
            predicted_stats_missing_columns = map(lambda x: f"predicted_{x}", predicted_stats_missing_columns)

            for col in predicted_stats_missing_columns:
                if col not in predicted_stats_df.columns:
                    predicted_stats_df[col] = float('nan')
                    col_dtype = str(eval_stats_df[col.replace('predicted_', '')].dtype)
                    predicted_stats_df[col] = predicted_stats_df[col].astype(col_dtype)

            possible_join_fields = {'stats.name.name',
                                    'stats.name.perturbation.computed_on',
                                    'stats.name.perturbation.fairness',
                                    'stats.name.perturbation.name',
                                    'stats.name.perturbation.prob',
                                    'stats.name.perturbation.robustness',
                                    'stats.name.split'}
            join_fields = possible_join_fields & set(eval_stats_df.columns)

            for col in join_fields:
                pred_col = f"predicted_{col}"
                col_dtype = str(eval_stats_df[col].dtype)
                predicted_stats_df[pred_col] = predicted_stats_df[pred_col].astype(col_dtype)

            merged = pd.merge(run_eval_stats_df, predicted_stats_df,
                     left_on=sorted(list(join_fields)),
                     right_on=sorted(list(map(lambda x: f"predicted_{x}", join_fields))))

            table = Table()
            for header in ('run_spec',
                           'source',
                           'name',
                           'mean',
                           'min',
                           'max',
                           'count',
                           'stddev',
                           'sum',
                           'sum_squared',
                           'variance'):
                table.add_column(header)

            def _field_formatter(x):
                if isinstance(x, float):
                    return f"{x:.3f}"
                else:
                    return str(x)

            for _, row in merged.iterrows():
                table.add_row(*map(_field_formatter,
                                   (row['run_spec.name_x'],
                                    'predicted',
                                    row['predicted_stats.name.name'],
                                    row['predicted_stats.mean'],
                                    row['predicted_stats.min'],
                                    row['predicted_stats.max'],
                                    row['predicted_stats.count'],
                                    row['predicted_stats.stddev'],
                                    row['predicted_stats.sum'],
                                    row['predicted_stats.sum_squared'],
                                    row['predicted_stats.variance'])))
                table.add_row(*map(_field_formatter,
                                   (row['run_spec.name_x'],
                                    'actual',
                                    row['stats.name.name'],
                                    row['stats.mean'],
                                    row['stats.min'],
                                    row['stats.max'],
                                    row['stats.count'],
                                    row['stats.stddev'],
                                    row['stats.sum'],
                                    row['stats.sum_squared'],
                                    row['stats.variance'])))

            console.print(table)


    def __call__(self, root_dir, suite):
        *predict_inputs, eval_stats_df = self.prepare_all_dataframes(
            root_dir, suite)

        # TODO: Move the encapsulated splits

        # train_split = TrainSplit(
        #     run_specs=train_run_specs_df,
        #     scenario_state=train_scenario_state_df,
        #     stats=train_stats_df,
        # )

        # test_split = TestSplit(
        #     run_specs=eval_run_specs_df,
        #     scenario_state=eval_scenario_state_df,
        # )

        # predicted_stats = self.predict(train_split, test_split)

        predicted_stats = self.predict(*predict_inputs)

        self.compare_predicted_to_actual(predicted_stats, eval_stats_df)


class DataSplit:
    """
    Enapsulates data for a particualr data split
    """
    def __init__(self, run_specs=None, scenario_state=None, stats=None):
        self.run_specs = run_specs
        self.scenario_state = scenario_state
        self.stats = stats


class TrainSplit(DataSplit):
    ...


class TestSplit:
    def __init__(self, run_specs=None, scenario_state=None):
        super().__init__(run_specs=run_specs, scenario_state=scenario_state)
