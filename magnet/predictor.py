import ubelt as ub
import pandas as pd
import kwarray

from magnet.helm_outputs import HelmSuite, HelmOutputs
from helm.benchmark.metrics.statistic import Stat


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
                train_split, sequestered_test_split) -> dict[str, list[Stat]]:
        raise NotImplementedError

    def prepare_predict_inputs(self, helm_suite_path):
        # TODO: Is this unused? Can we remove it?
        train_split, test_split = self.prepare_all_dataframes(helm_suite_path)
        sequestered_test_split = test_split.sequester()
        # predict method doesn't get `eval_stats_df`
        return train_split, sequestered_test_split

    def prepare_all_dataframes(self, helm_suite_path):
        random = kwarray.ensure_rng(self.random_seed, api='python')

        suite_output = HelmSuite.coerce(helm_suite_path)

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

        train_split = TrainSplit(
            run_specs=train_run_specs_df,
            scenario_state=train_scenario_state_df,
            stats=train_stats_df,
        )

        test_split = TestSplit(
            run_specs=eval_run_specs_df,
            scenario_state=eval_scenario_state_df,
            stats=eval_stats_df,
        )
        return train_split, test_split

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

    def _coerce_helm_suite_inputs(self, *args, **kwargs):
        """
        The original API definition had the user give inputs as "root_dir" and
        "suite", which is somewhat unintuitive because they are both parts of
        what should be the same path, but two intermediate directories are
        arbitrarilly removed. A better design would be to just path the full
        path to the helm suite of interest and then break out path components
        if we do need them.

        To maintain backwards compatibility this function checks for the
        2-input style argument and allows it to resolve to a HelmSuite, but
        also allows for a single path to that suite to be given.
        """
        legacy_args = None
        if len(args) == 2:
            legacy_args = args
            if len(kwargs) > 0:
                raise ValueError(ub.paragraph(
                    '''
                    input looked like legacy positional arguments, but extra
                    information was given. This is not handled.
                    '''))
        elif set(kwargs) == {'root_dir', 'suite'}:
            legacy_args = kwargs['root_dir'], kwargs['suite']
            if len(args) > 0:
                raise ValueError(ub.paragraph(
                    '''
                    input looked like legacy keyword arguments, but extra
                    information was given. This is not handled.
                    '''))

        if legacy_args is not None:
            ub.schedule_deprecation(
                modname='magnet', name='root_dir/suite', type='input arguments',
                migration='Pass the full path to the suite instead',
                deprecate='0.0.1', error='0.1.0', remove='0.2.0',
            )
            root_dir, suite = legacy_args
            # be flexiable about if benchmark_outputs is given or not
            root_dir = HelmOutputs._coerce_input_path(root_dir)
            helm_suite_path = root_dir / 'runs' / suite
        else:
            if len(kwargs) > 0 or len(args) > 1:
                raise ValueError('Expected only one positional argument')
            helm_suite_path = ub.Path(args[0])
        return helm_suite_path

    def _run(self, *args, **kwargs):
        """
        Note: I like when __call__ corresponds to a named function (e.g.
        forward in pytorch), but I'm not sure what to call it here as we
        already have a predict function. For now I'm naming it _run, but I
        would like to find a better name.
        """
        helm_suite_path = self._coerce_helm_suite_inputs(*args, **kwargs)
        train_split, test_split = self.prepare_all_dataframes(helm_suite_path)
        sequestered_test_split = test_split.sequester()
        eval_stats_df = test_split.stats

        # TODO: Move the encapsulated splits
        predicted_stats = self.predict(train_split, sequestered_test_split)

        self.compare_predicted_to_actual(predicted_stats, eval_stats_df)

    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)


class DataSplit:
    """
    Enapsulates data for a particualr data split

    Attributes:
        run_specs: DataFrame | None
        scenario_state: DataFrame | None
        stats: DataFrame | None
    """
    def __init__(self, run_specs=None, scenario_state=None, stats=None):
        self.run_specs = run_specs
        self.scenario_state = scenario_state
        self.stats = stats


class TrainSplit(DataSplit):
    ...


class TestSplit(DataSplit):

    def sequester(self):
        """
        Drop the results for components that should not have access to it.
        """
        sequestered_split = SequesteredTestSplit(
            run_specs=self.run_specs,
            scenario_state=self.scenario_state
        )
        return sequestered_split


class SequesteredTestSplit(TestSplit):
    def __init__(self, run_specs=None, scenario_state=None, stats=None):
        assert stats is None, 'cannot specify stats here'
        super().__init__(run_specs=run_specs, scenario_state=scenario_state)
