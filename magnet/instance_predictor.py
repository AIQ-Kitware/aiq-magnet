import pandas as pd
from helm.benchmark.metrics.statistic import Stat

from magnet.predictor import Predictor


class InstancePredictor(Predictor):
    def compare_predicted_to_actual(self,
                                    predicted_instance_stats,
                                    eval_instance_stats_df):
        import kwutil
        from rich.console import Console
        from rich.table import Table
        from magnet.utils import util_pandas

        def _field_formatter(x):
            if isinstance(x, float):
                return f"{x:.3f}"
            else:
                return str(x)

        console = Console()

        for key, run_eval_stats_df in eval_instance_stats_df.groupby(['run_spec.name']):
            run_spec_name, = key

            table = Table()
            for header in ('run_spec',
                           'instance_id',
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

            for subkey, instance_stats_df in run_eval_stats_df.groupby(['per_instance_stats.instance_id']):
                instance_id, = subkey

                predicted_instance_stats_for_run =\
                    predicted_instance_stats.get(run_spec_name, {}).get(
                        instance_id, [])

                stats_flat = [kwutil.DotDict.from_nested(stats.__dict__)
                              for stats in predicted_instance_stats_for_run]
                flat_table = util_pandas.DotDictDataFrame(stats_flat)
                # Add a prefix to enable joins for join keys
                flat_table = flat_table.insert_prefix('predicted_per_instance_stats.stats')
                # Enrich with contextual metadata (primary key for run_spec joins)
                flat_table['run_spec.name'] = run_spec_name
                flat_table['per_instance_stats.instance_id'] = instance_id
                # Not sure if it's correct to hardcode this value
                # here; we don't use it for any comparison or
                # analysis, just to make the dataframes joinable
                flat_table['per_instance_stats.train_trial_index'] = 0
                flat_table = flat_table.reorder(head=['run_spec.name'], axis=1)

                predicted_instance_stats_df = flat_table
                # Remove 'predicted' from predicted Stats name to try and
                # match with original metric
                predicted_instance_stats_df['predicted_per_instance_stats.stats.name.name'] =\
                    predicted_instance_stats_df['predicted_per_instance_stats.stats.name.name'].apply(
                        lambda n: n.replace('predicted_', ''))

                # If not populated already, these are required for merging
                predicted_instance_stats_missing_columns =\
                    (set(instance_stats_df.columns)
                     - (set(map(lambda x: x.replace('predicted_', ''), predicted_instance_stats_df.columns))))
                predicted_instance_stats_missing_columns = map(lambda x: f"predicted_{x}", predicted_instance_stats_missing_columns)

                for col in predicted_instance_stats_missing_columns:
                    if col not in predicted_instance_stats_df.columns:
                        import xdev
                        predicted_instance_stats_df[col] = float('nan')
                        col_dtype = str(instance_stats_df[col.replace('predicted_', '')].dtype)
                        predicted_instance_stats_df[col] = predicted_instance_stats_df[col].astype(col_dtype)

                possible_join_fields = {'per_instance_stats.stats.name.name',
                                        'per_instance_stats.stats.name.perturbation.computed_on',
                                        'per_instance_stats.stats.name.perturbation.fairness',
                                        'per_instance_stats.stats.name.perturbation.name',
                                        'per_instance_stats.stats.name.perturbation.prob',
                                        'per_instance_stats.stats.name.perturbation.robustness',
                                        'per_instance_stats.stats.name.split'}
                join_fields = possible_join_fields & set(instance_stats_df.columns)

                for col in join_fields:
                    pred_col = f"predicted_{col}"
                    col_dtype = str(instance_stats_df[col].dtype)
                    predicted_instance_stats_df[pred_col] = predicted_instance_stats_df[pred_col].astype(col_dtype)

                merged = pd.merge(instance_stats_df, predicted_instance_stats_df,
                                  left_on=sorted(list(join_fields)),
                                  right_on=sorted(list(map(lambda x: f"predicted_{x}", join_fields))))

                for _, row in merged.iterrows():
                    table.add_row(*map(_field_formatter,
                                       (row['run_spec.name_x'],
                                        row['per_instance_stats.instance_id_x'],
                                        'predicted',
                                        row['predicted_per_instance_stats.stats.name.name'],
                                        row['predicted_per_instance_stats.stats.mean'],
                                        row['predicted_per_instance_stats.stats.min'],
                                        row['predicted_per_instance_stats.stats.max'],
                                        row['predicted_per_instance_stats.stats.count'],
                                        row['predicted_per_instance_stats.stats.stddev'],
                                        row['predicted_per_instance_stats.stats.sum'],
                                        row['predicted_per_instance_stats.stats.sum_squared'],
                                        row['predicted_per_instance_stats.stats.variance'])))
                    table.add_row(*map(_field_formatter,
                                       (row['run_spec.name_x'],
                                        row['per_instance_stats.instance_id_x'],
                                        'actual',
                                        row['per_instance_stats.stats.name.name'],
                                        row['per_instance_stats.stats.mean'],
                                        row['per_instance_stats.stats.min'],
                                        row['per_instance_stats.stats.max'],
                                        row['per_instance_stats.stats.count'],
                                        row['per_instance_stats.stats.stddev'],
                                        row['per_instance_stats.stats.sum'],
                                        row['per_instance_stats.stats.sum_squared'],
                                        row['per_instance_stats.stats.variance'])))

            console.print(table)


    def predict(self,
                train_split,
                sequestered_test_split) -> dict[str, dict[str, list[Stat]]]:
        raise NotImplementedError

    def _run(self, *args, **kwargs):
        helm_suite_path = self._coerce_helm_suite_inputs(*args, **kwargs)
        train_split, test_split = self.prepare_all_dataframes(helm_suite_path)
        sequestered_test_split = test_split.sequester()
        eval_instance_stats_df = test_split.per_instance_stats

        # TODO: Move the encapsulated splits
        predicted_instance_stats = self.predict(train_split, sequestered_test_split)

        self.compare_predicted_to_actual(predicted_instance_stats, eval_instance_stats_df)
