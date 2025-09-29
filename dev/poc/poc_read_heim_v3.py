r"""
Example of loading precomputed HEIM metrics
"""

import ubelt as ub
import magnet
import kwutil
from line_profiler import profile
from magnet.utils.util_pandas import DotDictDataFrame


@profile
def main():
    heim_output_dpath = ub.Path('/data/crfm-helm-public/heim/benchmark_output')
    outs = magnet.HelmOutputs(heim_output_dpath)
    # Use both v1.0.0 and v1.1.0
    runs = []
    for suite in outs.suites('*'):
        runs.extend(suite.runs('*').existing())

    # Takes about 2 minutes to load everything.
    pman = kwutil.ProgressManager()
    with pman:
        tables = []
        for run in pman.progiter(runs, desc='Loading HELM runs'):
            table = load_relevant_run_info(run)
            if table is not None:
                tables.append(table)
            else:
                print(f'Skip {run}')

    import pandas as pd
    big_table = pd.concat(tables).reset_index(drop=True)

    from helm.common.object_spec import parse_object_spec
    run_specs = {k: parse_object_spec(k) for k in big_table['run_spec.name'].unique()}

    # Create helper columns
    big_table['run_spec.model'] = big_table['run_spec.name'].apply(lambda k: run_specs[k].args['model'])
    big_table['input_text_id'] = big_table['request_states.instance.input.text'].apply(ub.hash_data)

    for key, group in big_table.groupby(['run_spec.model']):
        print(key, len(group))

        # We now have a list of prompts and scores for this specific model.
        prompts = group['request_states.instance.input.text']
        scores = group['per_instance_stats.stat.expected_clip_score.max']


def load_relevant_run_info(run):
    """
    Build an aligned data frame with stats of interest.
    """
    # Only load data that has these stats computed
    stats_of_interest = [
        'expected_clip_score',
        'max_clip_score',
        # 'expected_clip_score_multilingual',
        # 'max_clip_score_multilingual'
    ]

    filtered_stats = []
    per_instance_stats = run.json.per_instance_stats()
    for instance_stats in per_instance_stats:
        # For this instance, determine if any of its statistics are of
        # interest.
        relevant_stats = [
            stat for stat in instance_stats['stats']
            if stat['name']['name'] in stats_of_interest
        ]
        if relevant_stats:
            base = ub.udict.difference(instance_stats, {'stats'})

            # Expand the relevant stats
            flat_stats = {}
            for stat in relevant_stats:
                stat_name = stat['name']['name']
                flat_stat = kwutil.DotDict.from_nested(stat, prefix=stat_name)
                flat_stats.update(flat_stat)

            new_info = {
                'per_instance_stats': {
                    **base,
                    'stat': flat_stats,
                },
                'run_spec.name': run.name
            }
            filtered_stats.append(new_info)

    if len(filtered_stats) == 0:
        # None of the instances had the requested metric data
        return None

    # Load up instance information from the scenario state
    filtered_states = []
    scenario_state = run.json.scenario_state()
    request_states = scenario_state.pop('request_states')

    # It seems like doing no filtering here could cause alignment issues, but
    # none of the subsequent asserts trigger on HEIM results.
    for request_state in request_states:
        new_state = {
            **scenario_state,
            'request_states': request_state}
        filtered_states.append(new_state)

    flat_filtered_stats = [kwutil.DotDict.from_nested(item) for item in filtered_stats]
    flat_filtered_states = [kwutil.DotDict.from_nested(item) for item in filtered_states]

    assert len(flat_filtered_states) == len(flat_filtered_stats), 'data is not aligned'
    combo_rows = []
    for state, stat in zip(flat_filtered_states, flat_filtered_stats):
        assert state['request_states.instance.id'] == stat['per_instance_stats.instance_id'], 'data is not aligned'
        combo_rows.append(state | stat)

    table = DotDictDataFrame(combo_rows)
    table['run_path'] = run.path
    return table


if __name__ == '__main__':
    """
    CommandLine:
        kernprof -lzvv -p magnet ~/code/magnet-sys-exploratory/dev/poc/poc_read_heim_v2.py
    """
    main()
