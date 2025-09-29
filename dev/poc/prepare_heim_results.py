#!/usr/bin/env python3
r"""
Example of to prepare precomputed HEIM metrics
"""

import ubelt as ub
import kwutil
import scriptconfig as scfg
import magnet
from magnet.utils.util_pandas import DotDictDataFrame


class PrepareHeimResultsConfig(scfg.DataConfig):
    r"""
    Read all HEIM results and output them in a simple JSON format.

    Usage
    -----
    First it is a good idea to download HEIM results outside of the context of
    this script to ensure full control.

    The DOWNLOAD_DIR environment variable gives a recommended location for
    downloading results, but feel free to change this.

    .. code:: bash

        DOWNLOAD_DIR=/data/crfm-helm-public

        python -m magnet.backends.helm.download_helm_results "$DOWNLOAD_DIR" --benchmark=heim --version=v1.0.0 --backend=gsutil
        python -m magnet.backends.helm.download_helm_results "$DOWNLOAD_DIR" --benchmark=heim --version=v1.1.0 --backend=gsutil

    Now Call this script (note the path to the script may need to be modified
    as it is not part of the magnet package in its current proof-of-concept
    form. We assume the code repo is checked out in ~/code

    .. code:: bash

        # Change if you want your outputs in a different location
        HEIM_RESULT_DIR=./heim_results

        python ~/code/aiq-magnet/dev/poc/prepare_heim_results.py \
                --download_dir "$DOWNLOAD_DIR" \
                --output_dir "$HEIM_RESULT_DIR"

    """

    output_dir = scfg.Value('./heim_results', help=ub.paragraph(
        '''
        Directory where output json files will be written.
        '''))

    download_dir = scfg.Value('/data/crfm-helm-public', help=ub.paragraph(
        '''
        This is where the heim/benchmark_output/runs/v1.*/* results should be.
        They will be downloaded if needed.
        '''))


def main(argv=None, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/aiq-magnet/dev/poc'))
        >>> from poc_read_heim_v3 import *  # NOQA
        >>> argv = 0
        >>> kwargs = dict()
        >>> main(argv=argv, **config)
    """
    config = PrepareHeimResultsConfig.cli(
        argv=argv, data=kwargs, strict=True, verbose='auto',
        special_options=False)

    # This is where the heim/benchmark_output/runs/v1.*/* results should be
    download_dir = config.download_dir

    heim_output_dpath = ensure_heim_is_downloaded(download_dir)

    # Takes about 2 minutes to load everything.
    pman = kwutil.ProgressManager()
    load_info = {
        'loaded': 0,
        'skipped': 0,
    }
    with pman:
        tables = []
        outs = magnet.HelmOutputs(heim_output_dpath)
        # Use both v1.0.0 and v1.1.0
        for suite in pman.progiter(outs.suites('*'), desc='Loading suites'):
            runs = suite.runs('*').existing()
            for run in pman.progiter(runs, desc=f'Loading HELM runs in suite {suite.name}'):
                table = load_relevant_run_info(run)
                if table is not None:
                    tables.append(table)
                    load_info['loaded'] += 1
                else:
                    print(f'Skip {run}')
                    load_info['skipped'] += 1
                pman.update_info(ub.urepr(load_info, nl=1))

    import pandas as pd
    big_table = pd.concat(tables).reset_index(drop=True)

    # Create helper columns
    from helm.common.object_spec import parse_object_spec
    run_specs = {k: parse_object_spec(k) for k in big_table['run_spec.name'].unique()}
    big_table['run_spec.model'] = big_table['run_spec.name'].apply(lambda k: run_specs[k].args['model'])
    big_table['input_text_id'] = big_table['request_states.instance.input.text'].apply(ub.hash_data)

    output_dpath = ub.Path(config.output_dir).ensuredir()

    for key, group in big_table.groupby(['run_spec.model']):
        print(key, len(group))
        model_name, = key

        fname = f'results-{model_name}.json'
        fpath = output_dpath / fname

        # We now have a list of prompts and scores for this model in:
        # * group['request_states.instance.input.text']
        # * group['per_instance_stats.stat.expected_clip_score.max']

        rows = []
        for record in group.to_dict('records'):
            # NOTE: If we want any more metadata, we may want to write more
            rows.append({
                'prompt': record['request_states.instance.input.text'],
                'clip_score': record['per_instance_stats.stat.expected_clip_score.max'],
            })

        print(f'Write results to: fpath={fpath}')
        fpath.write_text(kwutil.Json.dumps(rows))


def ensure_heim_is_downloaded(download_dir):
    from magnet.backends.helm import download_helm_results
    heim_output_dpath = ub.Path(download_dir) / 'heim/benchmark_output'

    if len(heim_output_dpath.ls('*/*')) < 2:
        # The download script will try to do everything again by default assume
        # everything is downloaded if the main directories are there.
        backend = 'gsutil'
        backend = 'fsspec'
        download_helm_results.main(
            argv=False, download_dir=download_dir, version='v1.1.0',
            benchmark='heim', backend=backend)
        download_helm_results.main(
            argv=False, download_dir=download_dir, version='v1.0.0',
            benchmark='heim', backend=backend)

    return heim_output_dpath


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
        python ~/code/aiq-magnet/dev/poc/prepare_heim_results.py --help
        kernprof -lzvv -p magnet ~/code/aiq-magnet/dev/poc/prepare_heim_results.py

    """
    main()
