import json

import kwutil
import kwconf
import ubelt as ub

from magnet.backends.helm.helm_outputs import HelmOutputs
from magnet.backends.helm.helm_outputs import HelmSuiteRuns


def read_gathered_run_dpaths(manifest_fpath):
    """
    Resolve a kwdagger gather manifest into the HELM run directories it names.

    Each line is one materialize node's output directory, which holds its run
    under ``benchmark_output/runs/<suite>/<run_name>``.

    Args:
        manifest_fpath (str | PathLike): newline-delimited paths from kwdagger.

    Returns:
        list: HELM run directories, in manifest order.
    """
    manifest_fpath = ub.Path(manifest_fpath)
    run_dpaths = []
    for line in manifest_fpath.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        runs_root = ub.Path(line) / 'benchmark_output' / 'runs'
        if not runs_root.exists():
            raise FileNotFoundError(
                f'materialized run {line!r} has no benchmark_output/runs'
            )
        for suite_dpath in sorted(runs_root.iterdir()):
            run_dpaths.extend(sorted(suite_dpath.iterdir()))
    return run_dpaths


def load_kwdagger_result(node, node_dpath):
    """Load this node's flat JSON plus ProcessContext for kwdagger aggregate."""
    from kwdagger.aggregate_loader import new_process_context_parser
    from kwdagger.utils import util_dotdict

    node_dpath = ub.Path(node_dpath)
    output_fpath = node_dpath / node.out_paths[node.primary_out_key]
    payload = json.loads(output_fpath.read_text())
    nested = {}
    process_info = payload.get('_process')
    if process_info:
        nested.update(new_process_context_parser(process_info))
    nested['metrics'] = {
        key: value for key, value in payload.items() if not key.startswith('_')
    }
    flat = util_dotdict.DotDict.from_nested(nested)
    return flat.insert_prefix(node.name, index=1)


class ExampleLlamaEndpointCLI(kwconf.Config):
    """
    Stub for a prediction algorithm that grabs relevant scores from HELM precomputed results
    """

    base_model: str = kwconf.Value(
        None,
        required=True,
        help=ub.paragraph(
            """
        String corresponding to the model common name (run_spec.adapter_spec.model) in HELM results.
        """
        ),
        tags=['algo_param'],
    )

    comp_model: str = kwconf.Value(
        None,
        required=True,
        help=ub.paragraph(
            """
        String corresponding to the model common name (run_spec.adapter_spec.model) in HELM results.
        """
        ),
        tags=['algo_param'],
    )

    threshold: float = kwconf.Value(
        0.1,
        help=ub.paragraph(
            """
        Float indicating the consistency threshold used in resolving the claim
        """
        ),
        tags=['algo_param'],
    )

    helm_runs_path: str = kwconf.Value(
        './data/crfm-helm-public/lite/benchmark_output',
        help=ub.paragraph(
            """
        Path to precomputed HELM results, scanned for llama MMLU runs. How the
        legacy `pipeline:` card supplies its data. Ignored when `run_dpaths` is
        given.
        """
        ),
        tags=['algo_param'],
    )

    run_dpaths: str | None = kwconf.Value(
        None,
        help=ub.paragraph(
            """
        KWDagger gather manifest naming the materialized runs to score. When
        set, the runs are exactly the ones listed and no directory is scanned.
        """
        ),
        tags=['in_path'],
    )

    results_fpath: str = kwconf.Value(
        'results.json',
        help=ub.paragraph(
            """
        Default output path to store sweep parameters.
        """
        ),
        tags=['out_path', 'primary'],
    )

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        run_data = {}

        proc_context = kwutil.ProcessContext(
            name='consistency_example',
            type='process',
            config=kwutil.Json.ensure_serializable(dict(config)),
            track_emissions=False,
        )

        proc_context.start()

        # EXISTING LLAMA EVALUATION CARD CODE AGGREGATED
        # ----------------------------------------------
        ## run_specs Symbol Resolution

        if config.run_dpaths is not None:
            # The runs are declared upstream: this scores exactly what the
            # pipeline materialized, and nothing that merely shares a directory
            # with it.
            helm_lite_runs = read_gathered_run_dpaths(config.run_dpaths)
        else:
            # The legacy route, where the corpus is a directory to search.
            helm_data = HelmOutputs(ub.Path(config.helm_runs_path))
            helm_lite_runs = []
            for suite in helm_data.suites():
                # unix glob filter runs for llama models evaluated on MMLU
                helm_lite_runs.extend(suite.runs('mmlu*model=meta_*llama*').paths)

        # Create an aggregate view of all HELM Lite runs used for latest leaderboard
        run_specs = HelmSuiteRuns.coerce(helm_lite_runs)

        ## exact_match_scores Symbol Resolution

        run_stats = run_specs.stats()
        # filter to benchmark stats per https://github.com/stanford-crfm/helm/issues/2362
        run_stats = run_stats[
            (run_stats['stats.name.name'] == 'exact_match')
            & (run_stats['stats.name.perturbation.computed_on'].isna())
            & (run_stats['stats.name.split'] == 'test')
        ]

        # extract HELM model common names
        helm_models = (
            run_specs.run_spec()
            .set_index('run_spec.name')['run_spec.adapter_spec.model']
            .to_dict()
        )
        run_stats['model'] = run_stats['run_spec.name'].map(helm_models)

        # only specific models
        run_stats = run_stats[
            (run_stats['model'] == config.base_model)
            | (run_stats['model'] == config.comp_model)
        ]

        # average exact_match scores across subjects
        exact_match_scores_df = run_stats.groupby('model')['stats.mean'].mean()

        exact_match_scores = list(exact_match_scores_df.items())

        ## base_score Symbol Resolution
        base_score = [
            (name, score)
            for name, score in exact_match_scores
            if name == config.base_model
        ][0][1]

        ## comp_score Symbol Resolution
        comp_score = [
            (name, score)
            for name, score in exact_match_scores
            if name == config.comp_model
        ][0][1]

        # Write comp_score and base_score to results file

        # Flat, because a card reads the primary output's keys directly.
        # Anything not a result of the run goes under a leading underscore.
        run_data.update({
            'helm_runs_path': config.helm_runs_path,
            'scored_runs': len(helm_lite_runs),
            'base_model': config.base_model,
            'base_score': base_score,
            'comp_model': config.comp_model,
            'comp_score': comp_score,
            'threshold': config.threshold,
        })

        run_data['_process'] = proc_context.stop()

        dst_fpath = ub.Path(config.results_fpath)
        dst_fpath.parent.ensuredir()
        # TODO: use safer for writing result files.
        dst_fpath.write_text(json.dumps(run_data, indent=2))
        print(f'Wrote results to: {dst_fpath=}')


if __name__ == '__main__':
    r"""
    CommandLine:
        python ./magnet/examples/llama_consistency/llama_predict.py \
            --base_model meta/llama-2-70b \
            --comp_model meta/llama-3-70b \
            --helm_runs_path ./data/crfm-helm-public/lite/benchmark_output \
            --results_fpath ./results.json
    """
    ExampleLlamaEndpointCLI.main()
