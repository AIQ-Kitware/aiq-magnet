# This takes the place of raw python right?
# What would an empty stub look like in the cookie cutter?
# This is what the user is on the hook to implement?

# TODO:
# TODO: confirm correct ordering of imports
# STEP 1: Define code logic
# STEP X: Create empty stub 
# TODO: write TODOs in the PR including CHANGELOG entry
# FIXME: be smarter about using existing YAML
# PLAN: 2 stage pipeline: fan out llamaendpoint(model=x) -> claim/run evaluation, then we can use aggregator in evaluation.py?
"""
# XXX: could I just extend yaml? or use yaml here: enabling a print style like original json writing
"""

import ubelt as ub
from magnet import HelmOutputs
from magnet.helm_outputs import HelmSuiteRuns
import scriptconfig as scfg
import rich
from rich.markup import escape
import kwutil
import json

class ExampleLlamaEndpointCLI(scfg.DataConfig):
    """
    Stub for a prediction algorithm that grabs relevant scores from HELM precomputed results
    """

    base_model = scfg.Value(None, required=True, help=ub.paragraph(
        '''
        String corresponding to the model common name (run_spec.adapter_spec.model) in HELM results.
        '''
        ),
        tags=['algo_param'])

    comp_model = scfg.Value(None, required=True, help=ub.paragraph(
        '''
        String corresponding to the model common name (run_spec.adapter_spec.model) in HELM results.
        '''
        ),
        tags=['algo_param'])
    
    helm_runs_path = scfg.Value('/home/bfenelon/AIQ-project/aiq-magnet/data/crfm-helm-public/lite/benchmark_output', help=ub.paragraph(
        '''
        Default path to precomputed HELM results.
        '''
        ),)
        #tags=['in_path'])

    results_fpath = scfg.Value('results.json', help=ub.paragraph(
        '''
        Default output path to store sweep parameters. 
        '''
        ),
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        run_data = {
            #'info': [], FIXME REMOVE for now to limit output
            'result': None,
        }

        proc_context = kwutil.ProcessContext(
            name='evaluate_consistency',
            type='process',
            config=kwutil.Json.ensure_serializable(dict(config)),
            track_emissions=False,
        )

        proc_context.start()

        # EXISTING LLAMA EVALUATION CARD CODE AGGREGATED
        # ----------------------------------------------
        ## run_specs Symbol Resolution

        # Load all HELM Lite releases 
        helm_data = HelmOutputs(ub.Path(config.helm_runs_path))

        # Collect runs from each release
        helm_lite_runs = []
        for suite in helm_data.suites():
            # unix glob filter runs for llama models evaluated on MMLU
            helm_lite_runs.extend(suite.runs("mmlu*model=meta_*llama*").paths)

        # Create an aggregate view of all HELM Lite runs used for latest leaderboard
        run_specs = HelmSuiteRuns.coerce(helm_lite_runs)

        ## exact_match_scores Symbol Resolution
        
        run_stats = run_specs.stats()
        # filter to benchmark stats per https://github.com/stanford-crfm/helm/issues/2362
        run_stats = run_stats[
            (run_stats['stats.name.name'] == 'exact_match') & 
            (run_stats['stats.name.perturbation.computed_on'].isna()) &
            (run_stats['stats.name.split'] == 'test')]

        # extract HELM model common names
        helm_models = run_specs.run_spec().set_index('run_spec.name')['run_spec.adapter_spec.model'].to_dict()
        run_stats['model'] = run_stats['run_spec.name'].map(helm_models)

        # only specific models
        run_stats = run_stats[(run_stats['model'] == config.base_model) | (run_stats['model'] == config.comp_model)]

        # average exact_match scores across subjects
        exact_match_scores_df = run_stats.groupby('model')['stats.mean'].mean()

        exact_match_scores = list(exact_match_scores_df.items())
        
        ## base_score Symbol Resolution
        base_score = [(name, score) for name, score in exact_match_scores if name == config.base_model][0][1]

        ## comp_score Symbol Resolution
        comp_score = [(name, score) for name, score in exact_match_scores if name == config.comp_model][0][1]

        # Write comp_score and base_score to results file

        run_data['result'] = {
            'helm_runs_path': config.helm_runs_path,
            'base_model': config.base_model,
            'base_score': base_score,
            'comp_model': config.comp_model,
            'comp_score': comp_score,
            'threshold': 0.1,
        }

        obj = proc_context.stop()
        # FIXME data['info'].append(obj)

        dst_fpath = ub.Path(config.results_fpath)
        dst_fpath.parent.ensuredir()
        dst_fpath.write_text(json.dumps(run_data, indent=2))
        print(f'Wrote results to: {dst_fpath=}')


__cli__ = ExampleLlamaEndpointCLI

if __name__ == '__main__':
    __cli__.main()

    r"""
    CommandLine:
        python ./cards/llama_consistency/cli/llama_predict.py \
            --base_model meta/llama-2-70b \
            --comp_model meta/llama-3-70b \
            --helm_runs_path ./data/crfm-helm-public/lite/benchmark_output \
            --results_fpath ./data/llama-example-runs
    """