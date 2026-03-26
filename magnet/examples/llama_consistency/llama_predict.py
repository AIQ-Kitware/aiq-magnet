import json

import ubelt as ub
import kwutil
import rich
from rich.markup import escape
import scriptconfig as scfg

from magnet import HelmOutputs
from magnet.helm_outputs import HelmSuiteRuns

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
    
    threshold = scfg.Value(0.1, help=ub.paragraph(
        '''
        Float indicating the consistency threshold used in resolving the claim
        '''
        ),
        tags=['algo_param'])
    
    helm_runs_path = scfg.Value('./data/crfm-helm-public/lite/benchmark_output', help=ub.paragraph(
        '''
        Default path to precomputed HELM results.
        '''
        ),
        tags=['algo_param'])

    results_fpath = scfg.Value('results.json', help=ub.paragraph(
        '''
        Default output path to store sweep parameters. 
        '''
        ),
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        run_data = {
            'result': None,
        }

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
            'threshold': config.threshold,
        }

        obj = proc_context.stop()

        dst_fpath = ub.Path(config.results_fpath)
        dst_fpath.parent.ensuredir()
        dst_fpath.write_text(json.dumps(run_data, indent=2))
        print(f'Wrote results to: {dst_fpath=}')


__cli__ = ExampleLlamaEndpointCLI

if __name__ == '__main__':
    __cli__.main()

    r"""
    CommandLine:
        python ./magnet/examples/llama_consistency/llama_predict.py \
            --base_model meta/llama-2-70b \
            --comp_model meta/llama-3-70b \
            --helm_runs_path ./data/crfm-helm-public/lite/benchmark_output \
            --results_fpath ./results.json
    """