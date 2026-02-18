"""
Two node pipeline for llama-consistency example card
"""

import kwdagger
import ubelt as ub

from llama_consistency.cli.llama_predict import ExampleLlamaEndpointCLI
from llama_consistency.cli.claim import ConsistencyClaimCLI

try:
    MODULE_DPATH = ub.Path(__file__).parent
except NameError:
    # for developer convenience
    MODULE_DPATH = ub.Path(".").resolve()


class ExampleLlamaEndpoint(kwdagger.ProcessNode):
    """Run the HELM results gathering step."""

    name = 'llama_predict'
    executable = f'python {MODULE_DPATH}/cli/llama_predict.py'
    params = ExampleLlamaEndpointCLI
    
    # FIXME: added manual tags to fix issue finding paths
    out_paths = {
        'results_fpath': 'results.json',
    }

    primary_out_key = 'results_fpath'

    algo_params = {
        'base_model',
        'comp_model',
    }

    def load_result(self, node_dpath):
        pass


class ConsistencyClaim(kwdagger.ProcessNode):
    """Score predictions against labels and expose metrics for aggregation."""

    name = 'claim_eval'
    executable = f'python {MODULE_DPATH}/cli/claim.py'
    params = ConsistencyClaimCLI

    in_paths = {
        'symbols_fpath',
    }

    out_paths = {
        'verdict_fpath': 'verdict.json',
    }

    primary_out_key = 'verdict_fpath'

    def load_result(self, node_dpath):
        pass

def llama_pipeline():
    """Create the prediction pipeline."""

    nodes = {
        'llama_predict': ExampleLlamaEndpoint(),
        'claim_eval': ConsistencyClaim(),
    }

    nodes['llama_predict'].outputs['results_fpath'].connect(
        nodes['claim_eval'].inputs['symbols_fpath']
    )

    dag = kwdagger.Pipeline(nodes)
    dag.build_nx_graphs()
    return dag