"""
Two node pipeline for llama-consistency example card
"""

import kwdagger
import ubelt as ub

from .llama_predict import ExampleLlamaEndpointCLI
from .claim import ConsistencyClaimCLI

class ExampleLlamaEndpoint(kwdagger.ProcessNode):
    """Run the HELM results gathering step."""

    name = 'llama_predict'
    executable = 'python -m magnet.examples.llama_consistency.llama_predict'
    params = ExampleLlamaEndpointCLI

    def load_result(self, node_dpath):
        pass


class ConsistencyClaim(kwdagger.ProcessNode):
    """Score predictions against labels and expose metrics for aggregation."""

    name = 'claim_eval'
    executable = f'python -m magnet.examples.llama_consistency.claim'
    params = ConsistencyClaimCLI

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