"""
The llama-consistency example pipeline.

    llama_predict[base_model, comp_model]   one job per model pair
        |
    compare                                 the node the card reads
"""

import kwdagger

from .compare import ConsistencyCompareCLI
from .llama_predict import ExampleLlamaEndpointCLI


class ExampleLlamaEndpoint(kwdagger.ProcessNode):
    """Gather HELM scores for a pair of models."""

    name = 'llama_predict'
    executable = 'python -m magnet.examples.llama_consistency.llama_predict'
    params = ExampleLlamaEndpointCLI

    def load_result(self, node_dpath):
        pass


class ConsistencyCompare(kwdagger.ProcessNode):
    """Reduce a pair of scores to their gap."""

    name = 'compare'
    executable = 'python -m magnet.examples.llama_consistency.compare'
    params = ConsistencyCompareCLI

    def load_result(self, node_dpath):
        pass


def llama_pipeline():
    """Create the prediction pipeline.

    Example:
        >>> from magnet.examples.llama_consistency.pipelines import llama_pipeline
        >>> dag = llama_pipeline()
        >>> sorted(dag.node_dict)
        ['compare', 'llama_predict']
        >>> assert dag.node_dict['compare'].inputs['scores_fpath'].pred
    """
    nodes = {
        'llama_predict': ExampleLlamaEndpoint(),
        'compare': ConsistencyCompare(),
    }
    nodes['llama_predict'].outputs['results_fpath'].connect(
        nodes['compare'].inputs['scores_fpath']
    )
    dag = kwdagger.Pipeline(list(nodes.values()))
    dag.build_nx_graphs()
    return dag
