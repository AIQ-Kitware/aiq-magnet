"""
The llama-consistency example pipeline.

    llama_predict[base_model, comp_model]   one job per model pair
        |
    llama_compare                           the node the card reads
"""

import kwdagger

from .llama_compare import ExampleLlamaConsistencyCompareCLI
from .llama_predict import ExampleLlamaEndpointCLI


class ExampleLlamaEndpoint(kwdagger.ProcessNode):
    """Gather HELM scores for a pair of models."""

    name = 'llama_predict'
    executable = 'python -m magnet.examples.llama_consistency.llama_predict'
    params = ExampleLlamaEndpointCLI

    def load_result(self, node_dpath):
        pass


class ExampleLlamaConsistencyCompare(kwdagger.ProcessNode):
    """Reduce a pair of scores to their gap."""

    name = 'llama_compare'
    executable = 'python -m magnet.examples.llama_consistency.llama_compare'
    params = ExampleLlamaConsistencyCompareCLI

    def load_result(self, node_dpath):
        pass


def llama_pipeline():
    """Create the prediction pipeline.

    Example:
        >>> from magnet.examples.llama_consistency.pipelines import llama_pipeline
        >>> dag = llama_pipeline()
        >>> sorted(dag.node_dict)
        ['llama_compare', 'llama_predict']
        >>> assert dag.node_dict['llama_compare'].inputs['scores_fpath'].pred
    """
    nodes = {
        'llama_predict': ExampleLlamaEndpoint(),
        'llama_compare': ExampleLlamaConsistencyCompare(),
    }
    nodes['llama_predict'].outputs['results_fpath'].connect(
        nodes['llama_compare'].inputs['scores_fpath']
    )
    dag = kwdagger.Pipeline(list(nodes.values()))
    dag.build_nx_graphs()
    return dag
