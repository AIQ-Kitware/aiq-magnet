"""Define the SmolLM example DAG."""

import kwdagger

from magnet.process_node import MagnetProcessNode

from smollm_example.cli.ask_model import AskModelCLI
from smollm_example.cli.compare_answers import CompareAnswersCLI
from smollm_example.cli.make_items import MakeItemsCLI

__all__ = ['Items', 'Ask', 'Compare', 'smollm_pipeline']


class Items(MagnetProcessNode):
    """Write the input dataset."""

    name = 'items'
    executable = 'python -m smollm_example.cli.make_items'
    params = MakeItemsCLI


class Ask(MagnetProcessNode):
    """Query one endpoint, leasing the alias stored in ``endpoint``."""

    name = 'ask'
    executable = 'python -m smollm_example.cli.ask_model'
    params = AskModelCLI
    endpoint_params = ('endpoint',)

    # The example uses a serial backend, so its own cells do not contend.
    # Fail on unavailable capacity instead of waiting behind external leases.
    lease_queue = False
    # Backstop for a hard-killed process; normal completion releases the lease.
    lease_ttl = '1h'


class Compare(MagnetProcessNode):
    """Gather and compare every endpoint's answers."""

    name = 'compare'
    executable = 'python -m smollm_example.cli.compare_answers'
    params = CompareAnswersCLI


def smollm_pipeline() -> kwdagger.Pipeline:
    """Build ``items -> ask (per endpoint) -> compare``."""
    nodes = {
        'items': Items(),
        'ask': Ask(),
        'compare': Compare(),
    }

    nodes['items'].outputs['out_fpath'].connect(
        nodes['ask'].inputs['items_fpath'],
    )

    nodes['ask'].outputs['out_fpath'].connect(
        nodes['compare'].inputs['answer_fpaths'],
        gather=kwdagger.GatherSpec(
            group_by=[],
            order_by=['endpoint'],
            require='all_success',
        ),
    )

    return kwdagger.Pipeline(list(nodes.values()))
