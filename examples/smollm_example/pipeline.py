"""
The DAG: three nodes, one gather edge, and which of them holds a model.

The scaffolding half of this example. Everything that talks to a file or an
endpoint lives in :mod:`smollm_example.cli`; this module says how those
programs are wired together and which parameters name external endpoints.
Host versus container and leasing on versus off are invocation policies, not
node types in this DAG.

Written in Python rather than declared in the card for one reason: a leasing
node has to be told which of its parameters holds a catalog alias, and
kwdagger's node-spec allow-list is closed, so a card cannot say
``endpoint_params`` until kwdagger 0.4.1 ships ``extra_node_spec_keys``. In
Python it is just a class attribute, and this runs on the released 0.4.0.

The card can be ported to a declarative ``nodes:`` block once 0.4.1 lands --
see the README. Nothing else about the example changes when it is, which is
the argument for keeping the executables free of any of this.

Each node's inputs, outputs and parameters come from its CLI's own kwconf
declaration via ``params``, so the tags on
``AskModelCLI.items_fpath`` are the only place that fact is written.
"""

import kwdagger

from magnet.execution import MagnetYamlProcessNode

from smollm_example.cli.ask_model import AskModelCLI
from smollm_example.cli.compare_answers import (
    CompareAnswersCLI,
)
from smollm_example.cli.make_items import MakeItemsCLI

__all__ = ['Items', 'Ask', 'Compare', 'smollm_pipeline']


class Items(MagnetYamlProcessNode):
    """Write the dummy dataset. Needs no model, so it holds no GPU."""

    name = 'items'
    executable = 'python -m smollm_example.cli.make_items'
    params = MakeItemsCLI


class Ask(MagnetYamlProcessNode):
    """Ask one endpoint every question, inside a lease for that endpoint.

    ``endpoint_params`` is what makes this node lease at all: it names the
    parameters whose *values* are catalog aliases, so the cell configured with
    ``endpoint='smol-135'`` renders ``infer-stack run --endpoint smol-135``.

    ``endpoint`` is otherwise an ordinary parameter and the matrix sweeps it
    like any other, which is why two endpoints give two cells that each hold
    only the model they are using.
    """

    name = 'ask'
    executable = 'python -m smollm_example.cli.ask_model'
    params = AskModelCLI
    endpoint_params = ('endpoint',)
    # This demo always uses kwdagger's serial backend, so its own cells never
    # contend with each other. Waiting in infer-stack's admission queue would
    # only hide external/stale leases; fail fast instead so the user can inspect
    # `infer-stack leases`. Keep a finite backstop for hard-killed jobs.
    lease_queue = False
    lease_ttl = '1h'


class Compare(MagnetYamlProcessNode):
    """Reduce every endpoint's answers. Needs no model either."""

    name = 'compare'
    executable = 'python -m smollm_example.cli.compare_answers'
    params = CompareAnswersCLI


def smollm_pipeline() -> kwdagger.Pipeline:
    """
    Build the DAG.

    Returns:
        kwdagger.Pipeline: ``items -> ask (per endpoint) -> compare``.

    Example:
        >>> from smollm_example.pipeline import smollm_pipeline
        >>> dag = smollm_pipeline()
        >>> sorted(dag.node_dict)
        ['ask', 'compare', 'items']
        >>> dag.node_dict['ask'].endpoint_params
        ('endpoint',)
    """
    nodes = {
        'items': Items(),
        'ask': Ask(),
        'compare': Compare(),
    }

    # One dataset, read by every endpoint. An ordinary edge: one upstream cell
    # feeding many downstream ones.
    nodes['items'].outputs['out_fpath'].connect(
        nodes['ask'].inputs['items_fpath'],
    )

    # Every endpoint's answers fan in to the single comparison. An empty
    # `group_by` collects all of them into one collection; `all_success` means
    # a coverage number is never computed from a partial cohort, which would
    # make it look better than it was. Gather membership is resolved when the
    # whole matrix is compiled, so the comparison names exactly what it read.
    nodes['ask'].outputs['out_fpath'].connect(
        nodes['compare'].inputs['answer_fpaths'],
        gather=kwdagger.GatherSpec(
            group_by=[],
            order_by=['endpoint'],
            require='all_success',
        ),
    )

    return kwdagger.Pipeline(list(nodes.values()))
