"""
The node classes this example's card names.

Only one node here needs a model, so only one node leases. That is the whole
argument for per-node leasing stated as a DAG: wrapping the *evaluation* in a
lease would hold both SmolLM models while the dataset is being written and
while the comparison runs, neither of which can use them.

``endpoint_params`` has no key in kwdagger's node spec -- it names parameters,
which are card data, but it is a statement about the node's behaviour -- so a
card that leases declares a subclass here and names it in ``class:``.
Everything else about the node stays data in the card.
"""

from magnet.containers import ContainerYamlProcessNode
from magnet.leasing import LeasedYamlProcessNode

__all__ = ['AskModel', 'PlainStep']


class AskModel(LeasedYamlProcessNode):
    """The one node that talks to a model, so the one node that leases.

    ``endpoint`` holds a catalog alias (``smol-135``), and its value is what
    ``infer-stack run --endpoint`` is given. Because it is an ordinary
    ``algo_param``, the matrix sweeps it like any other axis: two endpoints
    means two cells, each holding only the model it is using.
    """

    endpoint_params = ('endpoint',)


class PlainStep(ContainerYamlProcessNode):
    """A step that runs in the image but needs no model, and so no lease.

    Named rather than left as kwdagger's default ``YamlProcessNode`` because a
    plain node cannot be containerized: ``--container_image`` would be accepted
    and never read, and the run would go green having containerized nothing.
    """
