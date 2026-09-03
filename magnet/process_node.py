"""
The MAGNET process-node integration surface.

Containerization and endpoint leasing are independent capabilities implemented
in :mod:`magnet.containers` and :mod:`magnet.leasing`. Most users should not
compose those mixins themselves. ``MagnetProcessNode`` is the one supported
integration point and carries both capabilities so an invocation may enable
neither, either, or both::

    host only:             python -m pkg.node ...
    container only:        docker run ... python -m pkg.node ...
    lease only:            infer-stack run ... -- python -m pkg.node ...
    lease + container:     infer-stack run ... -- docker run ... python ...

The wrapper order is explicit here rather than encoded in inheritance between
the capabilities: first choose host versus container execution, then put any
lease around that command. kwdagger's cache guard remains outside the rendered
node command.
"""

from __future__ import annotations

from kwdagger.yaml_pipeline import YamlProcessNode

from magnet.containers import ContainerCapability, render_container_command
from magnet.leasing import (
    LEASE_NODE_SPEC_KEYS,
    LeaseCapability,
    render_lease_command,
)

__all__ = ['MagnetProcessNode']


class MagnetProcessNode(
    ContainerCapability,
    LeaseCapability,
    YamlProcessNode,
):
    """A kwdagger process node with MAGNET's optional execution policies.

    ``YamlProcessNode`` already derives from kwdagger's ordinary
    ``ProcessNode`` and adds declarative metadata hooks, so it is the single
    kwdagger base here. Python-defined and declarative DAGs therefore use the
    same MAGNET surface without an artificial inheritance diamond.
    The capability mixins deliberately own no ``command`` property; composition
    and wrapper order live here.
    """

    # Which parameter names hold infer-stack endpoint aliases is legitimate
    # declarative node data. kwdagger >= 0.4.1 consults this allow-list; the
    # project still supports 0.4.0, where Python-defined DAGs set the class
    # attribute directly.
    extra_node_spec_keys = LEASE_NODE_SPEC_KEYS

    @property
    def command(self) -> str:
        command = super().command
        command = render_container_command(self, command)
        command = render_lease_command(self, command)
        return command
