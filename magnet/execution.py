"""
MAGNET process nodes with orthogonal execution capabilities.

Containerization answers where a process runs. Leasing answers which inference
endpoint resources surround that process while it runs. Neither policy implies
the other, so their standalone node classes are siblings. This module provides
the common node that composes both capabilities for pipelines whose execution
policy is chosen by the invocation::

    host only:             python -m pkg.node ...
    container only:        docker run ... python -m pkg.node ...
    lease only:            infer-stack run ... -- python -m pkg.node ...
    lease + container:     infer-stack run ... -- docker run ... python ...

The composition order is explicit here: first choose the execution substrate
(host or container), then wrap that command in a lease. kwdagger's own cache
guard remains outside the rendered node command.
"""

from __future__ import annotations

import kwdagger
from kwdagger.yaml_pipeline import YamlProcessNode

from magnet.containers import ContainerCapability, render_container_command
from magnet.leasing import (
    LEASE_NODE_SPEC_KEYS,
    LeaseCapability,
    render_lease_command,
)

__all__ = [
    'MagnetProcessNode',
    'MagnetYamlProcessNode',
]


class MagnetProcessNode(
    ContainerCapability,
    LeaseCapability,
    kwdagger.ProcessNode,
):
    """A process node whose container and lease policies are independent.

    The class carries both capabilities so invocation settings may enable
    either, both, or neither. Capability inheritance is flat: leasing does not
    inherit containerization and containerization does not inherit leasing.
    """

    @property
    def command(self) -> str:
        command = super().command
        command = render_container_command(self, command)
        command = render_lease_command(self, command)
        return command


class MagnetYamlProcessNode(MagnetProcessNode, YamlProcessNode):
    """Declarative counterpart of :class:`MagnetProcessNode`.

    Lease-specific node-spec keys are inherited here so a declarative node can
    name which of its parameters contain catalog endpoint aliases. Container
    settings remain invocation settings and require no extra YAML keys.
    """

    extra_node_spec_keys = LEASE_NODE_SPEC_KEYS
