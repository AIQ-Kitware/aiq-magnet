"""MAGNET's kwdagger process-node integration.

``MagnetProcessNode`` combines containerization and endpoint leasing::

    host:                python -m pkg.node ...
    container:           docker run ... python -m pkg.node ...
    lease:               infer-stack run ... -- python -m pkg.node ...
    lease + container:   infer-stack run ... -- docker run ... python ...

Container selection is applied first and leasing wraps the resulting command.
"""

from __future__ import annotations

from kwdagger.yaml_pipeline import YamlProcessNode

from magnet.containers import ContainerCapability, host_interpreter
from magnet.leasing import LEASE_NODE_SPEC_KEYS, LeaseCapability

__all__ = ['MagnetProcessNode']


class MagnetProcessNode(
    ContainerCapability,
    LeaseCapability,
    YamlProcessNode,
):
    """A kwdagger process node with MAGNET container and lease capabilities."""

    # Leasing-specific keys accepted in declarative node specifications.
    extra_node_spec_keys = LEASE_NODE_SPEC_KEYS

    @property
    def command(self) -> str:
        command = super().command
        if self.containerization_is_enabled():
            command = self.wrap_with_container(command)
        else:
            command = host_interpreter(command)
        command = self.wrap_with_lease(command)
        return command
