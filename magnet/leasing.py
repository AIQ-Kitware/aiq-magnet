"""Per-node infer-stack leasing for kwdagger commands.

``LeaseCapability`` identifies endpoint aliases from node parameters and wraps
commands with ``infer-stack run``. Under Slurm it can restrict infer-stack to
the GPUs allocated to the job.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
from typing import Any


__all__ = [
    'LeaseCapability',
    'LeaseSettings',
    'INSIDE_LEASE_ENVVAR',
    'GPU_ALLOW_LIST_EXPANSION',
    'LEASE_NODE_SPEC_KEYS',
]


@dataclasses.dataclass(frozen=True)
class LeaseSettings:
    """Invocation-level settings for per-node endpoint leasing."""

    #: Wrap leasable node commands with ``infer-stack run``.
    enabled: bool = False

    #: Restrict leased nodes to the GPUs allocated by Slurm.
    allowed_gpus: bool = True

    def apply(self, pipeline: Any) -> None:
        """Apply these invocation settings to every leasable node in a DAG."""
        for node in (getattr(pipeline, 'node_dict', None) or {}).values():
            if isinstance(node, LeaseCapability):
                node.apply_lease_settings(self)


#: Exported by ``infer-stack run`` while a command is already inside a lease.
INSIDE_LEASE_ENVVAR = 'INFER_STACK_LEASE_ID'

#: YAML keys owned by the leasing capability.
LEASE_NODE_SPEC_KEYS = frozenset({
    'endpoint_params',
    'lease_ttl',
    'lease_timeout',
    'lease_queue',
})




#: Shell expansion for ``--allowed_gpus=<indices>`` inside a Slurm job.
#:
#: The DAG is rendered before a Slurm allocation exists, so the value must
#: expand in the job shell. When neither Slurm variable is set, the unquoted
#: expansion disappears from the argv. ``SLURM_STEP_GPUS`` is the fallback for
#: ``SLURM_JOB_GPUS``.
#:
#: ``CUDA_VISIBLE_DEVICES`` is excluded because it may contain GPU UUIDs while
#: infer-stack expects integer indices.
GPU_ALLOW_LIST_EXPANSION = (
    '${SLURM_JOB_GPUS:+--allowed_gpus=}'
    '${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:+--allowed_gpus=$SLURM_STEP_GPUS}}'
)



class LeaseCapability:
    """Mixin providing endpoint leasing state and command wrapping.

    ``endpoint_params`` names configuration fields whose values are infer-stack
    endpoint aliases. Override :meth:`resolve_lease_endpoints` when aliases are
    derived differently.
    """

    endpoint_params: tuple[str, ...] = ()
    lease_ttl: str | None = '8h'
    lease_timeout: str | int | None = 1800
    lease_queue: bool = True
    lease_enabled: bool = False
    lease_allowed_gpus: bool = True

    def leasing_is_enabled(self) -> bool:
        """Whether this node should bracket its command in a lease."""
        if not self.lease_enabled:
            return False
        return not os.environ.get(INSIDE_LEASE_ENVVAR)

    def _lease_slurm_gpu_allow_list(self) -> str:
        """Shell text restricting infer-stack to this Slurm allocation."""
        if not self.lease_allowed_gpus:
            return ''
        return GPU_ALLOW_LIST_EXPANSION

    def apply_lease_settings(self, settings: LeaseSettings) -> None:
        """Apply invocation-level leasing and GPU-allocation settings."""
        self.lease_enabled = bool(settings.enabled)
        self.lease_allowed_gpus = bool(settings.allowed_gpus)

    def resolve_lease_endpoints(self) -> list[str]:
        """Return non-empty endpoint aliases, deduplicated in declaration order."""
        config = self.final_config or {}
        names: list[str] = []
        for key in self.endpoint_params:
            value = config.get(key)
            if value is None:
                continue
            value = str(value).strip()
            if value and value not in names:
                names.append(value)
        return names

    def wrap_with_lease(self, command: str) -> str:
        """Wrap ``command`` with ``infer-stack run`` when leasing is enabled."""
        if not self.leasing_is_enabled():
            return command
        names = self.resolve_lease_endpoints()
        if not names:
            return command
        return self._lease_command_prefix(names) + ' \\\n    ' + command

    def _lease_command_prefix(self, names: list[str]) -> str:
        # infer-stack accepts one comma-separated --endpoint value.
        parts = ['infer-stack', 'run', '--endpoint', shlex.quote(','.join(names))]
        if self.lease_ttl:
            parts += ['--ttl', str(self.lease_ttl)]
        if self.lease_timeout is not None:
            parts += ['--timeout', str(self.lease_timeout)]
        if self.lease_queue:
            parts += ['--queue']
        # Keep the Slurm expression unquoted so it expands in the job shell.
        # On hosts without device cgroups, this prevents infer-stack from
        # planning against GPUs outside the allocation.
        allow_list = self._lease_slurm_gpu_allow_list()
        if allow_list:
            parts += [allow_list]
        # Everything after `--` is the command; without it a command starting
        # with a dash is parsed as an option to `run`.
        parts += ['--']
        return ' '.join(parts)
