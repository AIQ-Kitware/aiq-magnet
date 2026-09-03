"""
kwdagger nodes that lease their own inference endpoints.

Wrapping a whole evaluation in one lease holds every model in the cohort for
the entire run, including while analysis nodes that need no model are running.
Here the lease is a property of the node: it declares which of its parameters
name endpoints, and its command renders as::

    infer-stack run --endpoint <alias> ... -- <the original command>

so an endpoint is held for the jobs that use it and no longer. Concurrency
becomes infer-stack's problem, and switching from a simulator to a real GPU is
``INFER_STACK_CATALOG``, not a code change.

The cost: with ``reclaim: stop`` a cohort with more models than GPUs reloads
weights repeatedly. Use ``reclaim: keep-warm``, where the lease bounds
entitlement rather than container lifetime.

Under Slurm the rendered command also carries an allow-list of the GPUs the job
was actually given, since infer-stack otherwise plans against every card it can
see -- which on a host without a device cgroup is all of them, not the ones this
job was allocated. See :data:`GPU_ALLOW_LIST_EXPANSION`.

Opt-in via ``--per_node_leasing``, which reaches the nodes through
:class:`LeaseSettings`. Off by default because
plenty of legitimate runs point at a server infer-stack does not manage.
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
    """
    One invocation's answer to whether nodes lease their own endpoints.

    Built from CLI arguments and applied to the DAG with :meth:`apply`. Nothing
    reads it afterwards: by the time a command renders, the node carries what
    it needs.

    This is passed configuration, so it arrives as an argument; contrast
    :data:`INSIDE_LEASE_ENVVAR`, which is a fact about the surrounding process
    that only infer-stack can state, and is therefore still read from the
    environment where it is needed.
    """

    #: Render each node's command inside its own lease. **Opt-in**, off by
    #: default because plenty of legitimate runs point at a server infer-stack
    #: does not manage -- OpenRouter, a hand-started mock, a colleague's shared
    #: vLLM -- and for those, rendering an ``infer-stack run`` prefix would turn
    #: a working card into one that fails looking up an endpoint that was never
    #: in a catalog.
    enabled: bool = False

    #: Whether to emit ``--allowed_gpus``. Unlike leasing itself this defaults
    #: ON: off Slurm the flag renders to nothing at all, and under Slurm its
    #: absence is a correctness bug. The hatch exists for a site whose Slurm
    #: reports indices that do not match the ones the container runtime sees.
    allowed_gpus: bool = True

    def apply(self, pipeline: Any) -> None:
        """Apply these invocation settings to every leasable node in a DAG."""
        for node in (getattr(pipeline, 'node_dict', None) or {}).values():
            if isinstance(node, LeaseCapability):
                node.apply_lease_settings(self)


#: Exported by ``infer-stack run``. Its presence means we are already inside
#: someone else's lease, which holds every endpoint it named, so acquiring
#: again per node is pure overhead.
INSIDE_LEASE_ENVVAR = 'INFER_STACK_LEASE_ID'

#: YAML keys owned by the leasing capability.
LEASE_NODE_SPEC_KEYS = frozenset({
    'endpoint_params',
    'lease_ttl',
    'lease_timeout',
    'lease_queue',
})




#: An unquoted shell word that becomes ``--allowed_gpus=<indices>`` inside a
#: Slurm job and disappears entirely everywhere else.
#:
#: Deferred rather than interpolated because the DAG is rendered on the submit
#: host, where no allocation exists and the value is therefore unknowable; it
#: only becomes true once the job is running. Written as one word so that when
#: neither variable is set the whole thing is an empty unquoted expansion,
#: which a shell drops from the argument list -- as opposed to
#: ``--allowed_gpus ''``, which infer-stack would see and have to interpret.
#: The odd two-part shape is what makes ``SLURM_STEP_GPUS`` a fallback rather
#: than a second flag: the first half contributes only the flag name when
#: ``SLURM_JOB_GPUS`` is set, and the second half supplies either that
#: variable's value or, only if it is unset, the step's.
#:
#: ``CUDA_VISIBLE_DEVICES`` is deliberately not in the chain. It may
#: legitimately hold GPU UUIDs (``GPU-4d888104-...``) instead of indices, and
#: infer-stack parses this value with ``int()`` per element, so a UUID there is
#: a crash rather than a narrower allow-list. The two SLURM_* variables are
#: always numeric indices.
GPU_ALLOW_LIST_EXPANSION = (
    '${SLURM_JOB_GPUS:+--allowed_gpus=}'
    '${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:+--allowed_gpus=$SLURM_STEP_GPUS}}'
)



class LeaseCapability:
    """
    Lease-specific state and command wrapping, independent of node type.

    This mixin owns no ``command`` property and performs no container wrapping.
    :class:`magnet.process_node.MagnetProcessNode` is the supported integration
    surface. Tests also compose this mixin with a bare kwdagger node to pin
    the fact that leasing itself remains independently reusable.

    Subclasses declare :attr:`endpoint_params` -- the parameter names whose
    *values* are catalog aliases. Override :meth:`resolve_lease_endpoints`
    when the alias is not the parameter value itself.

    Attributes:
        endpoint_params (tuple[str, ...]): parameter names holding aliases.
        lease_ttl (str | None): a backstop for a hard-killed job, not a budget
            -- the lease is released when the command ends. Generous on
            purpose: a TTL expiring mid-job lets another lease reclaim the GPU
            out from under it.
        lease_timeout (str | int | None): readiness wait. Must exceed a cold
            model load, which on a cold HF cache is minutes.
        lease_queue (bool): wait for capacity rather than failing when the GPUs
            are busy. On by default -- with a DAG scheduling more jobs than the
            box has GPUs, busy is the normal case.
        lease_enabled (bool): whether this node leases at all. Written by
            :meth:`LeaseSettings.apply` from the invocation; a node may also
            set it outright.
        lease_allowed_gpus (bool): whether to confine the lease to the job's
            Slurm allocation. See :data:`GPU_ALLOW_LIST_EXPANSION`.
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
        """
        Adopt an invocation's leasing settings.

        Leasing and containerization are separate invocation policies. A
        combined MAGNET node may receive both settings objects, but neither
        capability depends on the other.

        Unlike the container settings there is no "the node already declared
        one" case to respect: leasing is a property of the invocation, not of
        the step. A card cannot ask to be leased, because whether a lease can be
        acquired at all depends on the catalog the run points at.
        """
        self.lease_enabled = bool(settings.enabled)
        self.lease_allowed_gpus = bool(settings.allowed_gpus)

    def resolve_lease_endpoints(self) -> list[str]:
        """Catalog aliases this node's job needs, deduplicated, order kept.

        Empty values are dropped, so an optional model -- an extractor that
        defaults to the answerer, say -- produces no bogus lease.
        """
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
        """Bracket the command in a lease when one is needed.

        The combined :class:`magnet.process_node.MagnetProcessNode` calls this
        after rendering its execution substrate, so a lease wraps Docker rather
        than running inside it. A lease-only node passes a host command here.
        """
        if not self.leasing_is_enabled():
            return command
        names = self.resolve_lease_endpoints()
        if not names:
            return command
        return self._lease_command_prefix(names) + ' \\\n    ' + command

    def _lease_command_prefix(self, names: list[str]) -> str:
        # The prefix is shell text run later, on a host that may not be this
        # one, so this is a courtesy check, not a guarantee: it catches the
        # common case where the environment rendering the DAG is the one that
        # will run it and infer-stack was never installed (the `leasing`
        # extra). Without it the failure is a bare `infer-stack: command not
        # found` in a job log, after the DAG has been submitted.
        import shutil
        import sys
        from pathlib import Path
        beside_python = Path(sys.executable).parent / 'infer-stack'
        if shutil.which('infer-stack') is None and not beside_python.exists():
            from magnet.exceptions import MissingOptionalDependency
            raise MissingOptionalDependency(
                "per-node leasing renders an `infer-stack run` prefix, but "
                "no `infer-stack` executable is on PATH. Install it with: "
                "pip install 'aiq-magnet[leasing]'")
        # ONE --endpoint with a comma-separated list. `infer-stack run` takes a
        # single string, so repeating the flag does not accumulate -- the last
        # one wins and every other model goes unleased, which stays
        # invisible until something races for a GPU.
        parts = ['infer-stack', 'run', '--endpoint', shlex.quote(','.join(names))]
        if self.lease_ttl:
            parts += ['--ttl', str(self.lease_ttl)]
        if self.lease_timeout is not None:
            parts += ['--timeout', str(self.lease_timeout)]
        if self.lease_queue:
            parts += ['--queue']
        # Which GPUs this node may place on. Not shlex.quote'd, and that is
        # the point: it has to reach the job script as an unexpanded shell
        # word, because the allocation it names does not exist yet on the host
        # that renders this string. Quoting would hand infer-stack the literal
        # characters `${SLURM_JOB_GPUS...}`, which it parses with `int()`. See
        # GPU_ALLOW_LIST_EXPANSION for why the value has to be deferred, and
        # why CUDA_VISIBLE_DEVICES is not the variable to read it from.
        #
        # Without it every node plans against every card on the box. `aiq-gpu`
        # sets ConstrainDevices=yes but TaskPlugin=task/none, so no device
        # cgroup is ever created and `nvidia-smi -L` inside a 2-GPU allocation
        # lists all four. infer-stack takes its inventory from that list, two
        # nodes place servers on the same card, and one dies with CUDA OOM.
        allow_list = self._lease_slurm_gpu_allow_list()
        if allow_list:
            parts += [allow_list]
        # Everything after `--` is the command; without it a command starting
        # with a dash is parsed as an option to `run`.
        parts += ['--']
        return ' '.join(parts)
