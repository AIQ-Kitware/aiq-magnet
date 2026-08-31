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

Opt-in via ``--per_node_leasing``. Off by default because plenty of
legitimate runs point at a server infer-stack does not manage.
"""

from __future__ import annotations

import os
import shlex

from kwdagger.yaml_pipeline import YamlProcessNode

from magnet.containers import ContainerProcessNode

__all__ = [
    'LeasedProcessNode',
    'LeasedYamlProcessNode',
    'configure',
    'leasing_is_enabled',
    'slurm_gpu_allow_list',
    'INSIDE_LEASE_ENVVAR',
    'GPU_ALLOW_LIST_EXPANSION',
]

#: Whether each node's command renders inside its own lease. **Opt-in**, set
#: from a CLI argument by :func:`configure`. Off by default because plenty of
#: legitimate runs point at a server infer-stack does not manage -- OpenRouter,
#: a hand-started mock, a colleague's shared vLLM -- and for those, rendering
#: an ``infer-stack run`` prefix would turn a working card into one that fails
#: looking up an endpoint that was never in a catalog.
_ENABLED = False

#: Whether to emit ``--allowed_gpus``. Unlike leasing itself this defaults ON:
#: off Slurm the flag renders to nothing at all, and under Slurm its absence is
#: a correctness bug. The hatch exists for a site whose Slurm reports indices
#: that do not match the ones the container runtime sees.
_ALLOWED_GPUS = True


def configure(enabled: bool = False, allowed_gpus: bool = True) -> bool:
    """
    Set whether nodes lease their own endpoints, and return the setting.

    Args:
        enabled: render each node's command inside its own lease.
        allowed_gpus: confine a leased node to its Slurm allocation.

    Called from the CLI before scheduling. This is passed configuration, so it
    arrives as an argument rather than from the environment; contrast
    :data:`INSIDE_LEASE_ENVVAR`, which is a fact about the surrounding process
    that only infer-stack can state.

    Example:
        >>> from magnet.leasing import configure, leasing_is_enabled
        >>> configure(True)
        True
        >>> leasing_is_enabled()
        True
        >>> configure(False)
        False
    """
    global _ENABLED, _ALLOWED_GPUS
    _ENABLED = bool(enabled)
    _ALLOWED_GPUS = bool(allowed_gpus)
    return _ENABLED

#: Exported by ``infer-stack run``. Its presence means we are already inside
#: someone else's lease, which holds every endpoint it named, so acquiring
#: again per node is pure overhead.
INSIDE_LEASE_ENVVAR = 'INFER_STACK_LEASE_ID'

def leasing_is_enabled() -> bool:
    """Whether rendered commands should bracket themselves in a lease.

    Requires an explicit opt-in, and stays off inside an outer lease so the
    two styles cannot nest by accident.
    """
    if not _ENABLED:
        return False
    return not os.environ.get(INSIDE_LEASE_ENVVAR)


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


def slurm_gpu_allow_list() -> str:
    """
    Shell text confining infer-stack to the GPUs this job was allocated.

    Returns:
        str: :data:`GPU_ALLOW_LIST_EXPANSION`, or empty when configured off.

    Example:
        >>> from magnet.leasing import configure, slurm_gpu_allow_list
        >>> configure(True, allowed_gpus=False)
        True
        >>> slurm_gpu_allow_list()
        ''
        >>> _ = configure(True)
        >>> slurm_gpu_allow_list().startswith('${SLURM_JOB_GPUS')
        True
        >>> _ = configure(False)
    """
    return GPU_ALLOW_LIST_EXPANSION if _ALLOWED_GPUS else ''


class LeasedProcessNode(ContainerProcessNode):
    """
    A node that acquires its endpoints for its own job.

    Also a :class:`~magnet.containers.ContainerProcessNode`, so a node holding
    a model can equally run in a pinned image; the lease ends up outside the
    container.

    Subclasses declare :attr:`endpoint_params` -- the parameter names whose
    *values* are catalog aliases. Override :meth:`resolve_endpoints` when the
    alias is not the parameter value itself.

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
    """

    endpoint_params: tuple[str, ...] = ()
    lease_ttl: str | None = '8h'
    lease_timeout: str | int | None = 1800
    lease_queue: bool = True

    def resolve_endpoints(self) -> list[str]:
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

    def _wrap_command(self, command: str) -> str:
        """Bracket the command in a lease when one is needed.

        Called by :class:`~magnet.containers.ContainerProcessNode` *after* it
        applies any ``docker run`` wrapper, so the lease ends up outside the
        container. That order matters: acquiring a lease needs the Docker
        daemon and the shared ledger, both on the host, and being inside means
        the container inherits OPENAI_BASE_URL / OPENAI_API_KEY from the lease
        with no extra plumbing.
        """
        if not leasing_is_enabled():
            return command
        names = self.resolve_endpoints()
        if not names:
            return command
        return self._lease_prefix(names) + ' \\\n    ' + command

    def _lease_prefix(self, names: list[str]) -> str:
        # ONE --endpoint with a comma-separated list. `infer-stack run` takes a
        # single string, so repeating the flag does not accumulate -- the last
        # one silently wins and every other model goes unleased, which stays
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
        allow_list = slurm_gpu_allow_list()
        if allow_list:
            parts += [allow_list]
        # Everything after `--` is the command; without it a command starting
        # with a dash is parsed as an option to `run`.
        parts += ['--']
        return ' '.join(parts)


class LeasedYamlProcessNode(LeasedProcessNode, YamlProcessNode):
    """
    A leased, containerized node that a card can declare in YAML.

    The leasing counterpart of
    :class:`~magnet.containers.ContainerYamlProcessNode`, and the same
    reasoning: a declarative card's nodes are
    :class:`~kwdagger.yaml_pipeline.YamlProcessNode`, so they inherited neither
    the container wrapper nor the lease.

    :attr:`~LeasedProcessNode.endpoint_params` names the parameters whose
    values are catalog aliases. kwdagger's node-spec allow-list has no key for
    it, so a card that leases still declares a small subclass::

        class MyInfer(LeasedYamlProcessNode):
            endpoint_params = ('model_id',)

    and names *that* in ``class``. Everything else -- the command, the paths,
    the result readout -- stays data in the card.
    """


