"""
kwdagger nodes that lease their own inference endpoints.

Wrapping a whole evaluation in one lease holds every model in the cohort for
the entire run, including while the analysis nodes that need no model at all
are running. Here the lease is a property of the node: it declares which of its
parameters name endpoints, and its command renders as::

    infer-stack run --endpoint <alias> ... -- <the original command>

so an endpoint is held for the jobs that use it and no longer. Concurrency
becomes infer-stack's problem -- kwdagger may have eight jobs in flight and
infer-stack coalesces the ones wanting the same model -- and switching from a
simulator to a real GPU is `INFER_STACK_CATALOG`, not a code change.

The cost: with ``reclaim: stop`` a cohort with more models than GPUs reloads
weights repeatedly. Use ``reclaim: keep-warm``, where the lease bounds
entitlement rather than container lifetime.

Opt-in via ``--per_node_leasing``. Off by default because plenty of
legitimate runs point at a server infer-stack does not manage.
"""

from __future__ import annotations

import os
import shlex

from magnet.containers import ContainerProcessNode

__all__ = [
    'LeasedProcessNode',
    'configure',
    'leasing_is_enabled',
    'INSIDE_LEASE_ENVVAR',
]

#: Whether each node's command renders inside its own lease. **Opt-in**, set
#: from a CLI argument by :func:`configure`. Off by default because plenty of
#: legitimate runs point at a server infer-stack does not manage -- OpenRouter,
#: a hand-started mock, a colleague's shared vLLM -- and for those, rendering
#: an ``infer-stack run`` prefix would turn a working card into one that fails
#: looking up an endpoint that was never in a catalog.
_ENABLED = False


def configure(enabled: bool = False) -> bool:
    """
    Set whether nodes lease their own endpoints, and return the setting.

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
    global _ENABLED
    _ENABLED = bool(enabled)
    return _ENABLED

#: Exported by ``infer-stack run``. Its presence means we are already inside
#: someone else's lease, which already holds every endpoint it named, so
#: acquiring again per node is pure overhead.
INSIDE_LEASE_ENVVAR = 'INFER_STACK_LEASE_ID'

def leasing_is_enabled() -> bool:
    """
    Whether rendered commands should bracket themselves in a lease.

    Requires an explicit opt-in, and stays off inside an outer lease so the
    two styles cannot nest by accident.

    Returns:
        bool
    """
    if not _ENABLED:
        return False
    return not os.environ.get(INSIDE_LEASE_ENVVAR)


class LeasedProcessNode(ContainerProcessNode):
    """
    A node that acquires its endpoints for its own job.

    Also a :class:`~magnet.containers.ContainerProcessNode`, so a node that
    holds a model can equally run in a pinned image; the lease ends up
    outside the container. Both layers are independently inert until asked
    for.

    Subclasses declare :attr:`endpoint_params` -- the parameter names whose
    *values* are endpoint aliases in the catalog. Override
    :meth:`resolve_endpoints` when the alias is not the parameter value
    itself (e.g. a named model config that has to be looked up).

    Attributes:
        endpoint_params (tuple[str, ...]): parameter names holding aliases.
        lease_ttl (str | None): TTL passed to ``infer-stack run``. This is a
            backstop for a hard-killed job, not a budget -- the lease is
            released when the command ends. Sized generously on purpose: a
            TTL that expires mid-job would let another lease reclaim the GPU
            out from under it.
        lease_timeout (str | int | None): how long to wait for readiness.
            Must exceed a cold model load, which for a large model on a cold
            HF cache is minutes, not seconds.
        lease_queue (bool): wait for capacity instead of failing when the
            GPUs are busy. On by default -- with a DAG scheduling more jobs
            than the box has GPUs, "busy" is the normal case, not an error.
    """

    endpoint_params: tuple[str, ...] = ()
    lease_ttl: str | None = '8h'
    lease_timeout: str | int | None = 1800
    lease_queue: bool = True

    def resolve_endpoints(self) -> list[str]:
        """
        Catalog aliases this node's job needs, in a stable order.

        The default reads :attr:`endpoint_params` out of the node's resolved
        configuration. Values that are empty are dropped, so an optional
        model (an extractor that defaults to the answerer, say) does not
        produce a bogus lease.

        Returns:
            list[str]: deduplicated aliases, order preserved.
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
        """
        Bracket the command in a lease when one is needed.

        Called by :class:`~magnet.containers.ContainerProcessNode` *after* it
        has applied any ``docker run`` wrapper, so the lease ends up outside
        the container. That order matters: acquiring a lease needs the Docker
        daemon and the shared ledger, both of which live on the host, and
        being inside means the container inherits ``OPENAI_BASE_URL`` /
        ``OPENAI_API_KEY`` from the lease with no extra plumbing.

        Args:
            command (str): the command as built so far.

        Returns:
            str
        """
        if not leasing_is_enabled():
            return command
        names = self.resolve_endpoints()
        if not names:
            return command
        return self._lease_prefix(names) + ' \\\n    ' + command

    def _lease_prefix(self, names: list[str]) -> str:
        # ONE --endpoint with a comma-separated list. `infer-stack run` takes
        # a single string here, so repeating the flag does not accumulate --
        # the last one silently wins and every other model goes unleased.
        # That failure is invisible until something races for a GPU.
        parts = ['infer-stack', 'run', '--endpoint', shlex.quote(','.join(names))]
        if self.lease_ttl:
            parts += ['--ttl', str(self.lease_ttl)]
        if self.lease_timeout is not None:
            parts += ['--timeout', str(self.lease_timeout)]
        if self.lease_queue:
            parts += ['--queue']
        # Everything after `--` is the command; without it a command that
        # starts with a dash would be parsed as an option to `run`.
        parts += ['--']
        return ' '.join(parts)
