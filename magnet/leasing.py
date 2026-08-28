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

Opt-in, because plenty of legitimate runs point at a server infer-stack does
not manage -- OpenRouter, a hand-started mock, a colleague's shared vLLM --
and for those an ``infer-stack run`` prefix turns a working card into one that
fails looking up an endpoint no catalog has.
"""

from __future__ import annotations

import os
import shlex

from magnet.containers import ContainerProcessNode

__all__ = ['LeasedProcessNode', 'leasing_is_enabled', 'LEASING_ENVVAR']

#: Set truthy to render each node's command inside its own lease.
LEASING_ENVVAR = 'MAGNET_PER_NODE_LEASING'

#: Exported by ``infer-stack run``. Its presence means we are already inside
#: someone else's lease, which holds every endpoint it named, so acquiring
#: again per node is pure overhead.
INSIDE_LEASE_ENVVAR = 'INFER_STACK_LEASE_ID'

_FALSEY = {'0', 'false', 'no', 'off', ''}


def leasing_is_enabled() -> bool:
    """Whether rendered commands should bracket themselves in a lease.

    Requires an explicit opt-in, and stays off inside an outer lease so the
    two styles cannot nest by accident.
    """
    explicit = os.environ.get(LEASING_ENVVAR, '')
    if explicit.strip().lower() in _FALSEY:
        return False
    return not os.environ.get(INSIDE_LEASE_ENVVAR)


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
        # Everything after `--` is the command; without it a command starting
        # with a dash is parsed as an option to `run`.
        parts += ['--']
        return ' '.join(parts)
