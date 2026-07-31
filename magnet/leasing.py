"""
kwdagger nodes that lease their own inference endpoints.

The obvious way to run a card against a served model is to wrap the whole
evaluation in one lease::

    infer-stack run --endpoint cohort -- python -m magnet.evaluation card.yaml

That works, and it is wrong as soon as the models are real.  It holds every
model in the cohort for the entire run -- including while the cheap
aggregation and analysis nodes run, which need no model at all -- so an
eight-model cohort occupies eight GPUs from the first token to the last
statistic.  It also means the DAG cannot say anything about what it needs:
the lease lives in a shell script beside the pipeline instead of in it.

Here the lease is a property of the node.  A node declares which of its
parameters name endpoints, and its rendered command becomes::

    infer-stack run --endpoint <alias> ... -- <the original command>

so the endpoint is acquired when that job starts and released when it ends,
whatever else the DAG is doing.  Three things follow:

* **Utilization is correct by construction.** A GPU is held for the jobs
  that use it and no longer. Nodes that touch no model never appear to the
  scheduler as if they did.
* **Concurrency is infer-stack's problem, not the card's.** kwdagger may
  have ``--jobs 8`` in flight; infer-stack coalesces jobs that want the same
  model onto one deployment and queues the ones that do not fit. Neither
  side has to know the other's limits.
* **A rehearsal and a real run are the same pipeline.** The node names an
  endpoint *alias*; which server that resolves to is the catalog's business.
  Switching from a simulator to a real GPU is `INFER_STACK_CATALOG`, not a
  code change.

The cost is honest and worth stating: with ``reclaim: stop`` a model is torn
down between jobs, so a cohort with more models than GPUs will reload
weights repeatedly. Real catalogs should use ``reclaim: keep-warm``, which
leaves a released deployment up until something else needs the GPU -- the
lease then bounds *entitlement*, not container lifetime.

Leasing is **opt-in** (``MAGNET_PER_NODE_LEASING=1``). Without it a
:class:`LeasedProcessNode` renders exactly the command a plain
:class:`kwdagger.ProcessNode` would, because plenty of legitimate runs point
at a server infer-stack does not manage.

Example:
    >>> import os, kwdagger
    >>> os.environ['MAGNET_PER_NODE_LEASING'] = '1'
    >>> from magnet.leasing import LeasedProcessNode
    >>> class Infer(LeasedProcessNode):
    ...     name = 'infer'
    ...     executable = 'python -m mypkg.infer'
    ...     endpoint_params = ('model_id',)
    ...     algo_params = {'model_id': None}
    >>> node = Infer()
    >>> node.configure({'model_id': 'Qwen/Qwen3-8B'})
    >>> print(node.command.split('--')[0].strip())
    infer-stack run
"""

from __future__ import annotations

import os
import shlex

import kwdagger

__all__ = ['LeasedProcessNode', 'leasing_is_enabled', 'LEASING_ENVVAR']

#: Set truthy to render each node's command inside its own lease. **Opt-in.**
#: Off by default because plenty of legitimate runs point at a server
#: infer-stack does not manage -- OpenRouter, a hand-started mock, a
#: colleague's shared vLLM -- and for those, rendering an ``infer-stack run``
#: prefix would turn a working card into one that fails looking up an
#: endpoint that was never in a catalog.
LEASING_ENVVAR = 'MAGNET_PER_NODE_LEASING'

#: Exported by ``infer-stack run``. Its presence means we are already inside
#: someone else's lease, which already holds every endpoint it named, so
#: acquiring again per node is pure overhead.
INSIDE_LEASE_ENVVAR = 'INFER_STACK_LEASE_ID'

_FALSEY = {'0', 'false', 'no', 'off', ''}


def leasing_is_enabled() -> bool:
    """
    Whether rendered commands should bracket themselves in a lease.

    Requires an explicit opt-in, and stays off inside an outer lease so the
    two styles cannot nest by accident.

    Returns:
        bool
    """
    explicit = os.environ.get(LEASING_ENVVAR, '')
    if explicit.strip().lower() in _FALSEY:
        return False
    return not os.environ.get(INSIDE_LEASE_ENVVAR)


class LeasedProcessNode(kwdagger.ProcessNode):
    """
    A :class:`kwdagger.ProcessNode` that acquires its endpoints for its own job.

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

    @property
    def command(self) -> str:
        """
        The node's command, bracketed in a lease when one is needed.

        Returns:
            str
        """
        base = super().command
        if not leasing_is_enabled():
            return base
        names = self.resolve_endpoints()
        if not names:
            return base
        return self._lease_prefix(names) + ' \\\n    ' + base

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
