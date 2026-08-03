"""
kwdagger nodes that run their command inside a container.

MAGNET itself stays on the host: it parses the card, compiles the DAG,
resolves the gather edges and submits the queue.  What goes in a container
is each *node's* command -- the process that actually does the work.  That
split is deliberate:

* The orchestrator needs the Docker socket, the infer-stack ledger and the
  host filesystem. Putting it in a container means either nesting Docker or
  handing a container control of the host's daemon.
* The node is the thing whose dependencies must be pinned to be
  reproducible. It is also the thing that runs many times, on many hosts,
  possibly under Slurm.

So the boundary is: **orchestration outside, work inside.**

Layering with the lease
-----------------------

A node can be both leased and containerized, and the order is not
arbitrary::

    test -e <output> || \\
    infer-stack run --endpoint qwen3-8b -- \\
        docker run --rm --network host ... image python -m pkg.node ...

The lease is *outside* the container because acquiring one needs the Docker
daemon and the shared ledger, both of which live on the host. The container
is *inside* because it is the thing that consumes the endpoint -- and being
inside means it inherits ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` from the
lease that wraps it, with no extra plumbing.

The cache guard stays outermost, so a node whose output already exists
neither leases nor starts a container.

Mounting
--------

The repository is mounted at **the same absolute path it has on the host**.
kwdagger bakes absolute output paths into every command, and node configs
carry paths relative to the run's working directory; keeping the paths
identical means none of that has to be rewritten, and a path in a log is a
path you can open. ``PYTHONPATH`` then points at the mounted sources, which
take precedence over any copy baked into the image -- so editing a node does
not mean rebuilding.

TODO:
    This is opt-in per node class, which means a pipeline has to inherit from
    :class:`ContainerProcessNode` to get it. MAGNET knows the whole DAG at
    compile time and could inject the wrapper into *every* node of a card
    that asks for containerized execution, without the pipeline mentioning
    containers at all -- the pipeline would then describe the work and the
    card would describe where it runs. That is the right shape; this is the
    hard-coded version of it.

Example:
    >>> import os
    >>> os.environ['MAGNET_NODE_IMAGE'] = 'aiq-eval-node:latest'
    >>> os.environ['MAGNET_NODE_MOUNTS'] = '/repo'
    >>> from magnet.containers import ContainerProcessNode
    >>> class Work(ContainerProcessNode):
    ...     name = 'work'
    ...     executable = 'python -m pkg.work'
    >>> node = Work()
    >>> node.configure({})
    >>> prefix = node.command.split(' \\\\\\n')[0]
    >>> prefix.startswith('docker run --rm --network host')
    True
    >>> '-v /repo:/repo' in prefix and prefix.endswith('aiq-eval-node:latest')
    True
"""

from __future__ import annotations

import os
import shlex

import kwdagger

__all__ = [
    'ContainerProcessNode',
    'containerization_is_enabled',
    'container_prefix',
    'forwarded_env',
    'IMAGE_ENVVAR',
    'MOUNTS_ENVVAR',
    'FORWARD_ENV_ENVVAR',
]

#: Image to run node commands in. Unset => run on the host, as before.
IMAGE_ENVVAR = 'MAGNET_NODE_IMAGE'

#: Colon-separated host paths to bind-mount at their own absolute paths.
#: Normally one entry: the repository root.
MOUNTS_ENVVAR = 'MAGNET_NODE_MOUNTS'

#: Extra ``docker run`` arguments, split with shell quoting. An escape hatch
#: for the things that vary by host and should not be guessed here -- GPU
#: reservations, an alternate network, a registry credential mount.
DOCKER_ARGS_ENVVAR = 'MAGNET_NODE_DOCKER_ARGS'

#: Colon- or comma-separated extra variable names to forward, on top of
#: :data:`DEFAULT_FORWARDED_ENV`. This is how a pipeline's own configuration
#: reaches its nodes: MAGNET has no business knowing what those variables are
#: called, so it does not enumerate them.
FORWARD_ENV_ENVVAR = 'MAGNET_NODE_FORWARD_ENV'

#: Variables forwarded into every containerized node, by name -- so the value
#: is read at job time rather than baked into a command string rendered much
#: earlier. The OPENAI_* pair is what a surrounding lease exports; the rest
#: are generic runtime settings, not anything specific to one evaluation.
DEFAULT_FORWARDED_ENV = (
    'OPENAI_BASE_URL',
    'OPENAI_API_KEY',
    'PYTHONPATH',
    'HF_TOKEN',
    'HF_HOME',
    'TRANSFORMERS_OFFLINE',
    'HF_HUB_OFFLINE',
)


def forwarded_env() -> list[str]:
    """
    Variable names to forward into the container, in a stable order.

    Returns:
        list[str]: :data:`DEFAULT_FORWARDED_ENV` followed by whatever
            :data:`FORWARD_ENV_ENVVAR` adds, deduplicated.

    Example:
        >>> import os
        >>> os.environ['MAGNET_NODE_FORWARD_ENV'] = 'MY_FACTORY,MY_URL'
        >>> names = forwarded_env()
        >>> names[0], names[-2:]
        ('OPENAI_BASE_URL', ['MY_FACTORY', 'MY_URL'])
    """
    names = list(DEFAULT_FORWARDED_ENV)
    raw = os.environ.get(FORWARD_ENV_ENVVAR, '')
    for chunk in raw.replace(',', ':').split(':'):
        chunk = chunk.strip()
        if chunk and chunk not in names:
            names.append(chunk)
    return names


def containerization_is_enabled() -> bool:
    """
    Whether node commands should be wrapped in ``docker run``.

    Returns:
        bool: true when :data:`IMAGE_ENVVAR` names an image.
    """
    return bool(os.environ.get(IMAGE_ENVVAR, '').strip())


def container_prefix() -> str:
    """
    The ``docker run`` invocation that node commands are appended to.

    Returns:
        str: everything up to and including the image name.
    """
    image = os.environ.get(IMAGE_ENVVAR, '').strip()
    parts = [
        'docker', 'run', '--rm',
        # The leased endpoint is reachable at 127.0.0.1:<gateway port> on the
        # host, and that is the URL the lease exports. Host networking makes
        # the exported URL true inside the container too, rather than having
        # to rewrite it to a compose-network DNS name that only some
        # deployments have.
        '--network', 'host',
        # Without this every artifact a node writes into the mounted run
        # directory comes out root-owned, and the next host-side step (or the
        # user) cannot delete it.
        '--user', f'{os.getuid()}:{os.getgid()}',
    ]
    for mount in os.environ.get(MOUNTS_ENVVAR, '').split(':'):
        mount = mount.strip()
        if mount:
            parts += ['-v', f'{mount}:{mount}']
    parts += [
        # Same cwd as the host job, resolved at job time. Node configs carry
        # paths relative to it (e.g. ./data/...), so it has to match.
        '-w', '"$PWD"',
        # A non-root uid has no home in the image; anything that touches a
        # cache directory (matplotlib, huggingface) fails without this.
        '-e', 'HOME=/tmp',
    ]
    for name in forwarded_env():
        parts += ['-e', name]
    parts += shlex.split(os.environ.get(DOCKER_ARGS_ENVVAR, ''))
    parts.append(image)
    return ' '.join(parts)


class ContainerProcessNode(kwdagger.ProcessNode):
    """
    A :class:`kwdagger.ProcessNode` whose command runs in a container.

    Inert unless :data:`IMAGE_ENVVAR` is set, so the same pipeline runs on
    the host during development and in a pinned image for a real run.
    """

    def _wrap_command(self, command: str) -> str:
        """Hook for subclasses that add another layer (see
        :class:`magnet.leasing.LeasedProcessNode`)."""
        return command

    @property
    def command(self) -> str:
        base = kwdagger.ProcessNode.command.fget(self)  # type: ignore[attr-defined]
        if containerization_is_enabled():
            base = container_prefix() + ' \\\n    ' + base
        return self._wrap_command(base)
