"""
kwdagger nodes that run their command inside a container.

Orchestration outside, work inside. MAGNET parses the card, compiles the DAG
and submits the queue on the host, because that needs the Docker socket, the
infer-stack ledger and the host filesystem. What goes in a container is each
node's command -- the process whose dependencies must be pinned and which runs
many times, on many hosts.

A node can be both leased and containerized, and the order is not arbitrary::

    test -e <output> || \\
    infer-stack run --endpoint qwen3-8b -- \\
        docker run --rm --network host ... image python -m pkg.node ...

The lease is outside because acquiring one needs the host's daemon and ledger.
The container is inside because it consumes the endpoint, and being inside
means it inherits ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` from the lease. The
cache guard stays outermost, so a node whose output exists neither leases nor
starts a container.

The repository is mounted at the same absolute path it has on the host:
kwdagger bakes absolute output paths into commands, so keeping them identical
means nothing has to be rewritten and a path in a log is one you can open.

TODO:
    Opt-in per node class, so a pipeline must inherit from
    :class:`ContainerProcessNode`. MAGNET knows the whole DAG at compile time
    and could inject the wrapper into every node of a card that asks for
    containerized execution, leaving the pipeline to describe the work and the
    card to describe where it runs.
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
        >>> from unittest import mock
        >>> with mock.patch.dict(os.environ, {'MAGNET_NODE_FORWARD_ENV': 'MY_FACTORY,MY_URL'}):
        ...     names = forwarded_env()
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
