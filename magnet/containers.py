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
from typing import Any
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
    'LEASE_ENV',
    'DEFAULT_CAPTURED_ENV',
]

#: Image to run node commands in. Unset => run on the host, as before.
IMAGE_ENVVAR = 'MAGNET_NODE_IMAGE'

#: Colon-separated host paths to bind-mount at their own absolute paths.
#: Normally one entry: the repository root.
MOUNTS_ENVVAR = 'MAGNET_NODE_MOUNTS'

#: Extra ``docker run`` arguments, shell-split. The escape hatch for what
#: varies by host: GPU reservations, an alternate network, a credential mount.
DOCKER_ARGS_ENVVAR = 'MAGNET_NODE_DOCKER_ARGS'

#: Extra names to forward, on top of :data:`DEFAULT_FORWARDED_ENV`. How a
#: pipeline's own configuration reaches its nodes without MAGNET enumerating it.
FORWARD_ENV_ENVVAR = 'MAGNET_NODE_FORWARD_ENV'

#: Supplied at job time by a surrounding lease, so forwarded BY NAME. Capturing
#: a value would freeze the orchestrator's shell over the endpoint actually
#: leased.
LEASE_ENV = (
    'OPENAI_BASE_URL',
    'OPENAI_API_KEY',
)

#: Exist at render time and are not recreated later, so their VALUES are
#: captured. PYTHONPATH is the one that matters: bare, it arrives empty in a
#: tmux worker that never inherited it and every import in the node fails.
DEFAULT_CAPTURED_ENV = (
    'PYTHONPATH',
    'HF_TOKEN',
    'HF_HOME',
    'TRANSFORMERS_OFFLINE',
    'HF_HUB_OFFLINE',
)

#: Every variable magnet forwards without being told to, in a stable order.
DEFAULT_FORWARDED_ENV = LEASE_ENV + DEFAULT_CAPTURED_ENV


def forwarded_env() -> list[str]:
    """Names to forward into the container, in a stable order.

    Example:
        >>> import os
        >>> from unittest import mock
        >>> with mock.patch.dict(os.environ, {'MAGNET_NODE_FORWARD_ENV': 'MY_FACTORY,MY_URL'}):
        ...     names = forwarded_env()
        >>> names[0], names[-2:]
        ('OPENAI_BASE_URL', ['MY_FACTORY', 'MY_URL'])
    """
    names = list(DEFAULT_FORWARDED_ENV)
    for chunk in _env_name_list(os.environ.get(FORWARD_ENV_ENVVAR, '')):
        if chunk not in names:
            names.append(chunk)
    return names


def node_image(node: Any = None) -> str:
    """The image for this node: its own declaration, else the process-wide one."""
    declared = getattr(node, 'container_image', None)
    if declared:
        return str(declared).strip()
    return os.environ.get(IMAGE_ENVVAR, '').strip()


def node_mounts(node: Any = None) -> list[str]:
    """Host paths to bind-mount at their own absolute paths."""
    declared = getattr(node, 'container_mounts', None)
    if declared:
        raw = declared if isinstance(declared, (list, tuple)) else [declared]
    else:
        raw = os.environ.get(MOUNTS_ENVVAR, '').split(':')
    return [str(m).strip() for m in raw if str(m).strip()]


def declared_env(node: Any = None) -> dict:
    """
    Render-time variables and their values, in a stable order.

    Values are captured here rather than forwarded by name because the
    environment that will run the command is not this one; see the note in
    :func:`container_prefix`.

    A declared name with no value keeps a bare ``-e NAME`` (value None in the
    result): a name that is not set yet can only be a job-time value.
    """
    names: list[str] = list(DEFAULT_CAPTURED_ENV)
    for source in (getattr(node, 'container_forward_env', None) or (),
                   _env_name_list(os.environ.get(FORWARD_ENV_ENVVAR, ''))):
        for name in source:
            name = str(name).strip()
            if name and name not in names and name not in LEASE_ENV:
                names.append(name)

    resolved: dict = {name: os.environ.get(name) or None for name in names}
    # An explicit mapping on the node wins over the environment.
    for name, value in (getattr(node, 'container_env', None) or {}).items():
        resolved[str(name)] = str(value)
    return resolved


def _env_name_list(raw: str) -> list[str]:
    return [c.strip() for c in str(raw).replace(',', ':').split(':') if c.strip()]


def containerization_is_enabled(node: Any = None) -> bool:
    """Whether node commands are wrapped in ``docker run``: an image is named."""
    return bool(node_image(node))


def container_prefix(node: Any = None) -> str:
    """The ``docker run`` invocation node commands are appended to.

    Args:
        node: the node being rendered, when it declares its own image, mounts
            or environment; otherwise the process-wide settings apply.
    """
    image = node_image(node)
    parts = [
        'docker', 'run', '--rm',
        # The lease exports a 127.0.0.1:<port> URL. Host networking makes that
        # URL true inside the container too, instead of rewriting it to a
        # compose-network DNS name only some deployments have.
        '--network', 'host',
        # Otherwise every artifact comes out root-owned and the next
        # host-side step cannot delete it.
        '--user', f'{os.getuid()}:{os.getgid()}',
    ]
    for mount in node_mounts(node):
        parts += ['-v', f'{mount}:{mount}']
    parts += [
        # Node configs carry paths relative to the job's cwd, so it must match.
        '-w', '"$PWD"',
        # A non-root uid has no home; anything touching a cache dir fails.
        '-e', 'HOME=/tmp',
    ]
    # Two kinds of variable, split by WHEN the value exists.
    #
    # Job-time (LEASE_ENV): `infer-stack run` writes these into the wrapped
    # command's environment long after this string is rendered, so they stay a
    # bare `-e NAME`. A captured value would freeze the orchestrator's shell
    # over the endpoint actually leased.
    #
    # Render-time: values exist now and are not recreated later. A tmux worker
    # inherits the tmux server's environment, not the orchestrator's, so a bare
    # `-e NAME` forwards nothing and does it silently -- that cost a full run
    # when OC_BACKEND_FACTORY vanished and every shard routed to a provider it
    # had no key for.
    for name in LEASE_ENV:
        parts += ['-e', name]
    for name, value in declared_env(node).items():
        if value is None:
            parts += ['-e', name]
        else:
            parts += ['-e', shlex.quote(f'{name}={value}')]
    parts += shlex.split(os.environ.get(DOCKER_ARGS_ENVVAR, ''))
    parts.append(image)
    return ' '.join(parts)


class ContainerProcessNode(kwdagger.ProcessNode):
    """
    A :class:`kwdagger.ProcessNode` whose command runs in a container.

    Inert unless an image is named, by the node or by :data:`IMAGE_ENVVAR`, so
    the same pipeline runs on the host during development and in a pinned image
    for a real run.

    A node may declare its own container settings instead of inheriting the
    process-wide MAGNET_NODE_* variables, which are a property of one
    invocation rather than of the step.
    """

    #: Image for this node's command. None => the process-wide setting.
    container_image: Any = None
    #: Host paths bind-mounted at their own absolute paths.
    container_mounts: Any = None
    #: Render-time variables, name -> value, captured into the command.
    container_env: Any = None
    #: Names whose values are captured from the environment at render time.
    container_forward_env: Any = ()

    def _wrap_command(self, command: str) -> str:
        """Hook for subclasses that add another layer (see
        :class:`magnet.leasing.LeasedProcessNode`)."""
        return command

    @property
    def command(self) -> str:
        base = kwdagger.ProcessNode.command.fget(self)  # type: ignore[attr-defined]
        if containerization_is_enabled(self):
            base = container_prefix(self) + ' \\\n    ' + base
        return self._wrap_command(base)
