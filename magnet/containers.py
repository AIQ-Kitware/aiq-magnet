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

import dataclasses

import kwdagger

__all__ = [
    'ContainerProcessNode',
    'ContainerSettings',
    'configure',
    'current_settings',
    'containerization_is_enabled',
    'container_prefix',
    'forwarded_env',
    'LEASE_ENV',
    'DEFAULT_CAPTURED_ENV',
]

#: Process-wide container settings. Empty by default, which is what makes an
#: uncontainerized run the same path with nothing prepended rather than a
#: fallback. Populated from CLI arguments by :func:`configure`; a node that
#: declares its own image, mounts or environment still wins over it.
#:
#: This is configuration a caller passes in, so it arrives as a CLI argument.
#: It used to be read from MAGNET_NODE_* environment variables, which hid where
#: a value came from and could not be seen in the record of an invocation.


@dataclasses.dataclass(frozen=True)
class ContainerSettings:
    """What to run node commands in, when the node does not say."""

    #: Image to run node commands in. Empty => run on the host.
    image: str = ''

    #: Host paths to bind-mount at their own absolute paths. Normally one
    #: entry: the repository root.
    mounts: tuple[str, ...] = ()

    #: Extra ``docker run`` arguments. An escape hatch for the things that vary
    #: by host and should not be guessed here -- GPU reservations, an alternate
    #: network, a registry credential mount.
    docker_args: str = ''

    #: Extra variable names to forward, on top of :data:`DEFAULT_FORWARDED_ENV`.
    #: This is how a pipeline's own configuration reaches its nodes: MAGNET has
    #: no business knowing what those variables are called.
    forward_env: tuple[str, ...] = ()


_SETTINGS = ContainerSettings()


def configure(
    image: str = '',
    mounts: Any = (),
    docker_args: str = '',
    forward_env: Any = (),
) -> ContainerSettings:
    """
    Set the process-wide container settings, and return them.

    Called once from the CLI before a pipeline is scheduled. Node commands are
    rendered in this process, so a process-wide value is enough to reach them;
    what matters is that it came from an argument rather than the ambient
    environment.

    Example:
        >>> from magnet.containers import configure, current_settings
        >>> before = current_settings()
        >>> configure(image='magnet:latest', mounts='/repo')
        ContainerSettings(image='magnet:latest', mounts=('/repo',), ...)
        >>> current_settings().image
        'magnet:latest'
        >>> _ = configure(**{f.name: getattr(before, f.name)
        ...                  for f in dataclasses.fields(before)})
    """
    global _SETTINGS
    _SETTINGS = ContainerSettings(
        image=str(image or '').strip(),
        mounts=tuple(_coerce_name_list(mounts)),
        docker_args=str(docker_args or ''),
        forward_env=tuple(_coerce_name_list(forward_env)),
    )
    return _SETTINGS


def current_settings() -> ContainerSettings:
    """The container settings in effect for this process."""
    return _SETTINGS


def _coerce_name_list(raw: Any) -> list[str]:
    """Accept a list, or one colon/comma separated string."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = raw
    else:
        items = str(raw).replace(',', ':').split(':')
    return [str(item).strip() for item in items if str(item).strip()]


#: Variables a surrounding lease supplies at job time: ``infer-stack run``
#: writes them into the environment of the command it wraps, long after this
#: string is rendered. Forwarded by name so docker reads them then; capturing a
#: value here would freeze the orchestrator's shell over the endpoint the job
#: actually leased.
LEASE_ENV = (
    'OPENAI_BASE_URL',
    'OPENAI_API_KEY',
)

#: Variables that exist at render time and are not recreated later, so their
#: values are captured. PYTHONPATH is the one that matters: left as a bare name
#: it arrives empty in a cmd_queue tmux worker that did not inherit it, and
#: every import in the node fails.
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
    """
    Variable names to forward into the container, in a stable order.

    Returns:
        list[str]: :data:`DEFAULT_FORWARDED_ENV` followed by whatever
            :attr:`ContainerSettings.forward_env` adds, deduplicated.

    Example:
        >>> from magnet.containers import configure, forwarded_env
        >>> before = configure(forward_env='MY_FACTORY,MY_URL')
        >>> names = forwarded_env()
        >>> names[0], names[-2:]
        ('OPENAI_BASE_URL', ['MY_FACTORY', 'MY_URL'])
        >>> _ = configure()
    """
    names = list(DEFAULT_FORWARDED_ENV)
    for chunk in current_settings().forward_env:
        if chunk not in names:
            names.append(chunk)
    return names


def node_image(node: Any = None) -> str:
    """The image for this node: its own declaration, else the process-wide one."""
    declared = getattr(node, 'container_image', None)
    if declared:
        return str(declared).strip()
    return current_settings().image


def node_mounts(node: Any = None) -> list[str]:
    """Host paths to bind-mount at their own absolute paths."""
    declared = getattr(node, 'container_mounts', None)
    if declared:
        return _coerce_name_list(declared)
    return list(current_settings().mounts)


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
                   current_settings().forward_env):
        for name in source:
            name = str(name).strip()
            if name and name not in names and name not in LEASE_ENV:
                names.append(name)

    resolved: dict = {name: os.environ.get(name) or None for name in names}
    # An explicit mapping on the node wins over anything read from the
    # environment.
    for name, value in (getattr(node, 'container_env', None) or {}).items():
        resolved[str(name)] = str(value)
    return resolved


def containerization_is_enabled(node: Any = None) -> bool:
    """
    Whether node commands should be wrapped in ``docker run``.

    Returns:
        bool: true when the node or the process settings name an image.
    """
    return bool(node_image(node))


def container_prefix(node: Any = None) -> str:
    """
    The ``docker run`` invocation that node commands are appended to.

    Args:
        node: the node being rendered, when it declares its own image, mounts
            or environment. Falls back to the process-wide
            :class:`ContainerSettings`.

    Returns:
        str: everything up to and including the image name.
    """
    image = node_image(node)
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
    for mount in node_mounts(node):
        parts += ['-v', f'{mount}:{mount}']
    parts += [
        # Same cwd as the host job, resolved at job time. Node configs carry
        # paths relative to it (e.g. ./data/...), so it has to match.
        '-w', '"$PWD"',
        # A non-root uid has no home in the image; anything that touches a
        # cache directory (matplotlib, huggingface) fails without this.
        '-e', 'HOME=/tmp',
    ]
    # Two kinds of variable, distinguished by when the value exists.
    #
    # Job-time (LEASE_ENV): supplied by whatever wraps the command at
    # execution; `infer-stack run` writes OPENAI_BASE_URL / OPENAI_API_KEY into
    # the environment of the command it wraps. These stay a bare `-e NAME` so
    # docker reads them then. Baking a value here would freeze whatever the
    # orchestrator happened to have, including nothing, over the endpoint
    # actually leased.
    #
    # Render-time (DEFAULT_CAPTURED_ENV and caller-declared): the pipeline's
    # own configuration, which exists now and is not recreated later. A bare
    # `-e NAME` reads the value when `docker run` executes, in a cmd_queue tmux
    # worker; a session created against an already-running tmux server inherits
    # that server's environment, not the orchestrator's. The variable is absent
    # by job time and `-e NAME` forwards nothing, silently: the container
    # starts, the setting is missing, the node falls back to a default. That
    # cost a full run when OC_BACKEND_FACTORY vanished this way and every shard
    # routed to a provider it had no key for.
    for name in LEASE_ENV:
        parts += ['-e', name]
    for name, value in declared_env(node).items():
        if value is None:
            parts += ['-e', name]
        else:
            parts += ['-e', shlex.quote(f'{name}={value}')]
    parts += shlex.split(current_settings().docker_args)
    parts.append(image)
    return ' '.join(parts)


class ContainerProcessNode(kwdagger.ProcessNode):
    """
    A :class:`kwdagger.ProcessNode` whose command runs in a container.

    Inert unless an image is named, by the node or by :func:`configure`, so
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
