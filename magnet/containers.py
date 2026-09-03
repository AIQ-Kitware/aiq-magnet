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

Containerization is an independent execution capability. The mixin here
contains only container state and settings; it does not inherit or know
about leasing. :class:`magnet.process_node.MagnetProcessNode` is the normal
integration surface that composes this capability with leasing.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import sys
from collections.abc import Mapping
from typing import Any


__all__ = [
    'ContainerCapability',
    'ContainerSettings',
    'apply_settings',
    'is_container_capable',
    'containerization_is_enabled',
    'host_interpreter',
    'container_prefix',
    'render_container_command',
]


@dataclasses.dataclass(frozen=True)
class ContainerSettings:
    """
    What to run node commands in, for nodes that do not say.

    One invocation's answer to "where does this run", built from CLI arguments
    and handed to :func:`apply_settings`, which writes it onto the nodes that
    did not declare their own. Nothing reads it afterwards: by the time a
    command renders, every value it needs is on the node.

    Empty by default, which is what makes an uncontainerized run the same path
    with nothing prepended rather than a fallback.
    """

    #: Image to run node commands in. Empty => run on the host.
    image: str = ''

    #: Host paths to bind-mount at their own absolute paths. Normally one
    #: entry: the repository root.
    mounts: tuple[str, ...] = ()

    #: Fixed environment values to put in every container unless a node
    #: overrides the same name. Useful when the image's import path or another
    #: execution setting must differ from the orchestrating host.
    env: dict[str, str] = dataclasses.field(default_factory=dict)

    #: Extra ``docker run`` arguments. An escape hatch for the things that vary
    #: by host and should not be guessed here -- GPU reservations, an alternate
    #: network, a registry credential mount.
    docker_args: str = ''

    #: Extra variable names to capture in addition to the node-class defaults.
    #: This is how a pipeline's own configuration reaches its nodes: MAGNET has
    #: no business knowing what those variables are called.
    forward_env: tuple[str, ...] = ()

    def apply(self, pipeline: Any) -> None:
        """Apply these invocation settings to every container-capable node."""
        apply_settings(pipeline, self)

    @classmethod
    def coerce(
        cls,
        image: str = '',
        mounts: Any = (),
        env: Any = None,
        docker_args: str = '',
        forward_env: Any = (),
    ) -> 'ContainerSettings':
        """
        Build settings from CLI argument values.

        Example:
            >>> from magnet.containers import ContainerSettings
            >>> ContainerSettings.coerce(image=' magnet:latest ',
            ...                          mounts='/repo')
            ContainerSettings(image='magnet:latest', mounts=('/repo',), ...)
        """
        return cls(
            image=str(image or '').strip(),
            mounts=tuple(_coerce_name_list(mounts)),
            env=_coerce_env_map(env),
            docker_args=str(docker_args or ''),
            forward_env=tuple(_coerce_name_list(forward_env)),
        )


def apply_settings(pipeline: Any, settings: ContainerSettings) -> None:
    """
    Write an invocation's settings onto the nodes that did not declare
    their own.

    Called once, after the DAG is built and before anything renders a command.
    This is the whole reason a node can be configured at all: kwdagger owns node
    construction -- for a declarative card it is kwdagger's YAML loader that
    instantiates the class the card names -- so MAGNET has no constructor to
    pass anything to, and ``command`` is a property that takes no arguments. The
    DAG is the one place MAGNET holds every node, so it is where configuration
    is applied.

    Nodes that are not containerizable are skipped rather than refused; naming
    them is :func:`magnet._kwdagger._check_container_settings_apply`'s job.

    Args:
        pipeline: the built pipeline whose nodes will render commands.
        settings: this invocation's defaults.

    Example:
        >>> from kwdagger.pipeline import coerce_pipeline
        >>> from magnet.containers import ContainerSettings, apply_settings
        >>> spec = {'nodes': {'work': {
        ...     'class': 'magnet.process_node.MagnetProcessNode',
        ...     'executable': 'python -m pkg.work',
        ...     'out_paths': {'results_fpath': 'results.json'}}}}
        >>> pipeline = coerce_pipeline(spec)
        >>> settings = ContainerSettings.coerce(image='magnet:latest')
        >>> apply_settings(pipeline, settings)
        >>> pipeline.node_dict['work'].container_image
        'magnet:latest'
    """
    for node in (getattr(pipeline, 'node_dict', None) or {}).values():
        if isinstance(node, ContainerCapability):
            node.apply_container_settings(settings)


def _coerce_env_map(raw: Any) -> dict[str, str]:
    """Accept a mapping or a JSON object of fixed container env values."""
    if raw in (None, ''):
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as ex:
            raise ValueError(
                'container_env must be a JSON object, for example '
                '{"PYTHONPATH": "/opt/app"}'
            ) from ex
    if not isinstance(raw, dict):
        raise TypeError('container_env must be a mapping or JSON object')
    return {str(name): str(value) for name, value in raw.items()}


def _coerce_name_list(raw: Any) -> list[str]:
    """Accept a list, or one colon/comma separated string."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = raw
    else:
        items = str(raw).replace(',', ':').split(':')
    return [str(item).strip() for item in items if str(item).strip()]


def node_image(node: Any = None) -> str:
    """The image this node's command runs in, empty when it runs on the host."""
    return str(getattr(node, 'container_image', None) or '').strip()


def node_mounts(node: Any = None) -> list[str]:
    """Host paths to bind-mount at their own absolute paths."""
    return _coerce_name_list(getattr(node, 'container_mounts', None))


def declared_env(node: Any = None) -> dict[str, str | None]:
    """
    Render-time variables and their values, in a stable order.

    Values are captured here rather than forwarded by name because the
    environment that will run the command is not this one; see the note in
    :func:`container_prefix`.

    A declared name with no value keeps a bare ``-e NAME`` (value None in the
    result): a name that is not set yet can only be a job-time value.
    """
    runtime_env = set(
        _coerce_name_list(getattr(node, 'container_runtime_env', None))
    )
    names = _coerce_name_list(getattr(node, 'container_capture_env', None))
    for name in _coerce_name_list(
        getattr(node, 'container_forward_env', None)
    ):
        if name not in names and name not in runtime_env:
            names.append(name)

    resolved: dict[str, str | None] = {
        name: os.environ.get(name) or None for name in names
    }
    # An explicit mapping on the node wins over the environment.
    for name, value in (getattr(node, 'container_env', None) or {}).items():
        resolved[str(name)] = str(value)
    return resolved


def host_interpreter(command: str) -> str:
    """
    Render a leading bare ``python`` as the interpreter that runs the node.

    ``python`` in a node's executable means "the interpreter that will run
    this node": inside an image that is the image's own, on PATH; on the host
    it is this process's, because a cmd_queue worker does not inherit the
    orchestrator's virtualenv and a bare ``python -m ...`` dies with
    "python: not found" before the node starts. The substitution happens at
    render time, on the node, so a pipeline never has to guess at
    construction which route the run will take. (Three pipelines used to
    guess by calling :func:`containerization_is_enabled` with no node while
    building their DAG; once settings moved onto the nodes that answer was
    always "host", and their containers were handed a path that did not
    exist inside them.)

    Only the first word is touched, and only when it is exactly ``python``.

    Example:
        >>> from magnet.containers import host_interpreter
        >>> import sys, shlex
        >>> host_interpreter('python -m pkg.work --x=1') == (
        ...     shlex.quote(sys.executable) + ' -m pkg.work --x=1')
        True
        >>> host_interpreter('python3 -m pkg.work')
        'python3 -m pkg.work'
        >>> host_interpreter('/usr/bin/python -m pkg.work')
        '/usr/bin/python -m pkg.work'
    """
    if command == 'python' or command.startswith(('python ', 'python\t', 'python\n', 'python \\')):
        return shlex.quote(sys.executable) + command[len('python'):]
    return command


def containerization_is_enabled(node: Any = None) -> bool:
    """
    Whether node commands should be wrapped in ``docker run``.

    Returns:
        bool: true when the node names an image, either its own or the one
            :func:`apply_settings` gave it.
    """
    return bool(node_image(node))


def container_prefix(node: Any = None) -> str:
    """The ``docker run`` invocation node commands are appended to.

    Args:
        node: the node being rendered. Every value comes off it -- its own
            declaration, or what :func:`apply_settings` wrote there.

    Returns:
        str: everything up to and including the image name.
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
    # Job-time (`container_runtime_env`): `infer-stack run` writes these into
    # the wrapped command's environment long after this string is rendered, so
    # they stay a bare `-e NAME`. A captured value would freeze the
    # orchestrator's shell over the endpoint actually leased.
    #
    # Render-time: values exist now and are not recreated later. A tmux worker
    # inherits the tmux server's environment, not the orchestrator's, so a bare
    # `-e NAME` forwards nothing and does it silently -- that cost a full run
    # when OC_BACKEND_FACTORY vanished and every shard routed to a provider it
    # had no key for.
    for name in _coerce_name_list(
        getattr(node, 'container_runtime_env', None)
    ):
        parts += ['-e', name]
    for name, value in declared_env(node).items():
        if value is None:
            parts += ['-e', name]
        else:
            parts += ['-e', shlex.quote(f'{name}={value}')]
    docker_args = getattr(node, 'container_docker_args', None)
    parts += shlex.split(str(docker_args or ''))
    parts.append(image)
    return ' '.join(parts)


class ContainerCapability:
    """
    Container-specific state and configuration, independent of node type.

    This mixin owns no ``command`` property and knows nothing about leasing.
    :class:`magnet.process_node.MagnetProcessNode` is the supported integration
    surface. Tests also compose this mixin with a bare kwdagger node to pin
    the fact that the capability itself remains independently reusable.

    Every value the command needs lives on the node by the time it renders.
    A node that declares its own keeps it; :func:`apply_settings` fills the
    rest in. That is what lets two pipelines in one process differ, and what
    keeps a rendered command explainable from the node alone.
    """

    #: Image for this node's command. Empty => run on the host.
    container_image: str | None = None
    #: Host paths bind-mounted at their own absolute paths.
    container_mounts: str | list[str] | tuple[str, ...] | None = None
    #: Render-time variables, name -> value, captured into the command.
    container_env: Mapping[str, object] | None = None
    #: Additional names whose values are captured at render time.
    container_forward_env: str | list[str] | tuple[str, ...] | None = ()
    #: Extra ``docker run`` arguments, shell-split into the prefix.
    container_docker_args: str | None = None

    #: Variables supplied by the surrounding job at execution time. These are
    #: forwarded by name; capturing them while the DAG is rendered could freeze
    #: a stale endpoint over a later ``infer-stack run`` lease.
    container_runtime_env: tuple[str, ...] = (
        'OPENAI_BASE_URL',
        'OPENAI_API_KEY',
    )

    #: Variables whose values exist when the DAG is rendered and therefore
    #: need to be embedded in the command. This policy is attached to the node
    #: class so a specialized node can override it without editing a module
    #: passlist.
    container_capture_env: tuple[str, ...] = (
        'PYTHONPATH',
        'HF_TOKEN',
        'HF_HOME',
        'TRANSFORMERS_OFFLINE',
        'HF_HUB_OFFLINE',
    )

    def apply_container_settings(self, settings: ContainerSettings) -> None:
        """
        Adopt an invocation's settings for anything this node did not declare.

        A node's own declaration is a property of the step and always wins.
        ``container_env`` merges by name so node values override invocation
        defaults without discarding unrelated defaults. ``forward_env`` also
        accumulates because both sides are adding names. Idempotent, so
        applying twice is the same as applying once.

        Example:
            >>> from magnet.process_node import MagnetProcessNode
            >>> from magnet.containers import ContainerSettings
            >>> node = MagnetProcessNode(name='n', executable='true')
            >>> node.container_image = 'declared:latest'
            >>> settings = ContainerSettings.coerce(image='other:latest')
            >>> node.apply_container_settings(settings)
            >>> node.container_image
            'declared:latest'
        """
        if not self.container_image:
            self.container_image = settings.image
        if not self.container_mounts:
            self.container_mounts = list(settings.mounts)
        # Invocation values are defaults; a node may override individual
        # names without having to restate the rest of the invocation mapping.
        env = dict(settings.env)
        env.update(self.container_env or {})
        self.container_env = env or None
        if not self.container_docker_args:
            self.container_docker_args = settings.docker_args
        names = _coerce_name_list(self.container_forward_env)
        for name in settings.forward_env:
            if name not in names:
                names.append(name)
        self.container_forward_env = tuple(names)


def is_container_capable(node: Any) -> bool:
    """Whether ``node`` carries the container execution capability."""
    return isinstance(node, ContainerCapability)


def render_container_command(node: Any, command: str) -> str:
    """Render the execution substrate: Docker when enabled, host otherwise."""
    if containerization_is_enabled(node):
        return container_prefix(node) + " \\\n    " + command
    return host_interpreter(command)
