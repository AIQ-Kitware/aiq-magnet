"""Container execution support for kwdagger nodes.

``ContainerCapability`` stores container settings and wraps a node command with
``docker run``. ``MagnetProcessNode`` composes it with endpoint leasing. When
both are enabled, the lease wraps the container command.

Container working directories are bind-mounted at the same absolute paths used
on the host so kwdagger artifact paths remain valid inside the container.
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
    'host_interpreter',
]


@dataclasses.dataclass(frozen=True)
class ContainerSettings:
    """Invocation-level defaults for container-capable nodes.

    :meth:`apply` fills values that a node did not declare itself. An empty
    image leaves the node on the host.
    """

    #: Image to run node commands in. Empty => run on the host.
    image: str = ''

    #: Host paths to bind-mount at their own absolute paths.
    mounts: tuple[str, ...] = ()

    #: Fixed environment values. Node-specific values override these defaults.
    env: dict[str, str] = dataclasses.field(default_factory=dict)

    #: Extra ``docker run`` arguments.
    docker_args: str = ''

    #: Extra environment variable names to capture at render time.
    forward_env: tuple[str, ...] = ()

    def apply(self, pipeline: Any) -> None:
        """Apply these invocation settings to every container-capable node."""
        for node in (getattr(pipeline, 'node_dict', None) or {}).values():
            if isinstance(node, ContainerCapability):
                node.apply_container_settings(self)

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



def host_interpreter(command: str) -> str:
    """Replace a leading bare ``python`` with :data:`sys.executable`.

    Host workers may not inherit the orchestrator's virtual environment. Only
    an exact leading ``python`` token is replaced; ``python3`` and explicit
    interpreter paths are unchanged.

    Example:
        >>> from magnet.containers import host_interpreter
        >>> import sys, shlex
        >>> host_interpreter('python -m pkg.work --x=1') == (
        ...     shlex.quote(sys.executable) + ' -m pkg.work --x=1')
        True
        >>> host_interpreter('python3 -m pkg.work')
        'python3 -m pkg.work'
    """
    if command == 'python' or command.startswith(('python ', 'python\t', 'python\n', 'python \\')):
        return shlex.quote(sys.executable) + command[len('python'):]
    return command



class ContainerCapability:
    """Mixin providing container state and command wrapping.

    The mixin does not define ``command``. ``MagnetProcessNode`` controls how
    containerization composes with other execution capabilities.
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

    #: Variables captured when the DAG command is rendered.
    container_capture_env: tuple[str, ...] = (
        'PYTHONPATH',
        'HF_TOKEN',
        'HF_HOME',
        'TRANSFORMERS_OFFLINE',
        'HF_HUB_OFFLINE',
    )

    def containerization_is_enabled(self) -> bool:
        """Whether this node should run inside its configured container."""
        return bool(str(self.container_image or '').strip())

    def _container_mount_paths(self) -> list[str]:
        """Host paths this node bind-mounts at their absolute paths."""
        return _coerce_name_list(self.container_mounts)

    def _container_resolved_env(self) -> dict[str, str | None]:
        """Render-time environment values to embed in ``docker run``."""
        runtime_env = set(_coerce_name_list(self.container_runtime_env))
        names = _coerce_name_list(self.container_capture_env)
        for name in _coerce_name_list(self.container_forward_env):
            if name not in names and name not in runtime_env:
                names.append(name)

        resolved: dict[str, str | None] = {
            name: os.environ.get(name) or None for name in names
        }
        for name, value in (self.container_env or {}).items():
            resolved[str(name)] = str(value)
        return resolved

    def _container_command_prefix(self) -> str:
        """Build the ``docker run`` prefix for this node."""
        image = str(self.container_image or '').strip()
        parts = [
            'docker', 'run', '--rm',
            # infer-stack exports a loopback endpoint; host networking keeps
            # that endpoint valid inside the container.
            '--network', 'host',
            # Keep artifacts writable by subsequent host-side jobs.
            '--user', f'{os.getuid()}:{os.getgid()}',
        ]
        for mount in self._container_mount_paths():
            parts += ['-v', f'{mount}:{mount}']
        parts += [
            '-w', '"$PWD"',
            '-e', 'HOME=/tmp',
        ]
        # Lease-provided variables only exist when the job executes, so they
        # must be forwarded by name rather than captured while rendering.
        for name in _coerce_name_list(self.container_runtime_env):
            parts += ['-e', name]
        for name, value in self._container_resolved_env().items():
            if value is None:
                parts += ['-e', name]
            else:
                parts += ['-e', shlex.quote(f'{name}={value}')]
        parts += shlex.split(str(self.container_docker_args or ''))
        parts.append(image)
        return ' '.join(parts)

    def wrap_with_container(self, command: str) -> str:
        """Wrap ``command`` in Docker when containerization is enabled."""
        if not self.containerization_is_enabled():
            return command
        return self._container_command_prefix() + " \\\n    " + command

    def apply_container_settings(self, settings: ContainerSettings) -> None:
        """Apply invocation defaults without overriding node declarations.

        Environment mappings merge by name, with node values taking precedence.
        Forwarded environment names accumulate.

        Example:
            >>> from magnet.process_node import MagnetProcessNode
            >>> from magnet.containers import ContainerSettings
            >>> node = MagnetProcessNode(name='n', executable='true')
            >>> node.container_image = 'declared:latest'
            >>> node.apply_container_settings(
            ...     ContainerSettings.coerce(image='other:latest'))
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
