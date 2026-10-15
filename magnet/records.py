"""Invocation identity and execution provenance, separate from scientific verdicts."""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

from magnet.results import file_digest


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as file:
            json.dump(value, file, indent=2, ensure_ascii=False)
            file.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _safe(value):
    """Do not persist credentials from execution options or container settings."""
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'env' and isinstance(item, dict):
                result['env_names'] = sorted(item)
            elif re.search(r'(?i)(api.?key|password|secret|authorization|credential|access.token)', str(key)):
                result[key] = '<redacted>'
            else:
                result[key] = _safe(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, str):
        value = re.sub(r'(?i)((?:api[_-]?key|password|secret|authorization|access[_-]?token)\s*[=:]\s*)[^\s,;]+',
                       r'\1<redacted>', value)
        return re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@', r'\1<redacted>@', value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return repr(value)


def source_snapshot(file: str | Path) -> dict:
    path = Path(file).resolve()
    result = {'file': str(path)}
    try:
        result['sha256'] = file_digest(path)
    except OSError as ex:
        result.update(sha256=None, source_error=type(ex).__name__)
    try:
        def git(*args):
            return subprocess.run(['git', '-C', str(path.parent), *args],
                                  check=True, capture_output=True, text=True, timeout=5).stdout.strip()
        result['revision'] = git('rev-parse', 'HEAD')
        result['dirty'] = bool(git('status', '--porcelain', '--untracked-files=normal'))
        # Hash the patch without copying arbitrary repository content into metadata.
        result['tracked_patch_sha256'] = __import__('hashlib').sha256(
            git('diff', 'HEAD', '--no-ext-diff').encode()).hexdigest()
    except (OSError, subprocess.SubprocessError):
        result['revision'] = None
    return result


def execution_snapshot(recipe, options: dict) -> dict:
    versions = {}
    sources = {}
    for package, module in [('aiq-magnet', 'magnet.evaluation_new'),
                            ('kwdagger', 'kwdagger'), ('cmd_queue', 'cmd_queue')]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
        imported = sys.modules.get(module)
        if imported is not None and getattr(imported, '__file__', None):
            sources[package] = source_snapshot(imported.__file__)
    return {'schema_version': 1, 'record_id': recipe._run_hash,
            'record_directory': str((recipe.output_path / recipe._run_hash).absolute()),
            'started': timestamp(), 'state': 'scheduling',
            'python': sys.executable, 'python_version': platform.python_version(),
            'host': platform.node(), 'cwd': str(Path.cwd()),
            'options': _safe(options), 'software': versions, 'sources': sources,
            'producer_history': 'Per-output origin is not inferred from this readout environment.'}
