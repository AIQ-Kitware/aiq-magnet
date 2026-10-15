"""Invocation identity and execution provenance, separate from scientific verdicts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import safer


class RecordFiles:
    """Names in the native record layout, shared by writers and readers."""

    CARD = 'card.yaml'
    VERDICT = 'verdict.json'
    CELLS = 'results'
    EXECUTION = 'execution.json'
    REQUESTED_RUNS = 'requested_runs.json'
    SYMBOL_METADATA = 'symbol_metadata.json'
    THEORY = 'theory.json'
    PROVENANCE = 'provenance.json'
    INVOCATION = 'invocation.json'
    MANIFEST = 'bundle_manifest.json'
    LOG = 'log'
    CONSOLE = 'console.log'
    JOB_CONFIG = 'job_config.json'
    METADATA = (
        REQUESTED_RUNS,
        SYMBOL_METADATA,
        THEORY,
        PROVENANCE,
        EXECUTION,
        INVOCATION,
    )
    REPORTS = {'report.md': 'markdown', 'report.txt': 'text'}

    @classmethod
    def cell_verdict(cls, key: str) -> Path:
        if Path(key).name != key or key in {'.', '..', ''}:
            raise ValueError(f'invalid claim ID: {key!r}')
        return Path(cls.CELLS) / key / cls.VERDICT


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict | list) -> None:
    """Write JSON using the same safer policy as the existing evaluators."""
    with safer.open(
        path,
        'w',
        encoding='utf-8',
        make_parents=True,
        temp_file=os.name != 'nt',
    ) as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write('\n')


def file_digest(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def source_snapshot(file: str | Path) -> dict:
    path = Path(file).resolve()
    result = {'file': str(path)}
    try:
        result['sha256'] = file_digest(path)
    except OSError as ex:
        result.update(sha256=None, source_error=type(ex).__name__)
    try:

        def git(*args):
            return subprocess.run(
                ['git', '-C', str(path.parent), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()

        result['revision'] = git('rev-parse', 'HEAD')
        result['dirty'] = bool(
            git('status', '--porcelain', '--untracked-files=normal')
        )
        # Hash the patch without copying arbitrary repository content into metadata.
        result['tracked_patch_sha256'] = hashlib.sha256(
            git('diff', 'HEAD', '--no-ext-diff').encode()
        ).hexdigest()
    except (OSError, subprocess.SubprocessError):
        result['revision'] = None
    return result


def execution_snapshot(
    recipe,
    *,
    backend: str | None = None,
    dry_run: bool = False,
    readout_only: bool = False,
) -> dict:
    """Record execution facts; arbitrary runtime settings are not captured."""
    versions = {}
    sources = {}
    for package, module in [
        ('aiq-magnet', 'magnet.evaluation_new'),
        ('kwdagger', 'kwdagger'),
        ('cmd_queue', 'cmd_queue'),
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
        imported = sys.modules.get(module)
        if imported is not None and getattr(imported, '__file__', None):
            sources[package] = source_snapshot(imported.__file__)
    return {
        'schema_version': 1,
        'record_id': recipe._run_hash,
        'record_directory': str(
            (recipe.output_path / recipe._run_hash).absolute()
        ),
        'started': timestamp(),
        'state': 'scheduling',
        'python': sys.executable,
        'python_version': platform.python_version(),
        'host': platform.node(),
        'cwd': str(Path.cwd()),
        'options': {
            'backend': backend,
            'dry_run': dry_run,
            'readout_only': readout_only,
        },
        'software': versions,
        'sources': sources,
        'producer_history': 'Per-output origin is not inferred from this readout environment.',
    }
