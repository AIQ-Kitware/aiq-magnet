"""Read, present and export persisted evaluation results without executing cards.

The existing ``card.yaml``, ``verdict.json`` and ``results/*/verdict.json``
files remain authoritative. The optional export manifest adds integrity and
portable artifact locations; it is not a second scientific result format.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import safer
import ubelt as ub
import yaml

from magnet.records import RecordFiles, atomic_json_write


def _mapping(
    path: Path, *, yaml_file: bool = False, optional: bool = False
) -> dict:
    if optional and not path.is_file():
        return {}
    text = path.read_text()
    if yaml_file:
        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f'expected an object in {path}')
    return value


def _relative(root: Path, name: str) -> Path:
    """Resolve a portable locator, rejecting escape from the bundle."""
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError(f'invalid bundle-relative path: {name!r}')
    path = root / rel
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'bundle path escapes through a symlink: {name!r}')
    return path


@dataclass
class ResultBundle:
    root: Path
    card: dict
    verdict: dict
    cells: dict[str, dict]
    missing_claims: list[str] = field(default_factory=list)
    execution: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    def core_files(self):
        """Scientific files referenced by this record, relative to its root."""
        yield RecordFiles.CARD
        if (self.root / RecordFiles.VERDICT).is_file():
            yield RecordFiles.VERDICT
        for key in self.cells:
            yield RecordFiles.cell_verdict(key).as_posix()

    def artifact_path(self, original: str) -> Path:
        """Prefer exported locations over original-machine provenance paths."""
        for item in self.manifest.get('artifacts', []):
            if item['source'] == original:
                return _relative(self.root, item['path'])
        path = Path(original)
        return path if path.is_absolute() else _relative(self.root, original)

    def measurements(self, cell: dict) -> dict:
        """Prefer terminal-node evidence; ancestors may have gathered instances."""
        evidence = cell.get('evidence') or {}
        node = (self.card.get('kwdagger') or {}).get('result_node')
        prefix = f'metrics.{node}.' if node else None
        selected = {k[len(prefix):]: v for k, v in evidence.items()
                    if prefix and k.startswith(prefix)}
        if selected:
            return selected
        # Older bundles preserve concrete claim inputs as dashboard symbols.
        return {k: v for k, v in (cell.get('symbols') or {}).items()
                if k.startswith('metrics.') or
                ((self.card.get('symbols') or {}).get(k, {}).get('metadata') or {}).get('kind') == 'metric'}


def load_result_bundle(path: str | os.PathLike) -> ResultBundle:
    root = Path(path).absolute()
    if root.is_file():
        root = root.parent
    card = _mapping(root / RecordFiles.CARD, yaml_file=True)
    verdict = _mapping(root / RecordFiles.VERDICT, optional=True)
    claims = verdict.get('claims', [])
    if not isinstance(claims, list) or not all(isinstance(k, str) for k in claims):
        raise ValueError('verdict claims must be a list of result IDs')
    cells = {}
    missing = []
    for key in claims:
        file = _relative(root, RecordFiles.cell_verdict(key).as_posix())
        if file.is_file():
            cells[key] = _mapping(file)
        else:
            missing.append(key)
    return ResultBundle(
        root,
        card,
        verdict,
        cells,
        missing,
        _mapping(root / RecordFiles.EXECUTION, optional=True),
        _mapping(root / RecordFiles.MANIFEST, optional=True),
    )


def discover_result_bundles(path: str | os.PathLike) -> list[dict]:
    """List saved evaluations, pruning caches rather than interpreting them."""
    records = []
    for folder, dirs, files in os.walk(
        Path(path).absolute(), followlinks=False
    ):
        dirs[:] = sorted(d for d in dirs if not d.startswith(('_', '.')))
        if RecordFiles.CARD not in files:
            continue
        dirs[:] = []
        bundle = load_result_bundle(folder)
        records.append({'path': str(bundle.root),
                        'result': bundle.verdict.get('result', 'NOT_RECORDED'),
                        'claims': len(bundle.verdict.get('claims', [])),
                        'available_claims': len(bundle.cells),
                        'missing_claims': bundle.missing_claims,
                        'execution_state': bundle.execution.get('state', 'unrecorded')})
    return sorted(records, key=lambda item: item['path'])


def _scalar(value: Any) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, '.8g')
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _cell(value: Any) -> str:
    return _scalar(value).replace('|', '\\|').replace('\n', '<br>')


def _present_value(name: str, value: Any, level: int = 4) -> list[str]:
    """Generic rendering for scalar, mapping and record-array measurements."""
    heading = '#' * min(level, 6)
    if isinstance(value, dict):
        lines = [f'{heading} {name}', '']
        for key, item in value.items():
            lines.extend(_present_value(str(key), item, level + 1))
        return lines
    if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
        columns = list(dict.fromkeys(k for row in value for k, v in row.items()
                                    if not isinstance(v, (list, dict))))
        lines = [f'{heading} {name}', '']
        if columns:
            lines += ['| ' + ' | '.join(_cell(c) for c in columns) + ' |',
                      '| ' + ' | '.join('---' for _ in columns) + ' |']
            lines += ['| ' + ' | '.join(_cell(row.get(c)) for c in columns) + ' |' for row in value]
            lines.append('')
        for i, row in enumerate(value, 1):
            for key, item in row.items():
                if isinstance(item, (list, dict)):
                    lines.extend(_present_value(f'{name}[{i}].{key}', item, level + 1))
        return lines
    return [f'- **{name}:** {_cell(value)}', '']


def render_result(bundle: ResultBundle, *, format: str = 'markdown') -> str:
    """Present stored values. Never evaluate Python claims or load pipelines."""
    if format == 'json':
        return json.dumps({'card': bundle.card, 'verdict': bundle.verdict,
                           'cells': bundle.cells, 'missing_claims': bundle.missing_claims,
                           'execution': bundle.execution}, indent=2, ensure_ascii=False) + '\n'
    if format not in {'markdown', 'text'}:
        raise ValueError(f'unknown report format: {format}')
    lines = [f"# {bundle.card.get('title', bundle.card.get('name', 'Evaluation'))}", '',
             f"Result: **{bundle.verdict.get('result', 'NOT_RECORDED')}**", '',
             str(bundle.card.get('description') or ''), '']
    if not bundle.verdict:
        lines += ['No aggregate verdict was recorded. Available execution metadata follows; '
                  'this is not a claim verdict.', '']
    if bundle.missing_claims:
        lines += ['Missing referenced claim records: ' + ', '.join(bundle.missing_claims), '']
    verdict_sections = ['evidence', 'requested_work', 'metrics', 'provenance']
    for key in verdict_sections:
        if key in bundle.verdict:
            lines.extend(_present_value(key, bundle.verdict[key], 2))
    claim = bundle.card.get('claim') or {}
    python = claim.get('python', '') if isinstance(claim, dict) else str(claim)
    fence = '```'
    while fence in python:
        fence += '`'
    lines += ['## Claim', '', fence + 'python', python, fence, '', '## Claim results', '']
    for key, cell in bundle.cells.items():
        lines += [f'### {key}', '', f"Status: **{cell.get('status', 'UNKNOWN')}**", '',
                  str(cell.get('output') or ''), '']
        measurements = bundle.measurements(cell)
        if measurements:
            lines.extend(_present_value('Measurements', measurements, 4))
        else:
            lines += ['No separately identified measurements; resolved claim inputs follow.', '']
        inputs = cell.get('symbols') or {}
        if measurements:
            inputs = {k: v for k, v in inputs.items()
                      if not k.startswith('metrics.') and k not in measurements}
        lines.extend(_present_value('Resolved claim inputs', inputs, 4))
        if cell.get('consumed'):
            lines.extend(_present_value('Consumed evidence fields', cell['consumed'], 4))
    if bundle.execution:
        lines.extend(_present_value('Execution snapshot', bundle.execution, 2))
    lines += ['Presentation generated from serialized results. Logs and original artifact '
              'paths are not required to read this report.', '']
    text = '\n'.join(line.rstrip() for line in lines)
    if format == 'text':
        import re
        text = re.sub(r'^#{1,6} ', '', text, flags=re.MULTILINE).replace('**', '')
    return text


def write_result_reports(
    bundle: ResultBundle, directory: Path | None = None
) -> list[Path]:
    """Generate the standard presentations for a record or export."""
    directory = bundle.root if directory is None else directory
    paths = []
    for name, fmt in RecordFiles.REPORTS.items():
        path = directory / name
        with safer.open(
            path, 'w', encoding='utf-8', temp_file=os.name != 'nt'
        ) as file:
            file.write(render_result(bundle, format=fmt))
        paths.append(path)
    return paths


def export_result_bundle(
    source: str | os.PathLike,
    destination: str | os.PathLike,
    *,
    supporting: str = 'none',
    nodes: tuple[str, ...] = (),
    include_logs: bool = False,
    attachments: dict[str, str | os.PathLike] | None = None,
) -> Path:
    """Copy a selected native result and optional declared output artifacts.

    Native files remain byte-identical. An index maps original artifact paths
    to copied locations. Missing optional artifacts are listed, not inferred
    to be scientific failures. Unrelated shared-cache files are never scanned.
    """
    if supporting not in {'none', 'terminal', 'requested'}:
        raise ValueError('supporting must be none, terminal or requested')
    bundle = load_result_bundle(source)
    if bundle.manifest:
        checked = verify_result_bundle(bundle.root)
        if not checked['valid']:
            raise ValueError(f'source bundle failed integrity checks: {checked["errors"]}')
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(bundle.root):
        raise ValueError('export destination must be outside the source record')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix='.magnet-export-', dir=destination.parent
    ) as folder:
        staging = Path(folder) / 'bundle'
        staging.mkdir()
        manifest: dict[str, Any] = {'schema_version': 1, 'format': 'magnet.result-bundle',
                    'source_record': str(bundle.root), 'files': {}, 'artifacts': [],
                    'missing_supporting': [], 'supporting': supporting}

        def copy(source: Path, rel: str) -> None:
            target = _relative(staging, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest['files'][rel] = {
                'sha256': ub.hash_file(target, hasher='sha256', base='hex'),
                'size': target.stat().st_size,
            }

        for name in bundle.core_files():
            copy(_relative(bundle.root, name), name)
        for name in RecordFiles.METADATA:
            if (bundle.root / name).is_file():
                copy(_relative(bundle.root, name), name)
        # A launcher may keep this explicit diagnostic outside the record and
        # link it in. Only --include_logs follows this known console locator.
        if include_logs:
            log_names = [RecordFiles.LOG, RecordFiles.CONSOLE]
            for name in log_names:
                if (bundle.root / name).is_file():
                    if name == RecordFiles.CONSOLE:
                        log_path = (bundle.root / name).resolve()
                    else:
                        log_path = _relative(bundle.root, name)
                    copy(log_path, name)

        wanted: dict[str, tuple[str, str]] = {}
        requests = []
        requests_path = bundle.root / RecordFiles.REQUESTED_RUNS
        if requests_path.is_file():
            requests = json.loads(requests_path.read_text())
        observed = {item['expected_output']: item for item in requests if item.get('expected_output')}
        if supporting in {'terminal', 'requested'}:
            for cell in bundle.cells.values():
                original = cell.get('artifact')
                if original:
                    wanted[original] = ('terminal', str(cell.get('cell') or Path(original).parent.name))
        if supporting == 'requested':
            for item in requests:
                if nodes and item['node'] not in nodes:
                    continue
                original = item.get('expected_output')
                if original:
                    wanted[original] = (item['node'], item['process_id'])
        for i, (original, (node, process)) in enumerate(sorted(wanted.items())):
            path = bundle.artifact_path(original)
            if not path.is_file():
                manifest['missing_supporting'].append({'source': original, 'process_id': process})
                continue
            # Numbered directories avoid collisions without trusting node names as paths.
            rel = f'artifacts/{i:05d}/{path.name}'
            copy(path, rel)
            prior = observed.get(original, {})
            expected = prior.get('output_sha256_at_readout')
            if expected and manifest['files'][rel]['sha256'] != expected:
                raise ValueError(f'supporting artifact changed since readout: {original}')
            manifest['artifacts'].append({'source': original, 'path': rel,
                                          'node': node, 'process_id': process})
            config = path.parent / RecordFiles.JOB_CONFIG
            if config.is_file():
                config_rel = f'artifacts/{i:05d}/{RecordFiles.JOB_CONFIG}'
                copy(config, config_rel)
                expected = prior.get('config_sha256_at_readout')
                if (
                    expected
                    and manifest['files'][config_rel]['sha256'] != expected
                ):
                    raise ValueError(f'supporting configuration changed since readout: {original}')
        for name, path in (attachments or {}).items():
            # Attachments cannot shadow the scientific record or its manifest.
            _relative(staging, name)
            copy(Path(path), f'attachments/{name}')
        for path in write_result_reports(bundle, staging):
            manifest['files'][path.name] = {
                'sha256': ub.hash_file(path, hasher='sha256', base='hex'),
                'size': path.stat().st_size,
            }
        atomic_json_write(staging / RecordFiles.MANIFEST, manifest)
        staging.rename(destination)
    return destination


def verify_result_bundle(path: str | os.PathLike) -> dict:
    """Check included file bytes and core references, without original paths."""
    bundle = load_result_bundle(path)
    errors = [f'missing claim record: {key}' for key in bundle.missing_claims]
    files = bundle.manifest.get('files')
    if files is None:
        return {'valid': not errors, 'integrity_manifest': False, 'checked_files': 0,
                'errors': errors, 'missing_supporting': []}
    if bundle.manifest.get('format') != 'magnet.result-bundle' or bundle.manifest.get('schema_version') != 1:
        errors.append('unsupported export manifest')
    for rel, expected in files.items():
        file = _relative(bundle.root, rel)
        if not file.is_file():
            errors.append(f'missing exported file: {rel}')
        elif (
            file.stat().st_size != expected['size']
            or ub.hash_file(file, hasher='sha256', base='hex') != expected['sha256']
        ):
            errors.append(f'content changed: {rel}')
    for item in bundle.manifest.get('artifacts', []):
        if item['path'] not in files:
            errors.append(f'artifact not covered by integrity manifest: {item["path"]}')
    for key in bundle.core_files():
        if key not in files:
            errors.append(f'core file not covered by integrity manifest: {key}')
    return {'valid': not errors, 'integrity_manifest': True, 'checked_files': len(files),
            'errors': errors, 'missing_supporting': bundle.manifest.get('missing_supporting', [])}
