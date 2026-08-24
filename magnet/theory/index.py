"""Theory objects and named premises that empirical annotations may reference."""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from magnet.theory.links import split_ref

__all__ = [
    'INDEX_SCHEMA_VERSION',
    'KINDS',
    'Formalization',
    'Premise',
    'Entry',
    'TheoryIndex',
    'load_index',
    'load_indexes',
    'parse_entries',
]

INDEX_SCHEMA_VERSION = 1
KINDS = ('theorem', 'conjecture', 'question', 'definition')


@dataclass(frozen=True)
class Formalization:
    """Structured provenance for a formal theory source."""

    system: str
    repository: str = ''
    revision: str = ''

    def to_dict(self) -> dict:
        data = {'system': self.system}
        if self.repository:
            data['repository'] = self.repository
        if self.revision:
            data['revision'] = self.revision
        return data


@dataclass(frozen=True)
class Premise:
    """One named premise, normally a proposition-valued formal binder."""

    id: str
    type: str = ''
    statement: str = ''

    def to_dict(self) -> dict:
        data = {'id': self.id}
        if self.type:
            data['type'] = self.type
        if self.statement:
            data['statement'] = self.statement
        return data


@dataclass(frozen=True)
class Entry:
    """One theoretical object, optionally tied to a formal declaration."""

    id: str
    kind: str
    statement: str = ''
    declaration: str = ''
    formalization: Formalization | None = None
    source_path: str = ''
    premises: tuple[Premise, ...] = ()

    def premise(self, premise_id: str) -> Premise:
        for premise in self.premises:
            if premise.id == premise_id:
                return premise
        raise KeyError(f'{self.id}::{premise_id}')

    def to_dict(self) -> dict:
        data = {'id': self.id, 'kind': self.kind}
        if self.statement:
            data['statement'] = self.statement
        if self.declaration:
            data['declaration'] = self.declaration
        if self.formalization is not None:
            data['formalization'] = self.formalization.to_dict()
        if self.source_path:
            data['source_path'] = self.source_path
        if self.premises:
            data['premises'] = [premise.to_dict() for premise in self.premises]
        return data


class TheoryIndex:
    """Entries loaded from inline card data and versioned theory index files."""

    def __init__(self, entries: Sequence[Entry] = ()) -> None:
        self._entries: dict[str, Entry] = {}
        for entry in entries:
            if entry.id in self._entries:
                raise ValueError(f'duplicate theory entry id: {entry.id!r}')
            self._entries[entry.id] = entry

    def __contains__(self, ref: str) -> bool:
        try:
            self.resolve(ref)
        except KeyError:
            return False
        return True

    def __getitem__(self, ref: str) -> Entry:
        entry_id, premise_id = split_ref(ref)
        if premise_id is not None:
            raise KeyError(
                f'{ref!r} names a premise; index entries are addressed by EntryId'
            )
        return self._entries[entry_id]

    def __iter__(self):
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def resolve(self, ref: str) -> Entry | Premise:
        entry_id, premise_id = split_ref(ref)
        entry = self._entries[entry_id]
        if premise_id is None:
            return entry
        return entry.premise(premise_id)

    def unresolved(self, refs: Sequence[str]) -> list[str]:
        missing = []
        for ref in dict.fromkeys(refs):
            try:
                self.resolve(ref)
            except (KeyError, ValueError):
                missing.append(ref)
        return sorted(missing)

    def to_list(self) -> list[dict]:
        return [entry.to_dict() for entry in self._entries.values()]


def _parse_formalization(raw, where: str) -> Formalization | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f'{where}: formalization must be a mapping')
    allowed = {'system', 'repository', 'revision'}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f'{where}: unknown formalization fields: {sorted(extra)}')
    system = raw.get('system')
    if not isinstance(system, str) or not system:
        raise ValueError(f'{where}: formalization.system is required')
    repository = raw.get('repository') or ''
    revision = raw.get('revision') or ''
    if not isinstance(repository, str) or not isinstance(revision, str):
        raise ValueError(f'{where}: formalization repository/revision must be strings')
    if repository and not revision:
        raise ValueError(f'{where}: formalization.revision is required with repository')
    return Formalization(system=system, repository=repository, revision=revision)


def _parse_premises(raw_premises, where: str) -> tuple[Premise, ...]:
    premises = []
    seen = set()
    for index, raw in enumerate(raw_premises or []):
        item_where = f'{where}[{index}]'
        if not isinstance(raw, dict):
            raise ValueError(f'{item_where} must be a mapping')
        allowed = {'id', 'type', 'statement'}
        extra = set(raw) - allowed
        if extra:
            raise ValueError(f'{item_where} has unknown fields: {sorted(extra)}')
        premise_id = raw.get('id')
        if not isinstance(premise_id, str) or not premise_id:
            raise ValueError(f'{item_where} has no id')
        if '::' in premise_id:
            raise ValueError(f'{item_where} id must be a binder name, not a full reference')
        if premise_id in seen:
            raise ValueError(f'{where}: duplicate premise id {premise_id!r}')
        seen.add(premise_id)
        premise_type = raw.get('type') or ''
        statement = raw.get('statement') or ''
        if not isinstance(premise_type, str) or not isinstance(statement, str):
            raise ValueError(f'{item_where} type/statement must be strings')
        premises.append(
            Premise(
                id=premise_id,
                type=premise_type.strip(),
                statement=statement.strip(),
            )
        )
    return tuple(premises)


def parse_entries(
    raw_entries,
    where: str = 'entries',
    *,
    default_formalization: Formalization | None = None,
) -> list[Entry]:
    """Validate entry mappings shared by inline cards and index files."""
    entries = []
    for index, raw in enumerate(raw_entries or []):
        item_where = f'{where}[{index}]'
        if not isinstance(raw, dict):
            raise ValueError(f'{item_where} must be a mapping')
        allowed = {
            'id',
            'kind',
            'statement',
            'declaration',
            'formalization',
            'source_path',
            'premises',
        }
        extra = set(raw) - allowed
        if extra:
            raise ValueError(f'{item_where} has unknown fields: {sorted(extra)}')

        ref = raw.get('id')
        if not isinstance(ref, str) or not ref:
            raise ValueError(f'{item_where}: an entry has no id')
        if '::' in ref:
            raise ValueError(f'{item_where}: entry ids may not contain ::')

        kind = raw.get('kind')
        if kind not in KINDS:
            raise ValueError(
                f'{item_where}: entry {ref!r} has kind {kind!r}; '
                f'known kinds are {list(KINDS)}'
            )

        statement = raw.get('statement') or ''
        declaration = raw.get('declaration') or ''
        source_path = raw.get('source_path') or ''
        if not all(isinstance(value, str) for value in (statement, declaration, source_path)):
            raise ValueError(
                f'{item_where}: statement, declaration, and source_path must be strings'
            )
        if not statement and not declaration:
            raise ValueError(
                f'{item_where}: entry {ref!r} needs a statement or declaration'
            )

        if 'formalization' in raw:
            formalization = _parse_formalization(
                raw.get('formalization'), f'{item_where}.formalization'
            )
        else:
            formalization = default_formalization

        premises = _parse_premises(raw.get('premises'), f'{item_where}.premises')
        entries.append(
            Entry(
                id=ref,
                kind=kind,
                statement=statement.strip(),
                declaration=declaration,
                formalization=formalization,
                source_path=source_path,
                premises=premises,
            )
        )
    return entries


def load_index(fpath) -> TheoryIndex:
    """Read one versioned YAML theory index."""
    path = Path(fpath)
    data = yaml.safe_load(path.read_text()) or {}
    schema_version = data.get('schema_version')
    if schema_version != INDEX_SCHEMA_VERSION:
        raise ValueError(
            f'{path}: expected schema_version {INDEX_SCHEMA_VERSION}, got {schema_version!r}'
        )
    allowed = {'schema_version', 'formalization', 'entries'}
    extra = set(data) - allowed
    if extra:
        raise ValueError(f'{path}: unknown top-level fields: {sorted(extra)}')
    formalization = _parse_formalization(data.get('formalization'), str(path))
    entries = parse_entries(
        data.get('entries'), str(path), default_formalization=formalization
    )
    return TheoryIndex(entries)


def load_indexes(fpaths: Sequence[str]) -> TheoryIndex:
    """Read several theory index files and reject duplicate entry IDs."""
    entries: list[Entry] = []
    for fpath in fpaths:
        entries.extend(load_index(fpath))
    return TheoryIndex(entries)
