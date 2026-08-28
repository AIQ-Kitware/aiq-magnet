"""Resolve a card's static theory annotations and build ``theory.json``."""
import json
from dataclasses import dataclass, field
from pathlib import Path

from magnet.theory.annotations import STATEMENT_RELATIONS
from magnet.theory.index import TheoryIndex, load_indexes, parse_entries
from magnet.theory.links import Link, split_ref
from magnet.theory.static import extract

__all__ = ['REPORT_SCHEMA_VERSION', 'TheoryReport', 'report_from_card']

REPORT_SCHEMA_VERSION = 1


def _links_from_card(raw_links) -> list[Link]:
    """Parse whole-statement links declared by the evaluation card."""
    links = []
    allowed = {'relation', 'ref', 'note'}
    for index, raw in enumerate(raw_links or []):
        where = f'theory.links[{index}]'
        if not isinstance(raw, dict):
            raise ValueError(f'{where} must be a mapping')
        extra = set(raw) - allowed
        if extra:
            raise ValueError(f'{where} has unknown fields: {sorted(extra)}')
        relation = raw.get('relation')
        if relation not in STATEMENT_RELATIONS:
            raise ValueError(
                f'{where} has relation {relation!r}; '
                f'card links use {list(STATEMENT_RELATIONS)}'
            )
        ref = raw.get('ref')
        if not isinstance(ref, str) or not ref:
            raise ValueError(f'{where} has no ref')
        _, premise_id = split_ref(ref)
        if premise_id is not None:
            raise ValueError(
                f'{where} targets a premise; premise relations belong in empirical source'
            )
        note = raw.get('note') or ''
        if not isinstance(note, str):
            raise ValueError(f'{where}.note must be a string')
        links.append(Link(relation=relation, ref=ref, note=note.strip()))
    return links


@dataclass
class TheoryReport:
    """Static statement links, premise links, and computed premise coverage."""

    statement_links: list[Link] = field(default_factory=list)
    premise_links: list[Link] = field(default_factory=list)
    index: TheoryIndex = field(default_factory=TheoryIndex)

    @property
    def unresolved(self) -> list[str]:
        refs = [link.ref for link in self.statement_links + self.premise_links]
        return self.index.unresolved(refs)

    def _coverage_for(self, entry_id: str) -> dict | None:
        entry = self.index[entry_id]
        if not entry.premises:
            return None

        links_by_premise: dict[str, list[Link]] = {
            premise.id: [] for premise in entry.premises
        }
        for link in self.premise_links:
            parent_id, premise_id = split_ref(link.ref)
            if parent_id == entry_id and premise_id in links_by_premise:
                links_by_premise[premise_id].append(link)

        premises = []
        unaccounted = []
        for premise in entry.premises:
            links = links_by_premise[premise.id]
            item = premise.to_dict()
            item['accounted'] = bool(links)
            if links:
                item['links'] = [link.to_dict() for link in links]
            else:
                unaccounted.append(premise.id)
            premises.append(item)

        accounted_count = len(entry.premises) - len(unaccounted)
        return {
            'ref': entry_id,
            'premise_count': len(entry.premises),
            'accounted_count': accounted_count,
            'complete': not unaccounted,
            'unaccounted': unaccounted,
            'premises': premises,
        }

    def to_dict(self) -> dict:
        """Build the portable, versioned ``theory.json`` payload."""
        entry_ids = []
        for link in self.statement_links + self.premise_links:
            entry_id = link.entry_id
            if entry_id not in entry_ids:
                entry_ids.append(entry_id)

        all_statement_entry_ids = []
        for link in self.statement_links:
            if link.entry_id not in all_statement_entry_ids:
                all_statement_entry_ids.append(link.entry_id)

        # Premise applicability is relevant when practice claims to test or
        # approximate a statement. A motivating observation does not claim the
        # statement applies, so it does not create a premise-coverage obligation.
        coverage_entry_ids = []
        for link in self.statement_links:
            if link.relation not in {'tests', 'approximates'}:
                continue
            if link.entry_id not in coverage_entry_ids:
                coverage_entry_ids.append(link.entry_id)

        coverage = []
        for entry_id in coverage_entry_ids:
            item = self._coverage_for(entry_id)
            if item is not None:
                coverage.append(item)

        attached = set(all_statement_entry_ids)
        unattached = [
            link.to_dict()
            for link in self.premise_links
            if link.entry_id not in attached
        ]

        data = {
            'schema_version': REPORT_SCHEMA_VERSION,
            'statement_links': [link.to_dict() for link in self.statement_links],
            'premise_links': [link.to_dict() for link in self.premise_links],
            'entries': [self.index[entry_id].to_dict() for entry_id in entry_ids],
            'premise_coverage': coverage,
            'unattached_premise_links': unattached,
        }
        if self.unresolved:
            data['unresolved'] = self.unresolved
        return data

    def write(self, fpath) -> None:
        path = Path(fpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + '\n')


def report_from_card(card: dict, root) -> TheoryReport | None:
    """Resolve a card's static theory description before evaluation executes."""
    spec = card.get('theory')
    if not spec:
        return None

    root = Path(root)

    def resolve(paths):
        return [
            str((path if (path := Path(item)).is_absolute() else root / path).resolve())
            for item in paths or []
        ]

    statement_links: list[Link] = _links_from_card(spec.get('links'))
    source_links = extract(spec.get('empirical_sources') or [], root=root)
    statement_links.extend(
        link for link in source_links if link.target_kind == 'statement'
    )
    premise_links = [link for link in source_links if link.target_kind == 'premise']

    entries = list(load_indexes(resolve(spec.get('indexes'))))
    entries.extend(parse_entries(spec.get('entries'), 'theory.entries'))
    report = TheoryReport(
        statement_links=statement_links,
        premise_links=premise_links,
        index=TheoryIndex(entries),
    )

    if report.unresolved:
        raise ValueError(
            'theory references with no matching entry or premise: '
            + ', '.join(report.unresolved)
        )

    return report
