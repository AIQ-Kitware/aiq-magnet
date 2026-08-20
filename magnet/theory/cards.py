"""
The card side: read a card's theory block and produce ``theory.json``.

A card names the annotated source, and either writes its theoretical objects
out or names files holding them::

    theory:
      sources:
        - experiment.py
      entries:
        - id: Examples.CoinFlip.Binomial
          kind: theorem
          statement: ...

    theory:
      sources:
        - experiment.py
      indexes:
        - ../../theory/indexes/dkps-144de76c.yaml

Inline suits a card with one or two objects of its own. A file suits an index
generated from a formalization, where a card points at two entries out of
fifty. Both may appear together, and paths are relative to the card.

Evaluating the card reads the source, collects the entries, checks that every
reference resolves, and writes the links beside the verdict. The annotation in
the code is the whole relationship; the card only says where to look.
"""
import json
from dataclasses import dataclass, field

import ubelt as ub

from magnet.theory.index import TheoryIndex, load_indexes, parse_entries
from magnet.theory.static import Link, extract

__all__ = ['TheoryReport', 'report_from_card']


@dataclass
class TheoryReport:
    """The links a card declares, and the entries they point at."""

    links: list = field(default_factory=list)
    index: TheoryIndex = field(default_factory=TheoryIndex)

    @property
    def unresolved(self) -> list:
        """References with no entry in any of the card's indexes."""
        return self.index.unresolved([link.ref for link in self.links])

    def to_dict(self) -> dict:
        """
        The ``theory.json`` payload.

        Only the entries something points at are carried, so the artifact
        describes this run rather than the whole index.
        """
        referenced = {link.ref for link in self.links}
        data = {
            'links': [link.to_dict() for link in self.links],
            'entries': [entry.to_dict() for entry in self.index
                        if entry.id in referenced],
        }
        if self.unresolved:
            data['unresolved'] = self.unresolved
        return data

    def write(self, fpath) -> None:
        path = ub.Path(fpath)
        path.parent.ensuredir()
        path.write_text(json.dumps(self.to_dict(), indent=2) + '\n')


def report_from_card(card: dict, root) -> TheoryReport | None:
    """
    Build the report for a card, or None when it declares no theory.

    Args:
        card (dict): the parsed card.
        root (str | PathLike): the directory holding the card; relative
            ``sources`` and ``indexes`` resolve against it.

    Returns:
        TheoryReport | None

    Raises:
        ValueError: if a reference has no entry in any declared index.

    Example:
        >>> import ubelt as ub
        >>> from magnet.theory.cards import report_from_card
        >>> dpath = ub.Path.appdir('magnet/tests/theory_cards').delete().ensuredir()
        >>> (dpath / 'demo.py').write_text(ub.codeblock(
        ...     '''
        ...     import magnet.theory as theory
        ...
        ...     @theory.tests('A.b')
        ...     def experiment():
        ...         pass
        ...     '''))
        >>> card = {'theory': {'sources': ['demo.py'], 'entries': [
        ...     {'id': 'A.b', 'kind': 'theorem', 'statement': 'the statement'}]}}
        >>> report = report_from_card(card, dpath)
        >>> report.to_dict()['links'][0]['relation']
        'tests'
        >>> report.to_dict()['entries']
        [{'id': 'A.b', 'kind': 'theorem', 'statement': 'the statement'}]
    """
    spec = card.get('theory')
    if not spec:
        return None

    root = ub.Path(root)

    def resolve(entries):
        # Normalized, so a card written as ../examples/... does not put a
        # `cards/../examples/...` path into the artifact.
        return [str((path if (path := ub.Path(entry)).is_absolute()
                     else root / path).resolve())
                for entry in entries or []]

    links: list[Link] = extract(resolve(spec.get('sources')))
    entries = list(load_indexes(resolve(spec.get('indexes'))))
    entries.extend(parse_entries(spec.get('entries'), 'theory.entries'))
    report = TheoryReport(links=links, index=TheoryIndex(entries))

    if report.unresolved:
        raise ValueError(
            'theory references with no entry in the card\'s indexes: '
            + ', '.join(report.unresolved))

    return report
