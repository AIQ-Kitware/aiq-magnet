"""
The theory side: a list of objects code can point at.

An index is a YAML file naming what exists, with enough structure to say what
kind of thing each entry is::

    entries:
      - id: Examples.CoinFlip.Binomial
        kind: theorem
        statement: >
          P(X = k) = C(n, k) p^k (1-p)^(n-k) for X ~ Binomial(n, p).

      - id: Examples.TrainingOrder.Why
        kind: question
        statement: >
          Why can changing only the order of otherwise identical training
          examples change the learned solution?

``question`` is a first-class kind. An experiment that establishes a phenomenon
points at the question it raises, and a later conjecture or theorem can take
the same id when someone answers it.

An entry may also name where the statement is formalized::

      - id: Dkps.CrossBudgetMAE
        kind: theorem
        declaration: DkpsQuench2026.Paper.TheoryPractice.highProbMAE_queryEfficient
        statement: >
          The cross-budget query-efficiency conclusion in mean absolute error.

``declaration`` is the fully-qualified name in whatever proof assistant states
it, and it is what an index generated from a Lean repository fills in. Reading
proof status out of the kernel, resolving the declaration against a pinned
commit, and accounting for a theorem's individual hypotheses are all built on
top of this field rather than replacing it.
"""
from dataclasses import dataclass
from typing import Sequence

import ubelt as ub
import yaml

__all__ = ['Entry', 'KINDS', 'TheoryIndex', 'load_index', 'load_indexes']

#: What an entry can be. A question becomes a conjecture becomes a theorem, and
#: the id does not have to change as it moves.
KINDS = ('theorem', 'conjecture', 'question')


@dataclass(frozen=True)
class Entry:
    """One theoretical object."""

    id: str
    kind: str
    statement: str = ''

    #: Fully-qualified name of the statement where it is formalized, e.g. a
    #: Lean declaration. Empty when the entry is prose only.
    declaration: str = ''

    def to_dict(self) -> dict:
        data = {'id': self.id, 'kind': self.kind, 'statement': self.statement}
        if self.declaration:
            data['declaration'] = self.declaration
        return data


class TheoryIndex:
    """
    Entries loaded from one or more index files.

    Example:
        >>> from magnet.theory.index import TheoryIndex, Entry
        >>> index = TheoryIndex([Entry('A.b', 'theorem', 'x = y')])
        >>> index['A.b'].kind
        'theorem'
        >>> 'A.c' in index
        False
    """

    def __init__(self, entries: Sequence[Entry] = ()) -> None:
        self._entries = {entry.id: entry for entry in entries}

    def __contains__(self, ref: str) -> bool:
        return ref in self._entries

    def __getitem__(self, ref: str) -> Entry:
        return self._entries[ref]

    def __iter__(self):
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def unresolved(self, refs: Sequence[str]) -> list[str]:
        """
        Which of ``refs`` this index cannot resolve.

        Example:
            >>> from magnet.theory.index import TheoryIndex, Entry
            >>> index = TheoryIndex([Entry('A.b', 'theorem')])
            >>> index.unresolved(['A.b', 'A.c', 'A.c'])
            ['A.c']
        """
        missing = {ref for ref in refs if ref not in self._entries}
        return sorted(missing)

    def to_list(self) -> list[dict]:
        return [entry.to_dict() for entry in self._entries.values()]


def load_index(fpath) -> TheoryIndex:
    """
    Read one index file.

    Args:
        fpath (str | PathLike): a YAML file with an ``entries`` list.

    Returns:
        TheoryIndex

    Raises:
        ValueError: on a missing id, or a kind outside :data:`KINDS`.
    """
    path = ub.Path(fpath)
    data = yaml.safe_load(path.read_text()) or {}
    entries = []
    for raw in data.get('entries', []):
        ref = raw.get('id')
        if not ref:
            raise ValueError(f'{path}: an entry has no id')
        kind = raw.get('kind', 'theorem')
        if kind not in KINDS:
            raise ValueError(
                f'{path}: entry {ref!r} has kind {kind!r}; '
                f'known kinds are {list(KINDS)}')
        entries.append(Entry(
            id=ref,
            kind=kind,
            statement=(raw.get('statement') or '').strip(),
            declaration=raw.get('declaration', ''),
        ))
    return TheoryIndex(entries)


def load_indexes(fpaths: Sequence[str]) -> TheoryIndex:
    """Read several index files into one index."""
    entries: list[Entry] = []
    for fpath in fpaths:
        entries.extend(load_index(fpath))
    return TheoryIndex(entries)
