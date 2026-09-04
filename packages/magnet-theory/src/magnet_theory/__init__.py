"""Dependency-free annotations connecting empirical code to theory."""
from __future__ import annotations

from types import TracebackType
from typing import Literal, Self, TypeVar

__version__ = '0.0.2'

__all__ = [
    'PREMISE_RELATIONS',
    'RELATIONS',
    'STATEMENT_RELATIONS',
    'TheoryLink',
    'approximates',
    'assumes',
    'checks',
    'ignores',
    'motivates',
    'satisfies',
    'substitutes',
    'tests',
    'violates',
]

STATEMENT_RELATIONS = ('tests', 'approximates', 'motivates')
PREMISE_RELATIONS = (
    'satisfies',
    'approximates',
    'substitutes',
    'assumes',
    'ignores',
    'violates',
    'checks',
)
RELATIONS = tuple(dict.fromkeys(STATEMENT_RELATIONS + PREMISE_RELATIONS))

_T = TypeVar('_T')


class TheoryLink:
    """The inert object returned by every theory annotation."""

    def __init__(self, relation: str, ref: str, note: str = '') -> None:
        self.relation = relation
        self.ref = ref
        self.note = note

    def __call__(self, obj: _T) -> _T:
        return obj

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        if self.note:
            return f'{self.relation}({self.ref!r}, note={self.note!r})'
        return f'{self.relation}({self.ref!r})'


def tests(ref: str, *, note: str = '') -> TheoryLink:
    """Practice directly evaluates the named claim or consequence."""
    return TheoryLink('tests', ref, note)


# pytest collects module-level callables whose names start with ``test``.
setattr(tests, '__test__', False)


def approximates(ref: str, *, note: str = '') -> TheoryLink:
    """Practice measures a finite or proxy version of the named object."""
    return TheoryLink('approximates', ref, note)


def motivates(ref: str, *, note: str = '') -> TheoryLink:
    """Practice establishes a phenomenon theory is asked to explain."""
    return TheoryLink('motivates', ref, note)


def satisfies(ref: str, *, note: str = '') -> TheoryLink:
    """The annotated code is asserted to establish the named premise."""
    return TheoryLink('satisfies', ref, note)


def substitutes(ref: str, *, note: str = '') -> TheoryLink:
    """A different empirical object stands in for the named premise."""
    return TheoryLink('substitutes', ref, note)


def assumes(ref: str, *, note: str = '') -> TheoryLink:
    """The named premise is relied on without being established or checked."""
    return TheoryLink('assumes', ref, note)


def ignores(ref: str, *, note: str = '') -> TheoryLink:
    """The named premise is deliberately left out of the empirical model."""
    return TheoryLink('ignores', ref, note)


def violates(ref: str, *, note: str = '') -> TheoryLink:
    """The named premise is known not to hold for the annotated code."""
    return TheoryLink('violates', ref, note)


def checks(ref: str, *, note: str = '') -> TheoryLink:
    """The annotated code contains a runtime check for the named premise."""
    return TheoryLink('checks', ref, note)
