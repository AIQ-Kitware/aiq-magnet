"""
Dependency-free annotations connecting empirical code to theory.

Copy this file into a repository as ``magnet_theory.py`` when that repository
should not depend on MAGNET. The annotations are intentionally inert at
runtime: decorators return their target unchanged and context managers only
provide the annotation object to an optional ``as`` clause.

Statement relations read as ``practice <relation> theory``::

    @theory.tests('Examples.Stability.Conclusion')
    @theory.approximates('Examples.Stability.PopulationClaim')
    @theory.motivates('Examples.Stability.OpenQuestion')

Premise relations describe how code treats a named premise of a statement::

    @theory.satisfies('Examples.Stability.Theorem::hbounded')
    @theory.assumes('Examples.Stability.Theorem::hiid')

Premise references use ``EntryId::binder``. MAGNET resolves the entry through a
theory index and the binder against the premises exported for that entry.
Nothing in this module imports MAGNET or records runtime values.
"""

__all__ = [
    'tests',
    'approximates',
    'motivates',
    'satisfies',
    'substitutes',
    'assumes',
    'ignores',
    'violates',
    'checks',
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


class TheoryLink:
    """The inert object returned by every theory annotation."""

    def __init__(self, relation, ref, note=''):
        self.relation = relation
        self.ref = ref
        self.note = note

    def __call__(self, obj):
        return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __repr__(self):
        if self.note:
            return f'{self.relation}({self.ref!r}, note={self.note!r})'
        return f'{self.relation}({self.ref!r})'


def tests(ref, *, note=''):
    """Practice directly evaluates the named claim or consequence."""
    return TheoryLink('tests', ref, note)


# pytest collects module-level callables whose names start with ``test``.
tests.__test__ = False


def approximates(ref, *, note=''):
    """Practice measures a finite or proxy version of the named object."""
    return TheoryLink('approximates', ref, note)


def motivates(ref, *, note=''):
    """Practice establishes a phenomenon theory is asked to explain."""
    return TheoryLink('motivates', ref, note)


def satisfies(ref, *, note=''):
    """The annotated code is asserted to establish the named premise."""
    return TheoryLink('satisfies', ref, note)


def substitutes(ref, *, note=''):
    """A different empirical object stands in for the named premise."""
    return TheoryLink('substitutes', ref, note)


def assumes(ref, *, note=''):
    """The named premise is relied on without being established or checked."""
    return TheoryLink('assumes', ref, note)


def ignores(ref, *, note=''):
    """The named premise is deliberately left out of the empirical model."""
    return TheoryLink('ignores', ref, note)


def violates(ref, *, note=''):
    """The named premise is known not to hold for the annotated code."""
    return TheoryLink('violates', ref, note)


def checks(ref, *, note=''):
    """The annotated code contains a runtime check for the named premise."""
    return TheoryLink('checks', ref, note)
