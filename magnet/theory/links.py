"""
How a piece of empirical code relates to a theoretical object.

Three relations, read as ``practice <relation> theory``::

    tests          practice directly evaluates the claim or consequence
    approximates   practice measures a finite or proxy version of theory
    motivates      practice establishes a phenomenon theory should explain

Each works as a decorator or as a context manager, and each is a no-op at
runtime: it returns what it wraps and records nothing. Annotations are read
from source by :mod:`magnet.theory.static`, so annotated code runs the same
whether or not anyone is reading it.

.. code:: python

    import magnet.theory as theory

    @theory.tests('Examples.CoinFlip.Binomial')
    def exact_tail_probability(n, k):
        ...

    def estimate(trials):
        with theory.approximates('Examples.Dice.SumSevenProbability'):
            ...

``motivates`` is the one that points at an open question. An experiment that
shows a phenomenon does not have to explain it; naming the question is what
lets the explanation arrive later.
"""
from dataclasses import dataclass

__all__ = ['Link', 'TheoryLink', 'RELATIONS', 'tests', 'approximates', 'motivates']

#: The relation names, in the order they appear in the module docstring.
RELATIONS = ('tests', 'approximates', 'motivates')


@dataclass(frozen=True)
class Link:
    """One serialized ``practice <relation> theory`` connection."""

    relation: str
    ref: str
    note: str = ''
    file: str = ''
    line: int | None = None
    qualname: str = ''

    def to_dict(self) -> dict:
        data = {
            'relation': self.relation,
            'ref': self.ref,
        }
        if self.note:
            data['note'] = self.note
        if self.file:
            data['file'] = self.file
        if self.line is not None:
            data['line'] = self.line
        if self.qualname:
            data['qualname'] = self.qualname
        return data


class TheoryLink:
    """
    The object a relation returns.

    Usable as a decorator or a context manager, and inert in both roles.

    Example:
        >>> from magnet.theory.links import tests
        >>> @tests('Examples.CoinFlip.Binomial')
        ... def experiment():
        ...     return 42
        >>> experiment()
        42
        >>> with tests('Examples.CoinFlip.Binomial') as link:
        ...     link.relation, link.ref
        ('tests', 'Examples.CoinFlip.Binomial')
    """

    def __init__(self, relation: str, ref: str, note: str = '') -> None:
        self.relation = relation
        self.ref = ref
        self.note = note

    def __call__(self, obj):
        return obj

    def __enter__(self) -> 'TheoryLink':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def __repr__(self) -> str:
        if self.note:
            return f'{self.relation}({self.ref!r}, note={self.note!r})'
        return f'{self.relation}({self.ref!r})'


def tests(ref: str, *, note: str = '') -> TheoryLink:
    """
    Practice directly evaluates the claim or consequence represented here.

    This relation does not assert that every hypothesis of a theorem applies;
    hypothesis accounting is a separate refinement layer.

    Args:
        ref (str): id of an entry in a theory index.
        note (str): optional short explanation of the connection.

    Example:
        >>> from magnet.theory.links import tests
        >>> tests('Examples.CoinFlip.Binomial')
        tests('Examples.CoinFlip.Binomial')
    """
    return TheoryLink('tests', ref, note)


#: pytest collects any module-level callable whose name starts with `test`, so
#: `from magnet.theory import tests` in a test file turns the relation itself
#: into a collected test and errors on its `ref` argument as a missing fixture.
tests.__test__ = False


def approximates(ref: str, *, note: str = '') -> TheoryLink:
    """
    Practice measures a finite, sampled, or proxy version of theory.

    Args:
        ref (str): id of an entry in a theory index.
        note (str): optional short explanation of what is approximated.

    Example:
        >>> from magnet.theory.links import approximates
        >>> approximates('Examples.Dice.SumSevenProbability')
        approximates('Examples.Dice.SumSevenProbability')
    """
    return TheoryLink('approximates', ref, note)


def motivates(ref: str, *, note: str = '') -> TheoryLink:
    """
    Practice establishes a phenomenon, and theory is asked to explain it.

    Args:
        ref (str): id of an entry in a theory index, usually a question.
        note (str): optional short explanation of the motivating observation.

    Example:
        >>> from magnet.theory.links import motivates
        >>> motivates('Examples.TrainingOrder.Why')
        motivates('Examples.TrainingOrder.Why')
    """
    return TheoryLink('motivates', ref, note)
