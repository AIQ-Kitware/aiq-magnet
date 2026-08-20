"""
How a piece of empirical code relates to a theoretical object.

Three relations, read as ``practice <relation> theory``::

    tests          theory says exactly what should happen; this checks it
    approximates   theory defines something exact; this estimates it
    motivates      this establishes a phenomenon; theory is asked to explain it

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

__all__ = ['TheoryLink', 'RELATIONS', 'tests', 'approximates', 'motivates']

#: The relation names, in the order they appear in the module docstring.
RELATIONS = ('tests', 'approximates', 'motivates')


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

    def __init__(self, relation: str, ref: str) -> None:
        self.relation = relation
        self.ref = ref

    def __call__(self, obj):
        return obj

    def __enter__(self) -> 'TheoryLink':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def __repr__(self) -> str:
        return f'{self.relation}({self.ref!r})'


def tests(ref: str) -> TheoryLink:
    """
    Theory says exactly what should happen, and this checks it.

    Args:
        ref (str): id of an entry in a theory index.

    Example:
        >>> from magnet.theory.links import tests
        >>> tests('Examples.CoinFlip.Binomial')
        tests('Examples.CoinFlip.Binomial')
    """
    return TheoryLink('tests', ref)


#: pytest collects any module-level callable whose name starts with `test`, so
#: `from magnet.theory import tests` in a test file turns the relation itself
#: into a collected test and errors on its `ref` argument as a missing fixture.
tests.__test__ = False


def approximates(ref: str) -> TheoryLink:
    """
    Theory defines something exact, and this estimates it.

    Args:
        ref (str): id of an entry in a theory index.

    Example:
        >>> from magnet.theory.links import approximates
        >>> approximates('Examples.Dice.SumSevenProbability')
        approximates('Examples.Dice.SumSevenProbability')
    """
    return TheoryLink('approximates', ref)


def motivates(ref: str) -> TheoryLink:
    """
    This establishes a phenomenon, and theory is asked to explain it.

    Args:
        ref (str): id of an entry in a theory index, usually a question.

    Example:
        >>> from magnet.theory.links import motivates
        >>> motivates('Examples.TrainingOrder.Why')
        motivates('Examples.TrainingOrder.Why')
    """
    return TheoryLink('motivates', ref)
