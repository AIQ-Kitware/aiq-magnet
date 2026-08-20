"""
A dependency-free stand-in for the three relations.

Copy this file into your repository as ``magnet_theory.py`` and annotate with
it. Nothing here imports anything, and every relation returns its target
unchanged, so your code runs the same whether or not anyone is reading it. The
annotations are collected from your source.

    import magnet_theory as theory

    @theory.tests('Examples.CoinFlip.Binomial')
    def exact_tail_probability(n, k):
        ...

    with theory.approximates('Examples.Dice.SumSevenProbability'):
        ...
"""

__all__ = ['tests', 'approximates', 'motivates']


class _Link:
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


def tests(ref, *, note=''):
    """Practice directly evaluates the named claim or consequence."""
    return _Link('tests', ref, note)


def approximates(ref, *, note=''):
    """Practice measures a finite or proxy version of theory."""
    return _Link('approximates', ref, note)


def motivates(ref, *, note=''):
    """Practice establishes a phenomenon theory is asked to explain."""
    return _Link('motivates', ref, note)
