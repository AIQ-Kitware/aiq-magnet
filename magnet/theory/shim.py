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
    def __init__(self, relation, ref):
        self.relation = relation
        self.ref = ref

    def __call__(self, obj):
        return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def tests(ref):
    """Theory says exactly what should happen, and this checks it."""
    return _Link('tests', ref)


def approximates(ref):
    """Theory defines something exact, and this estimates it."""
    return _Link('approximates', ref)


def motivates(ref):
    """This establishes a phenomenon, and theory is asked to explain it."""
    return _Link('motivates', ref)
