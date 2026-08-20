"""
Theory says exactly what should happen, and this checks it.

The binomial law gives the probability of every outcome count for a fixed
number of fair flips. Enumerating all 2**n sequences and counting them has to
agree, exactly, with no sampling and no tolerance.
"""
from fractions import Fraction
from itertools import product
from math import comb

import magnet.theory as theory


@theory.tests('Examples.CoinFlip.Binomial')
def enumerated_head_counts(n_flips: int) -> dict:
    """
    Exact probability of each head count, by enumerating every sequence.

    Args:
        n_flips (int): number of fair flips.

    Returns:
        dict: head count -> probability, as a Fraction.

    Example:
        >>> from magnet.examples.theory_links.coin_flip.experiment import enumerated_head_counts
        >>> enumerated_head_counts(2)[1]
        Fraction(1, 2)
    """
    total = 2 ** n_flips
    counts: dict[int, int] = {}
    for sequence in product((0, 1), repeat=n_flips):
        heads = sum(sequence)
        counts[heads] = counts.get(heads, 0) + 1
    return {heads: Fraction(n, total) for heads, n in sorted(counts.items())}


def binomial_probability(n_flips: int, heads: int) -> Fraction:
    """
    What the theorem states: C(n, k) / 2**n for a fair coin.

    Example:
        >>> from magnet.examples.theory_links.coin_flip.experiment import binomial_probability
        >>> binomial_probability(2, 1)
        Fraction(1, 2)
    """
    return Fraction(comb(n_flips, heads), 2 ** n_flips)


def max_absolute_deviation(n_flips: int) -> Fraction:
    """
    Largest gap between the enumeration and the binomial law.

    Zero, for every ``n_flips``. Fractions keep it exactly zero rather than
    nearly zero.

    Example:
        >>> from magnet.examples.theory_links.coin_flip.experiment import max_absolute_deviation
        >>> max_absolute_deviation(8)
        Fraction(0, 1)
    """
    observed = enumerated_head_counts(n_flips)
    return max(
        abs(probability - binomial_probability(n_flips, heads))
        for heads, probability in observed.items()
    )
