"""
Theory says exactly what should happen, and this checks it.

The binomial law gives the probability of every outcome count for a fixed
number of fair flips. Enumerating all 2**n sequences and counting them has to
agree, exactly, with no sampling and no tolerance.
"""
import json
from fractions import Fraction
from itertools import product
from math import comb

import kwconf
import ubelt as ub

import magnet_theory as theory


class CoinFlipCLI(kwconf.Config):
    r"""
    Enumerate every sequence of flips and compare against the binomial law.

    Already a kwdagger node executable, so porting the card is a block of YAML
    rather than a restructure.

    CommandLine:
        python -m magnet.examples.theory_links.coin_flip.experiment \
            --n_flips=10 --out_fpath=coin_flip.json
    """

    n_flips: int = kwconf.Value(10, help='number of fair flips to enumerate')
    out_fpath: str = kwconf.Value(
        'coin_flip.json', help='where to write the result',
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose='auto')
        n_flips = int(config['n_flips'])

        data = {
            'result': {
                'metrics': {
                    'n_flips': n_flips,
                    'outcomes': 2 ** n_flips,
                    'deviation': float(max_absolute_deviation(n_flips)),
                },
            },
        }
        out_fpath = ub.Path(config['out_fpath'])
        out_fpath.parent.ensuredir()
        out_fpath.write_text(json.dumps(data, indent=2))

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


__cli__ = CoinFlipCLI

if __name__ == '__main__':
    __cli__.main()
