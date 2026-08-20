"""
Theory defines something exact, and this estimates it.

A quarter of the unit disc occupies pi/4 of the unit square. Sampling points
and counting how many land inside estimates that ratio, and the estimate
carries error the closed form does not.

The sampler is a seeded linear congruential generator written out here, so the
estimate is identical on every machine and every Python version.
"""
import json
from math import pi

import kwconf

import magnet.theory as theory

class MonteCarloCLI(kwconf.Config):
    """
    Estimate the quarter-disc area ratio from a seeded sample.

    Already a kwdagger node executable, so porting the card is a block of YAML
    rather than a restructure.

    CommandLine:
        python -m magnet.examples.theory_links.monte_carlo.experiment \
            --seed=1 --samples=20000 --out_fpath=monte_carlo.json
    """

    __command__ = 'monte_carlo'

    seed: int = kwconf.Value(1, help='LCG seed; one job per value')
    samples: int = kwconf.Value(20000, help='points to draw')
    out_fpath: str = kwconf.Value(
        'monte_carlo.json', help='where to write the result',
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose='auto')
        seed = int(config['seed'])
        samples = int(config['samples'])
        payload = {
            'seed': seed,
            'samples': samples,
            'exact': exact_area_ratio(),
            'estimate': estimate_area_ratio(seed, samples),
            'pi': estimate_pi(seed, samples),
            'error': estimation_error(seed, samples),
        }
        with open(config['out_fpath'], 'w') as file:
            json.dump(payload, file, indent=2)

#: Numerical Recipes constants. Written out so this file depends on nothing.
_LCG_MODULUS = 2 ** 32
_LCG_MULTIPLIER = 1664525
_LCG_INCREMENT = 1013904223


def _unit_interval(seed: int, count: int):
    """A deterministic stream of values in [0, 1)."""
    state = seed % _LCG_MODULUS
    for _ in range(count):
        state = (_LCG_MULTIPLIER * state + _LCG_INCREMENT) % _LCG_MODULUS
        yield state / _LCG_MODULUS


def exact_area_ratio() -> float:
    """
    What the theorem states: the quarter disc is pi/4 of the unit square.

    Example:
        >>> from magnet.examples.theory_links.monte_carlo.experiment import exact_area_ratio
        >>> round(exact_area_ratio(), 6)
        0.785398
    """
    return pi / 4


def estimate_area_ratio(seed: int = 1, samples: int = 20000) -> float:
    """
    Estimate the same ratio by sampling points in the unit square.

    Args:
        seed (int): LCG seed; the same seed always gives the same estimate.
        samples (int): points to draw.

    Returns:
        float: the fraction landing inside the quarter disc.

    Example:
        >>> from magnet.examples.theory_links.monte_carlo.experiment import estimate_area_ratio
        >>> round(estimate_area_ratio(seed=1, samples=1000), 4)
        0.791
    """
    with theory.approximates('Examples.Circle.AreaRatio'):
        stream = _unit_interval(seed, samples * 2)
        inside = sum(1 for x, y in zip(stream, stream) if x * x + y * y <= 1.0)
        return inside / samples


def estimate_pi(seed: int = 1, samples: int = 20000) -> float:
    """
    The same estimate, read as pi.

    Example:
        >>> from magnet.examples.theory_links.monte_carlo.experiment import estimate_pi
        >>> round(estimate_pi(seed=1, samples=20000), 3)
        3.157
    """
    return 4 * estimate_area_ratio(seed, samples)


def estimation_error(seed: int = 1, samples: int = 20000) -> float:
    """
    How far this sample lands from the exact ratio.

    Positive for any finite sample, which is what separates this from the
    coin-flip example.

    Example:
        >>> from magnet.examples.theory_links.monte_carlo.experiment import estimation_error
        >>> estimation_error(seed=1, samples=20000) > 0
        True
    """
    return abs(estimate_area_ratio(seed, samples) - exact_area_ratio())


__cli__ = MonteCarloCLI

if __name__ == '__main__':
    __cli__.main()
