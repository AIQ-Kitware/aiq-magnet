"""
A runtime phenomenon with a known structural explanation.

Two Python functions compute the same Fibonacci number. The naive recursive
implementation is much slower for a moderate input even though its source is
short. If we did not know why, that empirical observation would naturally
motivate a theory question.

The companion Lean file does not claim to prove anything about CPython wall
clock time. It formalizes a simple operation-count model: naive recursion makes
many repeated calls, while iteration takes one loop step per input index. The
wall-clock benchmark is therefore annotated as an approximation to that
abstract cost gap, not as a direct test of a timing theorem.
"""
import json
import statistics
import time

import kwconf
import kwutil
import ubelt as ub

import magnet.theory as theory


class FibonacciPerformanceCLI(kwconf.Config):
    """Benchmark the two implementations and write a JSON summary."""

    __command__ = 'fibonacci_performance'

    repeats: int = kwconf.Value(5, help='timing repetitions per implementation')
    out_fpath: str = kwconf.Value(
        'fibonacci_performance.json',
        help='where to write the result',
        tags=['out_path', 'primary'],
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose='auto')

        context = kwutil.ProcessContext(
            name='fibonacci_performance',
            type='process',
            config=kwutil.Json.ensure_serializable(dict(config)),
            track_emissions=False,
        )
        context.start()
        data = {
            'result': {'metrics': benchmark_fibonacci(config['repeats'])},
            'info': [context.stop()],
        }
        out_fpath = ub.Path(config['out_fpath'])
        out_fpath.parent.ensuredir()
        out_fpath.write_text(json.dumps(data, indent=2))


def fibonacci_recursive(n: int) -> int:
    """Naive recursive Fibonacci, intentionally without memoization."""
    if n < 2:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_iterative(n: int) -> int:
    """Compute the same value with one loop step per input index."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def recursive_call_count(n: int) -> int:
    """Exact call count for :func:`fibonacci_recursive` without doing the calls."""
    if n < 2:
        return 1
    c0, c1 = 1, 1
    for _ in range(2, n + 1):
        c0, c1 = c1, 1 + c1 + c0
    return c1


def iterative_step_count(n: int) -> int:
    """Loop iterations executed by :func:`fibonacci_iterative`."""
    return n


def _median_runtime_ns(func, n: int, repeats: int) -> int:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        func(n)
        samples.append(time.perf_counter_ns() - start)
    return int(statistics.median(samples))


@theory.motivates('Examples.FibonacciPerformance.Why')
def benchmark_fibonacci(repeats: int = 5) -> dict:
    """
    Compare equal-result implementations and measure the runtime gap.

    The empirical observation motivates the open question. The timing
    measurements approximate the abstract cost result because elapsed time is
    only a proxy for the amount of algorithmic work.

    Example:
        >>> result = benchmark_fibonacci(repeats=1)
        >>> result['same_value']
        True
        >>> result['recursive_slower']
        True
        >>> result['abstract_cost_gap']
        True
    """
    n = 28
    if repeats < 1:
        raise ValueError('repeats must be at least 1')

    recursive_value = fibonacci_recursive(n)
    iterative_value = fibonacci_iterative(n)

    with theory.approximates(
        'Examples.FibonacciPerformance.RecursiveCallGapAt28',
        note=(
            'wall-clock runtime is a proxy for the abstract operation-count '
            'gap; the formal result does not model CPython timing'
        ),
    ):
        recursive_ns = _median_runtime_ns(fibonacci_recursive, n, repeats)
        iterative_ns = _median_runtime_ns(fibonacci_iterative, n, repeats)

    recursive_calls = recursive_call_count(n)
    iterative_steps = iterative_step_count(n)
    return {
        'n': n,
        'repeats': repeats,
        'recursive_value': recursive_value,
        'iterative_value': iterative_value,
        'same_value': recursive_value == iterative_value,
        'recursive_runtime_ns': recursive_ns,
        'iterative_runtime_ns': iterative_ns,
        'recursive_slower': recursive_ns > iterative_ns,
        'speed_ratio': recursive_ns / max(iterative_ns, 1),
        'recursive_calls': recursive_calls,
        'iterative_steps': iterative_steps,
        'abstract_cost_gap': recursive_calls > 1000 * iterative_steps,
    }


__cli__ = FibonacciPerformanceCLI

if __name__ == '__main__':
    __cli__.main()
