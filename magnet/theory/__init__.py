"""Static theory links, indexes, validation, and reporting for MAGNET.

The annotation vocabulary itself lives in the dependency-free
``magnet-theory`` distribution. This module reexports those runtime no-ops for
compatibility while MAGNET provides the machinery that interprets them.

Example:
    >>> import magnet.theory as theory
    >>> @theory.tests('Examples.Stability.Theorem')
    ... @theory.assumes('Examples.Stability.Theorem::hiid')
    ... def experiment():
    ...     return 42
    >>> experiment()
    42
"""
from __future__ import annotations

from magnet_theory import (
    approximates,
    assumes,
    checks,
    ignores,
    motivates,
    satisfies,
    substitutes,
    tests,
    violates,
)

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
