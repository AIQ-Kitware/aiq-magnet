"""
Static annotations describing how empirical code relates to theory.

The public API is the annotation vocabulary. Statement annotations connect
practice to a theoretical object; premise annotations explain how empirical
code treats one named premise of that object. All annotations are runtime
no-ops and are read from source by MAGNET.

Example:
    >>> import magnet.theory as theory
    >>> @theory.tests('Examples.Stability.Theorem')
    ... @theory.assumes('Examples.Stability.Theorem::hiid')
    ... def experiment():
    ...     return 42
    >>> experiment()
    42
"""
from magnet.theory.annotations import (
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
