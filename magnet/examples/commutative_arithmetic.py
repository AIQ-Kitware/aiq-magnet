"""
The same evaluation as ``magnet/cards/simple.yaml``, written as a recipe.

This module exists to be read side by side with that card. It is a deliberate
line-for-line port and not a showcase: same title, same symbols, same claim,
same verdict. What differs is only how the program is expressed, which is the
thing under review.

The YAML says::

    symbols:
      int_range_even:
        metadata:
          display_name: "Set of Even Numbers"
        type: List[int]
        python: |
          def create_even_range(start=-10, end=10):
            return [n for n in range(start, end+1) if n % 2 == 0]
          int_range_even = create_even_range()

      int_range_odd:
        type: List[int]
        depends_on:
          - int_range_even
        python: |
          int_range_odd = [n + 1 for n in int_range_even]

Three things are worth noticing in the Python below. The body is code rather
than a string, so a typo in it is a syntax error at import instead of a
traceback out of ``exec`` at run time. ``int_range_odd`` declares its dependency
by taking ``int_range_even`` as a parameter, so the graph cannot disagree with
the code the way a hand-maintained ``depends_on`` list can. And the YAML block
has to end by binding its own name -- ``int_range_even = create_even_range()``
-- where the function just returns.

What comes back is the other half of the change: :meth:`evaluate` returns an
:class:`~magnet.card.EvaluationResultCard`, a result rather than the program
that produced it, whereas :class:`magnet.evaluation.EvaluationCard` is loaded
from the YAML and is both at once.

Run it::

    python -m magnet.examples.commutative_arithmetic
"""
from magnet.recipe import claim, recipe, symbol


def create_even_range(start: int = -10, end: int = 10) -> list[int]:
    """
    The helper the YAML card defines inside its ``python:`` block.

    Out here it is an ordinary function: importable, type-annotated, and
    reachable by a test. Inside the block it was three lines of string that no
    tool could see.

    Example:
        >>> create_even_range(0, 5)
        [0, 2, 4]
    """
    return [n for n in range(start, end + 1) if n % 2 == 0]


@recipe(
    title='Arithmetic - Addition Commutative Property',
    description='Addition is commutative on pairs of even and odd integers',
    category='Mathematical Properties',
    version='1.0',
    organizations=['Kitware'],
    submitter={'name': 'Kitware TA2 Team', 'email': 'aiq-ta2@kitware.com'},
    tags=['example'],
    links=[
        {
            'title': 'MAGNET',
            'url': 'https://github.com/AIQ-Kitware/aiq-magnet',
            'type': 'software',
        }
    ],
)
class CommutativeAddition:
    @symbol(type='List[int]', display_name='Set of Even Numbers')
    def int_range_even():
        return create_even_range()

    @symbol(type='List[int]', display_name='Set of Odd Numbers')
    def int_range_odd(int_range_even):
        return [n + 1 for n in int_range_even]

    @claim
    def commutes(int_range_even, int_range_odd):
        for even, odd in zip(int_range_even, int_range_odd):
            assert even + odd == odd + even, f'{even} + {odd} is not commutative'


def main() -> None:
    CommutativeAddition.evaluate().summarize()


if __name__ == '__main__':
    main()
