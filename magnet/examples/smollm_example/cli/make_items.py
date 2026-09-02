"""
Write the dummy dataset this example asks about.

Contrived on purpose. A real benchmark would bring a download, a licence and a
scoring convention, and none of those are what this example is demonstrating.
Single-digit addition is enough to have a right answer, a wrong answer and a
non-answer, which is all the comparison downstream needs.

CommandLine:
    python -m magnet.examples.smollm_example.cli.make_items \
        --n_items=8 --out_fpath=items.json
"""

import json
import random

import kwconf
import ubelt as ub

__all__ = ['build_items', 'MakeItemsCLI']


def build_items(n_items: int, seed: int) -> list:
    """
    The dataset, derived from a seed so two runs ask the same questions.

    Args:
        n_items (int): how many questions to write.
        seed (int): what makes the set reproducible.

    Returns:
        list: dicts with ``id``, ``prompt`` and ``expected``.

    Example:
        >>> from magnet.examples.smollm_example.cli.make_items import (
        ...     build_items)
        >>> items = build_items(3, seed=0)
        >>> len(items)
        3
        >>> build_items(3, seed=0) == items
        True
        >>> sorted(items[0])
        ['expected', 'id', 'prompt']
    """
    rng = random.Random(seed)
    items = []
    for index in range(n_items):
        left = rng.randint(1, 9)
        right = rng.randint(1, 9)
        items.append({
            'id': index,
            'prompt': (
                f'What is {left} + {right}? '
                'Reply with the number only, no words.'
            ),
            'expected': str(left + right),
        })
    return items


class MakeItemsCLI(kwconf.Config):
    """Write the dummy dataset both endpoints will be asked about."""

    n_items: int = kwconf.Value(8, help='how many questions to ask')
    seed: int = kwconf.Value(0, help='makes the question set reproducible')
    out_fpath: str = kwconf.Value(
        'items.json', help='where to write the dataset',
        tags=['out_path', 'primary'],
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose='auto')
        items = build_items(int(config['n_items']), int(config['seed']))
        payload = {
            # `result.metrics` is where kwdagger's generic loader looks, so
            # this node needs no `load_result` of its own.
            'result': {'metrics': {
                'n_items': len(items),
                'seed': int(config['seed']),
            }},
            'items': items,
        }
        out_fpath = ub.Path(config['out_fpath'])
        out_fpath.parent.ensuredir()
        out_fpath.write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    MakeItemsCLI.main()
