"""
Record how empirical code relates to a theoretical object.

Annotate the code with one of three relations, each read as
``practice <relation> theory``::

    import magnet.theory as theory

    @theory.tests('Examples.CoinFlip.Binomial')
    def exact_tail_probability(n, k):
        ...

    @theory.motivates('Examples.TrainingOrder.Why')
    def training_order_sensitivity(examples):
        ...

Name the theoretical objects in an index::

    entries:
      - id: Examples.CoinFlip.Binomial
        kind: theorem
        statement: ...

      - id: Examples.TrainingOrder.Why
        kind: question
        statement: Why can training order change the learned solution?

Then point a card at them. The card can also link its overall evaluation claim
directly, while source annotations describe implementation sites::

    theory:
      links:
        - relation: tests
          ref: Examples.CoinFlip.Binomial
      sources: [../examples/theory_links/coin_flip.py]
      indexes: [../examples/theory_links/theory.yaml]

Evaluating it writes ``theory.json`` beside the verdict.

Annotations are collected from source, so nothing here executes to be read, and
a team can annotate with :mod:`magnet.theory.shim` instead of depending on
MAGNET.
"""
from magnet.theory.index import (
    KINDS,
    Entry,
    TheoryIndex,
    load_index,
    load_indexes,
)
from magnet.theory.links import (
    RELATIONS,
    Link,
    TheoryLink,
    approximates,
    motivates,
    tests,
)
from magnet.theory.static import extract, extract_tree

__all__ = [
    # relations
    'tests',
    'approximates',
    'motivates',
    'TheoryLink',
    'RELATIONS',
    # extraction
    'Link',
    'extract',
    'extract_tree',
    # index
    'Entry',
    'TheoryIndex',
    'KINDS',
    'load_index',
    'load_indexes',
]
