Linking Practice to Theory
==========================

An evaluation produces a number. What that number has to do with a theorem, a
conjecture, or an open question usually lives in someone's head, or in a paper
nobody reads next to the code. A theory link writes it down next to the code
that does the work, and carries it into the run's output.

There are three relations, each read as ``practice <relation> theory``:

===============  ==========================================================
``tests``        theory says exactly what should happen; this checks it
``approximates`` theory defines something exact; this estimates it
``motivates``    this establishes a phenomenon; theory is asked to explain it
===============  ==========================================================

``motivates`` is the one people miss. An experiment does not have to confirm or
estimate anything to be worth connecting: showing that a phenomenon exists
gives theory something to explain, and naming the question is what lets the
explanation arrive later.


Annotating code
---------------

Import the module and use a relation as a decorator:

.. code:: python

    import magnet.theory as theory

    @theory.tests('Examples.CoinFlip.Binomial')
    def enumerated_head_counts(n_flips):
        ...

or as a context manager, when the part that relates to the theory is smaller
than a function:

.. code:: python

    def estimate_area_ratio(seed, samples):
        with theory.approximates('Examples.Circle.AreaRatio'):
            ...

Both forms do nothing at runtime. A relation returns what it wraps, so
annotated code behaves identically whether or not anyone is reading the
annotations.


Naming the theory
-----------------

References point at entries in an index:

.. code:: yaml

    entries:
      - id: Examples.CoinFlip.Binomial
        kind: theorem
        statement: >
          For n independent fair flips, the probability of exactly k heads is
          C(n, k) / 2^n.

      - id: Examples.TrainingOrder.Why
        kind: question
        statement: >
          Why can changing only the order of otherwise identical training
          observations change the learned solution?

Three kinds are available: ``theorem``, ``conjecture`` and ``question``. An id
can keep its name as the object behind it develops, so code that points at a
question keeps working when someone turns that question into a conjecture and
then proves it.


Connecting it to a card
-----------------------

A card names the annotated source and the indexes its references live in. Both
are relative to the card:

.. code:: yaml

    theory:
      sources:
        - ../examples/theory_links/coin_flip.py
      indexes:
        - ../examples/theory_links/theory.yaml

Evaluating the card reads the source, reads the indexes, checks that every
reference resolves, and writes ``theory.json`` beside ``verdict.json``:

.. code:: json

    {
      "links": [
        {
          "relation": "motivates",
          "ref": "Examples.TrainingOrder.Why",
          "file": ".../training_order.py",
          "line": 72,
          "qualname": "training_order_sensitivity"
        }
      ],
      "entries": [
        {
          "id": "Examples.TrainingOrder.Why",
          "kind": "question",
          "statement": "Why can changing only the order of ..."
        }
      ]
    }

The entries a run points at travel with the links, so the artifact reads on its
own without the index beside it. A reference with no entry in any declared
index raises rather than passing silently.


Annotating without depending on MAGNET
--------------------------------------

Annotations are collected from **source**, so nothing has to import MAGNET for
them to be read. A repository that would rather not take the dependency can
copy ``magnet/theory/shim.py`` in as ``magnet_theory.py``:

.. code:: python

    import magnet_theory as theory

    @theory.tests('Examples.CoinFlip.Binomial')
    def experiment():
        ...

The shim implements the same three relations as no-ops and imports nothing.
The extractor reads that spelling exactly as it reads the real one.


What counts as an annotation
----------------------------

The extractor accepts one spelling, deliberately. A relation call is recorded
when all of the following hold:

1. the module is imported as ``import magnet.theory as theory`` or
   ``import magnet_theory as theory`` (any alias works; the full
   ``magnet.theory.tests(...)`` is also accepted)
2. the call is ``<alias>.tests``, ``.approximates`` or ``.motivates``
3. the first argument is a literal string
4. the call is a decorator or a ``with`` item

A variable holding the reference, a concatenated string, an unrecognized
relation, or a bare call in a function body is ignored. Keeping the accepted
set this small means the rule fits in your head, and an annotation either
appears in ``theory.json`` or does not exist.


Worked examples
---------------

Three cards ship with MAGNET, one per relation. Each runs offline in a fraction
of a second:

===================================  =================  ======================
card                                 relation           what it shows
===================================  =================  ======================
``theory_coin_flip_exact.yaml``      ``tests``          enumeration matches the
                                                        binomial law exactly
``theory_monte_carlo.yaml``          ``approximates``   sampling estimates
                                                        pi/4 with error
``theory_training_order.yaml``       ``motivates``      reordering the same
                                                        data changes the
                                                        learned solution
===================================  =================  ======================

Run one with:

.. code:: bash

    python -m magnet.evaluation magnet/cards/theory_training_order.yaml

and read ``theory.json`` in the run directory.
