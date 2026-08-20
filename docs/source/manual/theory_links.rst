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

References point at entries, which a card can write out directly:

.. code:: yaml

    theory:
      sources:
        - experiment.py
      entries:
        - id: Examples.CoinFlip.Binomial
          kind: theorem
          statement: >
            For n independent fair flips, the probability of exactly k heads
            is C(n, k) / 2^n.

        - id: Examples.TrainingOrder.Why
          kind: question
          statement: >
            Why can changing only the order of otherwise identical training
            observations change the learned solution?

or read from index files, which is what an index generated from a
formalization looks like:

.. code:: yaml

    theory:
      sources:
        - experiment.py
      indexes:
        - ../../theory/indexes/dkps-144de76c.yaml

Inline suits a card with one or two objects of its own. A file suits pointing
at two entries out of fifty. Both may appear together, and the entries are
validated the same way either way.

Three kinds are available: ``theorem``, ``conjecture`` and ``question``. An id
can keep its name as the object behind it develops, so code that points at a
question keeps working when someone turns that question into a conjecture and
then proves it.


Pointing at Lean
----------------

An entry may name where its statement is formalized:

.. code:: yaml

      - id: Examples.CoinFlip.Binomial
        kind: theorem
        declaration: MagnetExamples.CoinFlip.count_headCount_eq
        statement: >
          Exactly choose(n, k) of the 2^n sequences of n flips show k heads.

``declaration`` is the fully-qualified name in whatever proof assistant states
it. All three shipped examples carry one, with the Lean beside the Python that
points at it:

.. code:: text

    magnet/examples/theory_links/coin_flip/
        card.yaml           the evaluation card, with its entries inline
        experiment.py       the annotated code, runnable as a node
        CoinFlip.lean       the statement it points at

The statements import Mathlib, so checking them needs a Lean project that has
one built. Rather than carry a lake project and a Mathlib pin here, borrow one:

.. code:: bash

    MAGNET_LEAN_PROJECT=~/code/aiq-dkps-formalization \
        magnet/examples/theory_links/check_lean.sh

    coin_flip/CoinFlip.lean                  ok (0 sorry)
    monte_carlo/Circle.lean                  ok (1 sorry)
    training_order/TrainingOrder.lean        ok (0 sorry)

A ``sorry`` is reported rather than treated as a failure. A statement can be
well-formed and unproved, and which of the two it is belongs in the output:
the quarter-disc area the Monte Carlo card samples is stated and unproved,
while the unit-disc area beside it follows from Mathlib's
``Complex.volume_closedBall``.

Reading proof status out of the kernel rather than from a script's output,
resolving a declaration against a pinned commit, and accounting for a
theorem's individual hypotheses are built on top of this field. None of that
is here yet.


Connecting it to a card
-----------------------

Paths in ``sources`` and ``indexes`` are relative to the card, so an example
that keeps its parts together names its siblings.

Evaluating the card reads the source, reads the indexes, checks that every
reference resolves, and writes ``theory.json`` beside ``verdict.json``:

.. code:: json

    {
      "links": [
        {
          "relation": "motivates",
          "ref": "Examples.TrainingOrder.Why",
          "file": ".../training_order/experiment.py",
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

Three examples ship with MAGNET, one per relation. Each is a directory holding
its card, its code, its index and its Lean statement, so an example can be read
in one place and copied out in one piece:

=====================  =================  ==============================
directory              relation           what it shows
=====================  =================  ==============================
``coin_flip``          ``tests``          enumeration matches the
                                          binomial law exactly
``monte_carlo``        ``approximates``   sampling estimates pi/4, with
                                          error the closed form lacks
``training_order``     ``motivates``      reordering the same data
                                          changes the learned solution
=====================  =================  ==============================

Each runs offline in a fraction of a second:

.. code:: bash

    python -m magnet.evaluation \
        magnet/examples/theory_links/training_order/card.yaml

and leaves ``theory.json`` in the run directory beside the verdict.
