Linking Empirical Code to Theory
================================

MAGNET can record a static argument connecting empirical code to theoretical
objects. The connection is read from source without importing the annotated
module and without recording runtime values.

The model has two layers:

.. code:: text

    empirical claim/code
        -- tests / approximates / motivates --> theoretical statement

    empirical implementation
        -- satisfies / approximates / substitutes / assumes /
           ignores / violates / checks ------> named statement premise

The first layer says what theoretical object an evaluation bears on. The second
accounts for the assumptions underneath that object. Together they can produce
a useful report before Python semantics or runtime values are formally modeled.


Statement relations
-------------------

Statement relations read as ``practice <relation> theory``:

===============  ============================================================
``tests``        practice directly evaluates the named claim or consequence
``approximates`` practice measures a finite, sampled, or proxy version of it
``motivates``    practice establishes a phenomenon theory is asked to explain
===============  ============================================================

``tests`` should target the closest theoretical proposition that directly
matches the empirical claim. A finite empirical proposition is often a better
``tests`` target than a stronger population theorem. The same evaluation may
``approximates`` that population theorem separately.

``motivates`` targets a question, conjecture, or other object that the
observation asks theory to explain. It does not claim logical support. The
Fibonacci-performance example uses this distinction deliberately: a wall-clock
benchmark motivates the question of why two equal-result programs have very
different runtimes, while a separate Lean theorem gives an abstract call-count
explanation. The benchmark only ``approximates`` that cost theorem because
wall-clock time is a proxy for abstract work, not the quantity Lean proves.

Use a statement relation as a decorator or context manager:

.. code:: python

    import magnet_theory as theory

    @theory.tests('Examples.Stability.FiniteClaim')
    def evaluate_stability(...):
        ...

    with theory.approximates(
            'Examples.Stability.PopulationClaim',
            note='finite samples stand in for the population quantity'):
        ...

The annotations are runtime no-ops.


Premise relations
-----------------

A formal theorem usually depends on named premises. Empirical code can state
how it treats each premise using an ``EntryId::binder`` reference:

===============  ============================================================
``satisfies``    the code is asserted to establish the premise
``approximates`` the same object is represented finitely or numerically
``substitutes``  a different object stands in for the one theory requires
``assumes``      the premise is relied on without being established or checked
``ignores``      the premise is deliberately left out of the empirical model
``violates``     the premise is known not to hold
``checks``       the code contains a runtime check for the premise
===============  ============================================================

The shipped Monte Carlo example uses these relations against named binders
of a Lean sampling theorem:

.. code:: python

    import magnet_theory as theory

    @theory.satisfies(
        'Examples.Circle.MonteCarloConsistency::hindicator',
        note='the predicate exactly matches the formal quarter-disc region')
    def _inside_quarter_disc(x, y):
        ...

    @theory.approximates(
        'Examples.Circle.MonteCarloConsistency',
        note='a finite seeded LCG run stands in for the asymptotic IID model')
    @theory.assumes(
        'Examples.Circle.MonteCarloConsistency::hmeas',
        note='the measurable seed-state bridge is not formalized here')
    def estimate_area_ratio(...):
        with theory.substitutes(
                'Examples.Circle.MonteCarloConsistency::huniform',
                note='deterministic LCG outputs stand in for uniform draws'):
            ...

The theorem also has a named ``hiid`` premise. The example deliberately leaves
that premise unannotated so the report demonstrates an unaccounted obligation.

These annotations are authored scientific claims. ``satisfies`` does not mean
MAGNET has proved Python establishes the premise. It records a precise proof
obligation that can later be checked more formally.

``checks`` is also static. MAGNET records that a check exists at the annotated
site; it does not record whether a particular execution passed that check.


Theory indexes and Lean
-----------------------

Formalized theory should normally come through a versioned theory index. The
index is the bridge between compact MAGNET entry IDs and exact declarations in
a pinned formalization:

.. code:: yaml

    schema_version: 1

    formalization:
      system: lean4
      repository: https://example.invalid/formalization.git
      revision: 0123456789abcdef

    entries:
      - id: Examples.Stability.Theorem
        kind: theorem
        declaration: Example.Stability.theorem
        source_path: Example/Stability.lean
        statement: >
          A short human-readable description of the theorem.
        premises:
          - id: hbounded
            type: Bounded xs
          - id: hiid
            type: IID samples
          - id: hunique
            type: Unique optimum

``declaration`` names the formal object. ``formalization`` identifies the proof
system and, when available, the repository and revision. ``source_path`` points
inside that formalization. ``premises`` contains named formal binders that
empirical source may reference.

A future Lean exporter can generate this structure directly from declarations.
The important interface is already present: premise annotations refer to the
named binder through ``EntryId::binder`` rather than a source line number.
Renaming or removing a binder then breaks resolution and forces the empirical
annotation to be reconsidered.

When a formal declaration exists, that declaration is authoritative. The
optional ``statement`` is explanatory text for reports; it is not a second
formal definition.

Four entry kinds are available: ``theorem``, ``conjecture``, ``question`` and
``definition``. Entry IDs should identify stable theoretical objects. A new
conjecture or theorem answering an earlier question should receive its own ID
rather than changing what an existing ID denotes.


Inline entries
--------------

A card may define a small local object inline when no formal index exists:

.. code:: yaml

    theory:
      empirical_sources:
        - experiment.py
      entries:
        - id: Examples.FibonacciPerformance.Why
          kind: question
          statement: >
            Why can two programs that compute the same result have very
            different runtime costs?

Inline entries are useful for questions and local empirical claim shapes. Shared
or formalized theory should normally live in an index so multiple cards can
refer to one definition.


Connecting a card
-----------------

A card declares which empirical source files MAGNET should scan and which
indexes resolve their references:

.. code:: yaml

    theory:
      links:
        - relation: tests
          ref: Examples.Stability.FiniteClaim
          note: the card claim directly evaluates this finite proposition
      empirical_sources:
        - predictor.py
        - evaluation/
      indexes:
        - ../../theory/stability.yaml

``links`` is for relationships owned by the card-level evaluation claim.
Premise relationships belong next to the empirical implementation and therefore
come from ``empirical_sources``.

Paths are relative to the card. Source locations written to ``theory.json``
remain relative rather than embedding a workstation path.

Theory is preflighted before evaluation begins. Missing index entries, missing
premises, malformed annotations, and unparsable declared source files fail
before empirical work runs.


Premise coverage
----------------

Coverage is computed from the theory index and annotations; authors do not
maintain a coverage status manually.

The Monte Carlo example exports four named premises from
``MagnetExamples.Circle.monteCarloEstimator_consistent``. Static source
annotations account for three, so ``theory.json`` records ``hiid`` as the gap:

.. code:: json

    {
      "schema_version": 1,
      "statement_links": [
        {
          "relation": "approximates",
          "ref": "Examples.Circle.MonteCarloConsistency",
          "file": "experiment.py",
          "qualname": "estimate_area_ratio"
        }
      ],
      "premise_links": [
        {
          "relation": "satisfies",
          "ref": "Examples.Circle.MonteCarloConsistency::hindicator"
        },
        {
          "relation": "assumes",
          "ref": "Examples.Circle.MonteCarloConsistency::hmeas"
        },
        {
          "relation": "substitutes",
          "ref": "Examples.Circle.MonteCarloConsistency::huniform"
        }
      ],
      "premise_coverage": [
        {
          "ref": "Examples.Circle.MonteCarloConsistency",
          "premise_count": 4,
          "accounted_count": 3,
          "complete": false,
          "unaccounted": ["hiid"]
        }
      ]
    }

The full artifact also carries the referenced entries and premise descriptions,
so reporting code can display what each binder means and which source site
accounts for it.

This report does not claim formal verification of the empirical implementation.
It makes the team's argument explicit and identifies the premise-level proof
obligations that remain.


Annotating without depending on MAGNET
--------------------------------------

The annotation vocabulary is published separately as ``aiq-magnet-theory``. It has
no runtime dependencies and does not import MAGNET, so an application can add
just that package:

.. code:: bash

    pip install aiq-magnet-theory

Annotated code imports the standalone package directly:

.. code:: python

    import magnet_theory as theory

    @theory.tests('Examples.Stability.Theorem')
    @theory.assumes('Examples.Stability.Theorem::hiid')
    def experiment():
        ...

The full ``aiq-magnet`` distribution depends on the same version of
``aiq-magnet-theory`` and reexports the decorators through ``magnet.theory`` for
compatibility. The static extractor recognizes either import spelling.


What counts as an annotation
----------------------------

A relation is read when it appears as a decorator or ``with`` item on a
recognized theory namespace and uses literal strings for the reference and
optional ``note=`` value.

Once a recognized relation is used as an annotation, malformed arguments are an
error. For example, ``@theory.tests(REF)`` does not disappear from the report;
it fails because the reference is not a literal. Bare calls in ordinary
function bodies are not annotations.

Statement-only verbs reject ``::binder`` references. Premise-only verbs require
them. ``approximates`` is valid at either layer.


Current boundary
----------------

The static model deliberately leaves out runtime theory recording. MAGNET does
not bind Lean parameters to Python values, attach annotations to runtime
objects, or record per-run outcomes of premise checks.

The current artifact also leaves out severity scales, human review workflow,
proof-status dashboards, axiom accounting, and freshness status fields. Exact
reference resolution against versioned indexes provides the structural check
needed by this layer; additional verification can build on the same statement
and premise identities later.
