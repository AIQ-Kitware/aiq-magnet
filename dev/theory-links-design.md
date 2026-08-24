# Static theory links: design boundary

The feature has one job: make the team's argument about how empirical code
relates to theory explicit, structured, and reportable without importing the
team's code or recording runtime theory state.

## Two static layers

Statement links connect empirical claims or procedures to theoretical objects:

    practice tests theory
    practice approximates theory
    practice motivates theory

Premise links connect implementation sites to named premises of those objects:

    satisfies
    approximates
    substitutes
    assumes
    ignores
    violates
    checks

The shared `approximates` verb is intentional. At statement scope it says the
empirical procedure is a finite or proxy version of a theoretical object. At
premise scope it says the implementation represents the same required object
finitely or numerically.

A premise reference is `EntryId::binder`. The entry resolves through a theory
index to a formal declaration; the binder resolves against the premise list
exported for that entry. This gives static reporting a precise join point and
creates a future formal-verification obligation without defining Python runtime
semantics now.

## Lean owns formal statements

Formalized objects should normally enter MAGNET through a versioned index:

    Lean declaration -> generated theory index -> MAGNET entry ID

The index carries formalization provenance, declaration identity, and named
premises. Human-readable `statement` text is report metadata when a declaration
exists. Cards do not redefine the formal proposition.

Inline entries remain useful for local questions, conjectures, and claim shapes
that do not yet have a formal home.

Entry IDs identify stable theoretical objects. A question that later receives a
conjecture or theorem keeps its historical identity and gets a new related
object; an old ID does not change its proposition underneath existing empirical
annotations.

## Empirical source owns premise accounting

The card names `empirical_sources` to scan. Statement links may also live in the
card when the relationship belongs to the card-level claim. Premise links stay
next to the implementation because they describe what code does about a named
formal assumption.

Annotations are runtime no-ops. Static extraction accepts a narrow literal
syntax and fails on malformed recognized annotations. The dependency-free
`annotations.py` file can be copied verbatim into a team package as
`magnet_theory.py`.

## Coverage is derived

A statement entry declares its premise list. Source annotations account for
some subset of those premises. MAGNET computes the remainder as unaccounted and
writes that result to the versioned `theory.json` artifact.

No author maintains a coverage flag. No severity, review, or freshness workflow
is required to answer the basic reporting question:

    Which premises does this empirical argument satisfy, approximate,
    substitute, assume, ignore, violate, check, or leave unaccounted?

## The worked example proves the richer shape

The Monte Carlo example is the premise-aware demonstration. Its Lean file
contains an exact geometric theorem and a stronger sampling theorem with named
`hindicator`, `hmeas`, `hiid`, and `huniform` premises. The Python source
`satisfies` the indicator contract, `assumes` the measurability bridge, and
`substitutes` a deterministic LCG for uniform draws. It intentionally says
nothing about `hiid`, so the derived report has one unaccounted premise.

That omission is part of the example rather than a test fixture: it shows the
main value of static premise accounting without requiring runtime theory state.

The Fibonacci-performance example demonstrates a different case: an empirical
phenomenon can motivate a theoretical question before the mechanism is known.
The example also includes a known Lean explanation in an abstract operation-
count model. The wall-clock benchmark ``approximates`` that theorem rather than
``tests`` it, because the formal statement explains a structural work gap but
does not model Python execution time. This is the intended pattern for teams
that have a strong empirical observation but little bespoke theory: name the
question honestly, then connect whatever generic or mechanistic theory is
actually justified without manufacturing a theorem that merely restates the
measurement.

## Deliberately absent

This layer does not record runtime theory state. In particular it does not:

- bind Lean parameters to concrete Python values;
- attach theory metadata to runtime Python objects;
- record whether a particular runtime premise check passed;
- model severity or human review status;
- model proof status or axiom dependencies;
- maintain freshness as a status axis;
- require a separate generated ledger.

Those capabilities can use the same stable statement and premise identities if
they become necessary later.
