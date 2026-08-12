"""
Connecting formalized theory to empirical evaluation.

MAGNET's premise is that it "connects theoretical claims about AI model
generalization to empirical validation through standardized evaluation cards".
This package is the connection.

A card measures a claim. Sometimes that claim is the finite-sample shadow of a
statement that has been proved -- in Lean, from a paper, about an idealized
version of the same setup. The statement holds *provided a list of explicit
hypotheses*; the experiment almost never satisfies them literally. Each way it
departs is an **edge**, declared at the code that departs:

.. code:: python

    from magnet.theory import approximates, assumes, grounds, substitutes

    @substitutes('Paper.main::yNN', kind='estimator-swap',
                 informal='ordinary least squares, not the nearest-neighbour '
                          'estimator the theorem covers')
    def predict(self, train, test): ...

    @approximates('Paper.main::hcover',
                  informal='fixed reference pool; the theorem needs density')
    def __init__(self, num_example_runs=64): ...

    assumes('Paper.main::hlipschitz', severity='high',
            informal='score smoothness in embedding distance is never tested')

    @grounds('Paper.TheoryPractice.EmpiricalCrossBudgetMAEClaim')
    def claim(results):
        assert results['candidate_mae'] <= results['baseline_mae']

Which yields, next to the verdict:

.. code:: text

    RESULT:  VERIFIED
    BASIS:   standing on 6 assumptions: 1 discharged, 4 relaxed (max severity
             high), 1 unaccounted

Three things about the design are worth knowing before using it.

**The verb carries the relation.** ``satisfies``, ``approximates``,
``substitutes``, ``assumes``, ``ignores``, ``violates`` -- and ``checks``, which
resolves at run time. Each reads as a true sentence at the site it annotates.

**Every predicate activates three ways** -- bare call, decorator, context
manager -- with identical registration, so the ledger never depends on a code
path being reached. See :mod:`magnet.theory.predicates`.

**Annotations are read from source, not from imports.**
:mod:`magnet.theory.static` extracts the ledger by parsing, so a repository can
be audited without installing its dependencies or executing its code. The
runtime registry is for evidence -- what actually ran, with what values.

Teams annotating their own code do not need MAGNET installed:
:mod:`magnet.theory.shim` is a dependency-free file to vendor, in which every
predicate is a no-op.
"""
from magnet.theory.basis import (
    AssumptionCoverage,
    CoverageReport,
    TheoreticalBasis,
)
from magnet.theory.index import (
    hygiene,
    load,
    load_index,
    load_manifest,
    save_index,
)
from magnet.theory.model import (
    KERNEL_AXIOMS,
    Check,
    CodeSite,
    Formalization,
    Freshness,
    Hypothesis,
    HypothesisRef,
    Observation,
    ProofStatus,
    Relation,
    ReviewStatus,
    Severity,
    Theorem,
    parse_ref,
)
from magnet.theory.predicates import (
    PREDICATE_NAMES,
    Edge,
    Grounding,
    approximates,
    assumes,
    checks,
    grounds,
    ignores,
    satisfies,
    substitutes,
    violates,
)
from magnet.theory.registry import REGISTRY, TheoryRegistry

__all__ = [
    # predicates
    'satisfies',
    'approximates',
    'substitutes',
    'assumes',
    'ignores',
    'violates',
    'checks',
    'grounds',
    # annotation objects
    'Edge',
    'Grounding',
    'PREDICATE_NAMES',
    # model
    'Formalization',
    'Theorem',
    'Hypothesis',
    'HypothesisRef',
    'CodeSite',
    'Observation',
    'Check',
    'Relation',
    'Severity',
    'ProofStatus',
    'ReviewStatus',
    'Freshness',
    'KERNEL_AXIOMS',
    'parse_ref',
    # assembly
    'TheoreticalBasis',
    'CoverageReport',
    'AssumptionCoverage',
    # registry
    'REGISTRY',
    'TheoryRegistry',
    # loading
    'load',
    'hygiene',
    'load_index',
    'load_manifest',
    'save_index',
]
