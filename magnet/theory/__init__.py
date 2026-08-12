"""
Connecting formalized theory to empirical evaluation.

A card measures a claim that is often the finite-sample shadow of a proved
statement. The statement holds given a list of hypotheses; the experiment
rarely satisfies them literally. Each departure is an **edge**, declared at the
code that departs:

.. code:: python

    from magnet.theory import approximates, assumes, grounds, substitutes

    @substitutes('Paper.main::yNN', kind='estimator-swap',
                 informal='ordinary least squares, not the nearest-neighbour '
                          'estimator the theorem covers')
    def predict(self, train, test): ...

    assumes('Paper.main::hlipschitz', severity='high',
            informal='score smoothness in embedding distance is never tested')

    @grounds('Paper.TheoryPractice.EmpiricalCrossBudgetMAEClaim')
    def claim(results):
        assert results['candidate_mae'] <= results['baseline_mae']

Reported as::

    standing on 6 assumptions: 1 discharged, 4 relaxed (max severity high),
    1 unaccounted

The verb carries the relation -- ``satisfies``, ``approximates``,
``substitutes``, ``assumes``, ``ignores``, ``violates`` -- so each annotation
reads as a sentence at its site.

Annotations are read from source, not imports: :mod:`magnet.theory.static`
parses them, so a repository is auditable without installing its dependencies
or running it. Nothing executes an annotation. Teams can therefore vendor
:mod:`magnet.theory.shim`, in which every predicate is a no-op, rather than
depend on MAGNET.
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
)
from magnet.theory.model import (
    KERNEL_AXIOMS,
    CodeSite,
    Formalization,
    Freshness,
    Hypothesis,
    HypothesisRef,
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
    grounds,
    ignores,
    satisfies,
    substitutes,
    violates,
)

__all__ = [
    # predicates
    'satisfies',
    'approximates',
    'substitutes',
    'assumes',
    'ignores',
    'violates',
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
    # loading
    'hygiene',
    'load',
    'load_index',
]
