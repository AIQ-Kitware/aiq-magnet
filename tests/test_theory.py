"""
Tests for the theory model, predicates, and coverage accounting.

The behaviour worth protecting: a hypothesis nobody handled shows up as
unaccounted, and a statement whose binders were never exported never reads as
covered.
"""
import pytest

from magnet.theory import (
    Formalization,
    Freshness,
    Hypothesis,
    ProofStatus,
    Relation,
    ReviewStatus,
    Severity,
    Theorem,
    TheoreticalBasis,
    approximates,
    assumes,
    grounds,
    hygiene,
    satisfies,
    substitutes,
    violates,
)


@pytest.fixture
def formalization():
    return Formalization(
        name='Example',
        commit='deadbeef',
        theorems=[
            Theorem(
                declaration='Paper.Section.main',
                informal='The estimator converges.',
                hypotheses=[
                    Hypothesis('hiid', 'draws are iid'),
                    Hypothesis('hvar', 'finite variance'),
                    Hypothesis('hlim', 'n tends to infinity'),
                    Hypothesis('psi', 'the true embedding', structural=True),
                ],
                axioms=['propext', 'Classical.choice', 'Quot.sound'],
            )
        ],
    )


def _basis(formalization, *annotations):
    grounded = [a for a in annotations if a.__class__.__name__ == 'Grounding']
    edges = [a for a in annotations if a.__class__.__name__ != 'Grounding']
    return TheoreticalBasis(
        groundings=tuple(grounded),
        edges=tuple(edges),
        formalizations=(formalization,),
    )


# ------------------------------------------------------------------- the model


def test_hypothesis_keys_are_qualified(formalization):
    assert formalization['Paper.Section.main']['hiid'].key == 'Paper.Section.main::hiid'


def test_unknown_binder_is_an_error(formalization):
    with pytest.raises(KeyError, match='has no hypothesis'):
        formalization['Paper.Section.main']['h_typo']


def test_unenumerated_statement_invents_binders_on_demand():
    statement = Theorem('Paper.capstone')
    assert not statement.hypotheses_enumerated
    assert statement['anything'].key == 'Paper.capstone::anything'


@pytest.mark.parametrize(
    'axioms, status',
    [
        (['propext', 'Classical.choice', 'Quot.sound'], ProofStatus.PROVED),
        (['propext'], ProofStatus.PROVED),
        (['propext', 'sorryAx'], ProofStatus.SORRY),
        (None, ProofStatus.UNKNOWN),
    ],
)
def test_proof_status_tracks_the_axiom_set(axioms, status):
    assert Theorem('A', axioms=axioms).proof is status


def test_status_axes_are_independent():
    # A statement can be proved, draft, and stale at once. Merging these into
    # one indicator is the mistake this triple exists to prevent.
    statement = Theorem('A', axioms=['propext'], review='draft')
    assert statement.proof is ProofStatus.PROVED
    assert statement.review is ReviewStatus.DRAFT


def test_lookup_suggests_near_misses(formalization):
    with pytest.raises(KeyError, match='did you mean'):
        formalization['Paper.Section.mian']


# --------------------------------------------------------------- the predicates


def test_bare_form_has_no_site():
    edge = assumes('Paper.Section.main::hiid')
    assert edge.site is None
    assert edge.relation is Relation.ASSUMES


def test_decorator_records_the_site_and_returns_the_function():
    @approximates('Paper.Section.main::hlim')
    def build_pool(num_example_runs=64):
        return num_example_runs

    edge = build_pool.__magnet_edges__[0]
    assert edge.site.qualname.endswith('build_pool')
    # Annotated code runs identically whether or not anyone is auditing it.
    assert build_pool(32) == 32


def test_decorating_a_class_annotates_without_wrapping():
    @assumes('Paper.Section.main::hiid')
    class Predictor:
        pass

    assert Predictor.__magnet_edges__[0].relation is Relation.ASSUMES
    assert isinstance(Predictor, type)


def test_severity_defaults_per_predicate():
    assert satisfies('A::h').severity is Severity.NONE
    assert approximates('A::h').severity is Severity.MEDIUM
    assert substitutes('A::h').severity is Severity.HIGH
    assert violates('A::h').severity is Severity.HIGH


def test_severity_can_be_overridden_positionally_or_by_keyword():
    assert approximates('A::h', 'high').severity is Severity.HIGH
    assert approximates('A::h', severity='low').severity is Severity.LOW


# ------------------------------------------------------------------- coverage


def test_unaccounted_hypotheses_are_reported(formalization):
    basis = _basis(
        formalization,
        grounds('Paper.Section.main'),
        approximates('Paper.Section.main::hlim', 'high'),
        satisfies('Paper.Section.main::hvar'),
    )
    coverage = basis.coverage()

    assert not coverage.is_complete
    assert sorted(h.name for h in coverage.unaccounted) == ['hiid', 'psi']
    assert coverage.max_severity is Severity.HIGH
    assert 'unaccounted' in coverage.summary()


def test_complete_coverage(formalization):
    annotations = [grounds('Paper.Section.main')]
    annotations += [
        satisfies(f'Paper.Section.main::{name}')
        for name in ('hiid', 'hvar', 'hlim', 'psi')
    ]
    coverage = _basis(formalization, *annotations).coverage()
    assert coverage.is_complete
    assert coverage.max_severity is Severity.NONE
    assert coverage.summary() == 'standing on 4 assumptions: 4 discharged'


def test_unenumerated_statement_never_reads_as_complete():
    # A statement whose binders nobody exported is not a covered one; letting
    # it pass would make a missing exporter look like a clean bill of health.
    formalization = Formalization(name='F', theorems=[Theorem('Paper.capstone')])
    basis = _basis(formalization, grounds('Paper.capstone'))
    coverage = basis.coverage()
    assert not coverage.is_complete
    assert 'hypotheses not enumerated' in coverage.summary()


def test_orphaned_edges_are_surfaced(formalization):
    basis = _basis(
        formalization,
        grounds('Paper.Section.main'),
        assumes('Paper.Section.main::h_renamed_upstream'),
    )
    coverage = basis.coverage()
    assert [e.ref.hypothesis for e in coverage.per_theorem[0].orphaned] == [
        'h_renamed_upstream'
    ]
    assert not coverage.is_complete


def test_dangling_edges_are_surfaced(formalization):
    # An edge naming a statement nothing is grounded on would otherwise vanish
    # from the report with no explanation.
    basis = _basis(
        formalization,
        grounds('Paper.Section.main'),
        assumes('Some.Other.Statement::hx'),
    )
    coverage = basis.coverage()
    assert [e.ref.key for e in coverage.dangling] == ['Some.Other.Statement::hx']
    assert not coverage.is_complete
    assert 'dangling' in coverage.summary()


def test_unproved_grounding_is_reported():
    formalization = Formalization(
        name='F', theorems=[Theorem('Paper.shaky', axioms=['propext', 'sorryAx'])]
    )
    basis = _basis(formalization, grounds('Paper.shaky'))
    assert [t.declaration for t in basis.coverage().unproved] == ['Paper.shaky']


def test_resolve_marks_broken_references(formalization):
    basis = _basis(
        formalization,
        grounds('Paper.Section.main'),
        assumes('Paper.Section.main::hiid'),
        assumes('Paper.Section.main::h_gone'),
    ).resolve()
    freshness = {e.ref.hypothesis: e.freshness for e in basis.edges}
    assert freshness['hiid'] is Freshness.CURRENT
    assert freshness['h_gone'] is Freshness.BROKEN


# --------------------------------------------------------------------- hygiene


def test_hygiene_library_loads():
    library = hygiene()
    assert len(library) >= 5
    statement = library['Hygiene.Conformal.marginal_coverage_of_exchangeable']
    assert {h.name for h in statement.hypotheses} == {
        'hexch',
        'hsplit',
        'hscore_fixed',
        'hquantile',
    }


def test_hygiene_statements_are_honestly_unproved():
    # They are stated, not formalized. Claiming otherwise would be the exact
    # dishonesty this package exists to prevent.
    for statement in hygiene():
        assert statement.proof is ProofStatus.UNKNOWN
        assert statement.review is ReviewStatus.DRAFT
