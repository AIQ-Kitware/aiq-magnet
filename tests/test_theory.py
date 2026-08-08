"""
Tests for the theory model, predicates, and coverage accounting.

The behaviour worth protecting: a hypothesis nobody handled shows up as
unaccounted, the three activation modes register identically, a checker that
fails flips the relation, and a statement whose binders were never exported
never reads as covered.
"""
import pytest

from magnet.theory import (
    Check,
    Formalization,
    Freshness,
    Hypothesis,
    ProofStatus,
    Relation,
    ReviewStatus,
    Severity,
    TheoreticalBasis,
    TheoryRegistry,
    Theorem,
    approximates,
    assumes,
    checks,
    grounds,
    hygiene,
    satisfies,
    substitutes,
    violates,
)


@pytest.fixture
def registry():
    return TheoryRegistry()


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
    return TheoreticalBasis.collect(
        registry=None,
        extra_groundings=grounded,
        extra_edges=edges,
        formalizations=[formalization],
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


def test_predicate_registers_when_called_not_when_activated(registry):
    # The ledger must not depend on a code path being reached.
    assumes('Paper.Section.main::hiid', registry=registry)
    assert len(registry) == 1


def test_bare_form_has_no_site(registry):
    edge = assumes('Paper.Section.main::hiid', registry=registry)
    assert edge.site is None
    assert edge.relation is Relation.ASSUMES


def test_decorator_form_records_site_and_witnesses_arguments(registry):
    @approximates('Paper.Section.main::hlim', registry=registry)
    def build_pool(num_example_runs=64, label='x'):
        return num_example_runs

    edge = build_pool.__magnet_edges__[0]
    assert edge.site.qualname.endswith('build_pool')
    assert not edge.observations

    build_pool()
    assert len(edge.observations) == 1
    assert edge.witnessed == {'num_example_runs': 64, 'label': 'x'}

    build_pool(32)
    assert edge.witnessed['num_example_runs'] == 32


def test_decorator_and_context_manager_agree(registry):
    # Decorating a function is defined as running its body in the context
    # manager, so the two must produce the same record.
    @approximates('Paper.Section.main::hlim', registry=registry)
    def decorated(n=8):
        return n

    decorated()

    with approximates('Paper.Section.main::hlim', registry=registry) as edge:
        edge.witness(n=8)

    decorated_edge = decorated.__magnet_edges__[0]
    assert len(decorated_edge.observations) == len(edge.observations) == 1
    assert decorated_edge.witnessed == edge.witnessed == {'n': 8}
    assert decorated_edge.relation is edge.relation


def test_context_manager_locates_the_with_statement(registry):
    with assumes('Paper.Section.main::hiid', registry=registry) as edge:
        pass
    assert edge.site is not None
    assert edge.site.file.endswith('test_theory.py')


def test_context_manager_does_not_swallow_exceptions(registry):
    with pytest.raises(ValueError):
        with assumes('Paper.Section.main::hiid', registry=registry):
            raise ValueError('boom')


def test_observe_false_skips_the_wrapper(registry):
    @assumes('Paper.Section.main::hiid', observe=False, registry=registry)
    def hot_path(n=1):
        return n

    hot_path()
    assert hot_path.__magnet_edges__[0].observations == []


def test_decorating_a_class_annotates_without_wrapping(registry):
    @assumes('Paper.Section.main::hiid', registry=registry)
    class Predictor:
        pass

    assert Predictor.__magnet_edges__[0].relation is Relation.ASSUMES
    assert isinstance(Predictor, type)


def test_severity_defaults_per_predicate(registry):
    assert satisfies('A::h', registry=registry).severity is Severity.NONE
    assert approximates('A::h', registry=registry).severity is Severity.MEDIUM
    assert substitutes('A::h', registry=registry).severity is Severity.HIGH
    assert violates('A::h', registry=registry).severity is Severity.HIGH


def test_severity_can_be_overridden_positionally_or_by_keyword(registry):
    assert approximates('A::h', 'high', registry=registry).severity is Severity.HIGH
    assert approximates('A::h', severity='low', registry=registry).severity is Severity.LOW


# -------------------------------------------------------------------- checks


def test_passing_check_resolves_to_satisfies(registry):
    @checks('Paper.Section.main::hvar', registry=registry)
    def rank_ok(rank, budget):
        return Check(ok=rank <= budget, value=rank)

    rank_ok(7, 8)
    edge = rank_ok.__magnet_edges__[0]
    assert edge.check_outcome == 'passed'
    assert edge.resolved_relation is Relation.SATISFIES
    assert not edge.is_gap


def test_failing_check_resolves_to_violates_at_high_severity(registry):
    # A checker that runs and comes back false is a measured contradiction,
    # not a missing annotation.
    @checks('Paper.Section.main::hvar', registry=registry)
    def rank_ok(rank, budget):
        return Check(ok=rank <= budget, value=rank)

    rank_ok(9, 8)
    edge = rank_ok.__magnet_edges__[0]
    assert edge.check_outcome == 'failed'
    assert edge.resolved_relation is Relation.VIOLATES
    assert edge.resolved_severity is Severity.HIGH


def test_check_that_never_runs_leaves_the_hypothesis_assumed(registry):
    @checks('Paper.Section.main::hvar', registry=registry)
    def never_called():
        return Check(ok=True)

    edge = never_called.__magnet_edges__[0]
    assert edge.check_outcome == 'not-run'
    assert edge.resolved_relation is Relation.ASSUMES


def test_check_returns_the_underlying_value(registry):
    @checks('Paper.Section.main::hvar', registry=registry)
    def compute():
        return Check(ok=True, value=42)

    assert compute().value == 42


# ------------------------------------------------------------------- coverage


def test_unaccounted_hypotheses_are_reported(formalization, registry):
    basis = _basis(
        formalization,
        grounds('Paper.Section.main', registry=registry),
        approximates('Paper.Section.main::hlim', 'high', registry=registry),
        satisfies('Paper.Section.main::hvar', registry=registry),
    )
    coverage = basis.coverage()

    assert not coverage.is_complete
    assert sorted(h.name for h in coverage.unaccounted) == ['hiid', 'psi']
    assert coverage.max_severity is Severity.HIGH
    assert 'unaccounted' in coverage.summary()


def test_complete_coverage(formalization, registry):
    annotations = [grounds('Paper.Section.main', registry=registry)]
    annotations += [
        satisfies(f'Paper.Section.main::{name}', registry=registry)
        for name in ('hiid', 'hvar', 'hlim', 'psi')
    ]
    coverage = _basis(formalization, *annotations).coverage()
    assert coverage.is_complete
    assert coverage.max_severity is Severity.NONE
    assert coverage.summary() == 'standing on 4 assumptions: 4 discharged'


def test_unenumerated_statement_never_reads_as_complete(registry):
    # A statement whose binders nobody exported is not a covered one; letting
    # it pass would make a missing exporter look like a clean bill of health.
    formalization = Formalization(name='F', theorems=[Theorem('Paper.capstone')])
    basis = _basis(formalization, grounds('Paper.capstone', registry=registry))
    coverage = basis.coverage()
    assert not coverage.is_complete
    assert 'hypotheses not enumerated' in coverage.summary()


def test_orphaned_edges_are_surfaced(formalization, registry):
    basis = _basis(
        formalization,
        grounds('Paper.Section.main', registry=registry),
        assumes('Paper.Section.main::h_renamed_upstream', registry=registry),
    )
    coverage = basis.coverage()
    assert [e.ref.hypothesis for e in coverage.per_theorem[0].orphaned] == [
        'h_renamed_upstream'
    ]
    assert not coverage.is_complete


def test_dangling_edges_are_surfaced(formalization, registry):
    # An edge naming a statement nothing is grounded on would otherwise vanish
    # from the report with no explanation.
    basis = _basis(
        formalization,
        grounds('Paper.Section.main', registry=registry),
        assumes('Some.Other.Statement::hx', registry=registry),
    )
    coverage = basis.coverage()
    assert [e.ref.key for e in coverage.dangling] == ['Some.Other.Statement::hx']
    assert not coverage.is_complete
    assert 'dangling' in coverage.summary()


def test_failed_checks_are_collected(formalization, registry):
    @checks('Paper.Section.main::hvar', registry=registry)
    def failing():
        return Check(ok=False, detail='rank exceeded budget')

    failing()
    basis = _basis(
        formalization,
        grounds('Paper.Section.main', registry=registry),
        failing.__magnet_edges__[0],
    )
    assert len(basis.coverage().failed_checks) == 1


def test_unproved_grounding_is_reported(registry):
    formalization = Formalization(
        name='F', theorems=[Theorem('Paper.shaky', axioms=['propext', 'sorryAx'])]
    )
    basis = _basis(formalization, grounds('Paper.shaky', registry=registry))
    assert [t.declaration for t in basis.coverage().unproved] == ['Paper.shaky']


def test_resolve_marks_broken_references(formalization, registry):
    basis = _basis(
        formalization,
        grounds('Paper.Section.main', registry=registry),
        assumes('Paper.Section.main::hiid', registry=registry),
        assumes('Paper.Section.main::h_gone', registry=registry),
    ).resolve()
    freshness = {e.ref.hypothesis: e.freshness for e in basis.edges}
    assert freshness['hiid'] is Freshness.CURRENT
    assert freshness['h_gone'] is Freshness.BROKEN


def test_registry_collection_finds_edges_the_card_does_not_contain(registry):
    # The motivating case: the relaxation lives in a predictor one repo away.
    @approximates('Paper.Section.main::hlim', registry=registry)
    def somewhere_else():
        return 1

    grounding = grounds('Paper.Section.main', registry=registry)
    basis = TheoreticalBasis.collect(
        registry=registry, extra_groundings=[grounding]
    )
    assert [e.ref.hypothesis for e in basis.edges] == ['hlim']


def test_registry_does_not_sweep_in_ungrounded_edges(registry):
    assumes('Other.thing::hx', registry=registry)
    grounding = grounds('Paper.Section.main', registry=registry)
    basis = TheoreticalBasis.collect(registry=registry, extra_groundings=[grounding])
    assert basis.edges == ()


def test_registry_separates_observed_from_declared(registry):
    @assumes('A::h1', registry=registry)
    def ran():
        return 1

    @assumes('A::h2', registry=registry)
    def never_ran():
        return 1

    ran()
    assert len(registry.observed()) == 1
    assert len(registry.declared_but_unobserved()) == 1


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
