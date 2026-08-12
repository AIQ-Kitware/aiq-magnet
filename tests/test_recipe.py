"""
Tests for Python-defined evaluation recipes.

Two properties matter most. A recipe and the YAML card it compiles to must
agree, or the export is worse than not having one. And the verdict mapping must
match the YAML runner's, because FALSIFIED (tested and false) and INCONCLUSIVE
(never actually tested) are not interchangeable.
"""
import pytest

from magnet.card import Verdict
from magnet.recipe import Sweep, _reduce, claim, recipe, symbol


def test_constants_sweeps_and_computed_symbols():
    @recipe(title='Basic', version='1.0')
    class Basic:
        offset = 2
        base = Sweep([1, 2, 3])

        @symbol(type='int')
        def total(base, offset):
            return base + offset

        @claim
        def positive(total):
            assert total > 0

    assert Basic.symbols['offset'].value == 2
    assert Basic.symbols['base'].sweep == (1, 2, 3)
    assert Basic.symbols['total'].dependencies == ('base', 'offset')
    assert Basic.claims['positive'].dependencies == ('total',)


def test_helpers_are_not_symbols():
    @recipe(title='Helpers')
    class WithHelper:
        def helper(x):
            return x

        @symbol
        def value():
            return 1

        @claim
        def ok(value):
            assert value == 1

    assert set(WithHelper.symbols) == {'value'}


def test_undefined_dependency_fails_at_definition_time():
    with pytest.raises(ValueError, match='undefined names'):

        @recipe(title='Broken')
        class Broken:
            @symbol
            def a(does_not_exist):
                return does_not_exist

            @claim
            def ok(a):
                assert a


def test_sweep_produces_one_outcome_per_point():
    @recipe(title='Sweeps')
    class Sweeps:
        a = Sweep([1, 2])
        b = Sweep([10, 20, 30])

        @claim
        def ok(a, b):
            assert a < b

    card = Sweeps.evaluate()
    assert len(card.empirical.outcomes) == 6
    assert card.verdict is Verdict.VERIFIED


def test_assertion_failure_is_falsified_and_carries_the_message():
    @recipe(title='Falsify')
    class Falsify:
        @claim
        def ok():
            assert False, 'the number was wrong'

    card = Falsify.evaluate()
    assert card.verdict is Verdict.FALSIFIED
    assert 'the number was wrong' in card.empirical.outcomes[0].message


def test_other_exceptions_are_inconclusive_not_falsified():
    # A claim that blew up was never tested. Reporting it as FALSIFIED would
    # assert something about the world that the run did not establish.
    @recipe(title='Blows up')
    class BlowsUp:
        @claim
        def ok():
            raise RuntimeError('the endpoint was down')

    card = BlowsUp.evaluate()
    assert card.verdict is Verdict.INCONCLUSIVE
    assert 'endpoint was down' in card.empirical.outcomes[0].message


def test_symbol_resolution_failure_is_inconclusive():
    @recipe(title='Bad symbol')
    class BadSymbol:
        @symbol
        def broken():
            raise ValueError('no data on disk')

        @claim
        def ok(broken):
            assert broken

    card = BadSymbol.evaluate()
    assert card.verdict is Verdict.INCONCLUSIVE
    assert 'no data on disk' in card.empirical.outcomes[0].message


def test_symbols_outside_the_sweep_are_computed_once():
    # Not a micro-optimisation: the leading symbol of a real card loads a
    # benchmark suite off disk, and the sweep runs dozens of points.
    calls = {'expensive': 0, 'cheap': 0}

    @recipe(title='Sharing')
    class Sharing:
        seed = Sweep([1, 2, 3, 4])

        @symbol
        def expensive():
            calls['expensive'] += 1
            return 100

        @symbol
        def cheap(expensive, seed):
            calls['cheap'] += 1
            return expensive + seed

        @claim
        def ok(cheap):
            assert cheap > 0

    Sharing.evaluate()
    assert calls['expensive'] == 1
    assert calls['cheap'] == 4


def test_fraction_aggregation():
    @recipe(
        title='Fraction',
        claim_aggregation_strategy={'type': 'fraction', 'parameters': {'threshold': 0.5}},
    )
    class Fraction:
        value = Sweep([1, 2, 3, 4])

        @claim
        def small(value):
            assert value <= 2

    card = Fraction.evaluate()
    assert card.empirical.counts == {'VERIFIED': 2, 'FALSIFIED': 2}
    assert card.verdict is Verdict.VERIFIED  # 2/4 >= 0.5


@pytest.mark.parametrize(
    'strategy',
    [
        {'type': 'all'},
        {'type': 'any'},
        {'type': 'fraction', 'parameters': {'threshold': 0.8}},
        {'type': 'fraction', 'parameters': {'threshold': 0.2}},
    ],
)
@pytest.mark.parametrize(
    'verdicts',
    [
        ['VERIFIED'],
        ['FALSIFIED'],
        ['INCONCLUSIVE'],
        ['VERIFIED', 'FALSIFIED'],
        ['VERIFIED', 'INCONCLUSIVE'],
        ['VERIFIED', 'VERIFIED', 'VERIFIED', 'FALSIFIED'],
        ['FALSIFIED', 'INCONCLUSIVE'],
    ],
)
def test_reduce_agrees_with_the_yaml_runner(strategy, verdicts):
    # A recipe and its exported card disagreeing about the verdict would be
    # worse than either of them being wrong.
    from magnet.evaluation import _reduce_results

    ours = _reduce([Verdict(v) for v in verdicts], strategy)
    theirs = _reduce_results(list(verdicts), strategy)
    assert str(ours) == theirs


def test_export_compiles_to_a_runnable_card():
    from magnet.examples.commutative_arithmetic import CommutativeAddition

    spec = CommutativeAddition.to_schema_dict()
    assert spec['title'] == 'Arithmetic - Addition Commutative Property'
    assert spec['category'] == 'Mathematical Properties'
    assert spec['symbols']['int_range_even']['type'] == 'List[int]'
    assert (
        spec['symbols']['int_range_even']['metadata']['display_name']
        == 'Set of Even Numbers'
    )

    # Computed symbols call back into the recipe rather than duplicating source,
    # so the exported card runs the same code the recipe does.
    body = spec['symbols']['int_range_odd']['python']
    assert 'from magnet.examples.commutative_arithmetic import CommutativeAddition' in body
    assert "CommutativeAddition.compute('int_range_odd', int_range_even=int_range_even)" in body
    assert spec['symbols']['int_range_odd']['depends_on'] == ['int_range_even']


def test_export_carries_sweeps_and_aggregation():
    @recipe(
        title='Swept',
        claim_aggregation_strategy={'type': 'fraction', 'parameters': {'threshold': 0.8}},
    )
    class Swept:
        seed = Sweep([1, 2, 3, 4, 5])
        sample_size = 16

        @claim
        def positive(seed, sample_size):
            assert seed * sample_size > 0

    spec = Swept.to_schema_dict()
    assert spec['claim_aggregation_strategy']['parameters']['threshold'] == 0.8
    assert spec['symbols']['seed']['sweep'] == [1, 2, 3, 4, 5]
    assert spec['symbols']['sample_size']['value'] == 16


def test_exported_python_blocks_execute_like_the_recipe():
    from magnet.examples.commutative_arithmetic import CommutativeAddition

    spec = CommutativeAddition.to_schema_dict()
    context = {'int_range_even': [0, 2, 4]}
    exec(spec['symbols']['int_range_odd']['python'], context)
    assert context['int_range_odd'] == [1, 3, 5]


def test_export_validates_against_the_card_schema():
    from magnet.examples.commutative_arithmetic import CommutativeAddition

    # A recipe that compiles to something the schema rejects would not be an
    # export at all.
    CommutativeAddition.to_schema()


def test_the_yaml_runner_agrees_with_the_recipe(tmp_path):
    # The claim this whole layer rests on: a recipe and the card it compiles to
    # are the same evaluation. Checked against the real runner, not a model of
    # it -- the exported YAML is loaded and executed by
    # `magnet.evaluation.EvaluationCard`, which knows nothing about recipes.
    from magnet.evaluation import EvaluationCard
    from magnet.examples.commutative_arithmetic import CommutativeAddition

    card_path = tmp_path / 'commutative.yaml'
    CommutativeAddition.write_card(card_path)

    theirs = EvaluationCard(card_path, tmp_path / 'results').evaluate()
    ours = CommutativeAddition.evaluate()

    assert str(ours.verdict) == theirs == 'VERIFIED'


def test_a_card_with_no_basis_says_so():
    @recipe(title='Ungrounded')
    class Ungrounded:
        @claim
        def ok():
            assert True

    card = Ungrounded.evaluate()
    # The slot is present and empty rather than absent, so a reader can tell
    # "nothing recorded" from "nothing to record".
    assert card.theoretical is None
    assert 'BASIS:   not recorded' in card.render()
