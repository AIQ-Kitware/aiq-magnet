"""
A new-evaluator cell identifies one available kwdagger result row, plus
non-measured recipe symbols consumed by the claim environment.
"""

import pytest

from magnet.evaluation import Symbols
from magnet.evaluation_new import (
    ClaimResultNamespace,
    _evaluate_claim_cell,
    _fill_declared_symbols,
)


def _cell_result(
    symbols,
    evidence_row=None,
    cell_key='n_id_abc',
    measured=None,
    claim='assert True',
):
    return _evaluate_claim_cell(
        claim,
        Symbols.decompose_symbol_defs(symbols)[0],
        evidence_row or {},
        cell_key,
        measured or set(),
    )


def test_a_moving_metric_does_not_move_the_cell():
    symbols = {'tolerance': {'type': 'float', 'value': 0.1}}
    first = _cell_result(symbols, {'metrics.n.mae': 0.030})
    second = _cell_result(symbols, {'metrics.n.mae': 0.031})

    assert first.result_id == second.result_id


def test_cells_of_one_recipe_stay_distinct():
    symbols = {'tolerance': {'type': 'float', 'value': 0.1}}
    a = _cell_result(symbols, {'metrics.n.mae': 0.03}, cell_key='n_id_abc')
    b = _cell_result(symbols, {'metrics.n.mae': 0.03}, cell_key='n_id_def')

    assert a.result_id != b.result_id


def test_a_result_cannot_shadow_a_symbol():
    with pytest.raises(ValueError, match='collides'):
        _cell_result(
            {'metrics': {'type': 'str', 'value': 'mine'}},
            {'metrics.n.mae': 0.03},
        )


def test_a_claim_reads_results_through_their_qualified_names():
    result = _cell_result(
        {'tolerance': {'type': 'float', 'value': 0.1}},
        {'metrics.n.mae': 0.03, 'metrics.n.unused': 9},
        claim='assert metrics.n.mae < tolerance',
    )

    assert result.status == 'VERIFIED'
    assert result.consumed == ['metrics.n.mae']


def test_a_declared_symbol_is_filled_from_the_result_of_that_name():
    symbols, measured = _fill_declared_symbols(
        {'mae': {'type': 'float'}, 'tolerance': {'value': 0.1}},
        {'metrics.n.mae': 0.03, 'metrics.n.rmse': 0.05},
    )
    assert symbols['mae']['value'] == 0.03
    assert symbols['tolerance']['value'] == 0.1
    assert measured == {'mae'}


def test_a_filled_symbol_stays_out_of_the_result_id():
    symbols = {'mae': {'type': 'float'}}
    first, measured = _fill_declared_symbols(
        symbols, {'metrics.n.mae': 0.03}
    )
    second, _ = _fill_declared_symbols(
        symbols, {'metrics.n.mae': 0.99}
    )

    a = _cell_result(first, {'metrics.n.mae': 0.03}, measured=measured)
    b = _cell_result(second, {'metrics.n.mae': 0.99}, measured=measured)

    assert a.result_id == b.result_id


def test_claim_result_namespace_reports_what_is_available():
    namespace = ClaimResultNamespace({'metrics.n.mae': 0.03})
    with pytest.raises(AttributeError, match="available: \\['mae'\\]"):
        namespace.bind()['metrics'].n.rmse
