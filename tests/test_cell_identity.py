"""
A cell's identity comes from the node that produced it, not from its results.
"""

import pytest

from magnet.evaluation import (
    Claim, EvaluationTask, Results, Symbols, _fill_declared_symbols)


def _task(symbols, results=None, cell_key=None):
    return EvaluationTask(
        Claim({'python': 'assert True'}),
        Symbols.decompose_symbol_defs(symbols)[0],
        results=results,
        cell_key=cell_key,
    )


def test_a_moving_metric_does_not_move_the_cell():
    # Results used to be bound as symbols, so re-running a card whose metric
    # drifted wrote a second verdict beside the first instead of replacing it.
    symbols = {'tolerance': {'type': 'float', 'value': 0.1}}
    first = _task(symbols, {'metrics.n.mae': 0.030}, cell_key='n_id_abc')
    second = _task(symbols, {'metrics.n.mae': 0.031}, cell_key='n_id_abc')

    assert first.cell_id == second.cell_id


def test_cells_of_one_card_stay_distinct():
    symbols = {'tolerance': {'type': 'float', 'value': 0.1}}
    a = _task(symbols, {'metrics.n.mae': 0.03}, cell_key='n_id_abc')
    b = _task(symbols, {'metrics.n.mae': 0.03}, cell_key='n_id_def')

    assert a.cell_id != b.cell_id


def test_a_card_without_a_pipeline_keeps_hashing_its_symbols():
    task = _task({'seed': {'type': 'int', 'value': 1}})
    assert task.cell_id == task._execution_hash


def test_a_result_cannot_shadow_a_symbol():
    task = _task(
        {'metrics': {'type': 'str', 'value': 'mine'}},
        {'metrics.n.mae': 0.03},
    )
    with pytest.raises(ValueError, match='collides'):
        task.execute()


def test_a_claim_reads_results_through_their_qualified_names():
    task = EvaluationTask(
        Claim({'python': 'assert metrics.n.mae < tolerance'}),
        Symbols.decompose_symbol_defs(
            {'tolerance': {'type': 'float', 'value': 0.1}})[0],
        results={'metrics.n.mae': 0.03, 'metrics.n.unused': 9},
    )
    status, _ = task.execute()

    assert status == 'VERIFIED'
    assert task.log['consumed'] == ['metrics.n.mae']


def test_a_declared_symbol_is_filled_from_the_result_of_that_name():
    # How a card that defines a metric gets its value: it names the symbol,
    # and the result node supplies it.
    symbols, measured = _fill_declared_symbols(
        {'mae': {'type': 'float'}, 'tolerance': {'value': 0.1}},
        {'metrics.n.mae': 0.03, 'metrics.n.rmse': 0.05},
    )
    assert symbols['mae']['value'] == 0.03
    assert symbols['tolerance']['value'] == 0.1
    assert measured == {'mae'}


def test_a_filled_symbol_stays_out_of_the_cell_id():
    symbols = {'mae': {'type': 'float'}}
    first, _ = _fill_declared_symbols(symbols, {'metrics.n.mae': 0.03})
    second, measured = _fill_declared_symbols(symbols, {'metrics.n.mae': 0.99})

    a = EvaluationTask(
        Claim({'python': 'assert True'}),
        Symbols.decompose_symbol_defs(first)[0],
        cell_key='n_id_abc', measured=measured)
    b = EvaluationTask(
        Claim({'python': 'assert True'}),
        Symbols.decompose_symbol_defs(second)[0],
        cell_key='n_id_abc', measured=measured)

    assert a.cell_id == b.cell_id


def test_results_report_what_is_available():
    results = Results({'metrics.n.mae': 0.03})
    with pytest.raises(AttributeError, match="available: \\['mae'\\]"):
        results.bind()['metrics'].n.rmse
