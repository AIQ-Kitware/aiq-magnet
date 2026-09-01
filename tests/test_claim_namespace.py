"""
How a Python claim addresses one kwdagger aggregate row.

A row's columns are qualified by namespace and node --
``metrics.llama_compare.gap``. The namespace is not a card author's choice:
loaders supply ``metrics``/``machine``/``resources``/``context`` and kwdagger
aggregate supplies ``params``/``resolved_params``. So a claim may also address
a column by node alone, wherever the node reports that name once.
"""

import pytest

from magnet.evaluation_new import ClaimResultNamespace


ROW = {
    'node': 'compare',
    'metrics.compare.gap': 0.15,
    'metrics.compare.threshold': 0.1,
    'metrics.predict.base_score': 0.58,
    'params.predict.base_model': 'family/big',
    'resolved_params.predict.base_model': 'family/big',
    'specified.params.predict.base_model': 1,
    'machine.predict.host': 'somehost',
}


def _nodes(row=None):
    return ClaimResultNamespace(row or ROW).bind_nodes()


def _namespaces(row=None):
    return ClaimResultNamespace(row or ROW).bind()


def test_qualified_path_still_addresses_a_column():
    assert _namespaces()['metrics'].compare.gap == 0.15


def test_node_view_drops_the_namespace():
    assert _nodes()['compare'].gap == 0.15
    assert _nodes()['predict'].base_score == 0.58


def test_node_view_reaches_across_namespaces():
    """One node's columns arrive under several namespaces; the view is flat."""
    predict = _nodes()['predict']
    assert predict.base_score == 0.58   # metrics.
    assert predict.host == 'somehost'   # machine.


def test_node_names_tolerate_the_namespace_depth_varying():
    """`metrics` puts the node second; `specified.params` puts it third."""
    assert ClaimResultNamespace(ROW).node_names() == ['compare', 'predict']


def test_a_node_is_not_confused_with_a_namespace():
    """`params` is interior to `specified.params.predict.x`, not a node."""
    assert 'params' not in ClaimResultNamespace(ROW).node_names()


def test_agreeing_columns_collapse_to_their_shared_value():
    """`params` and `resolved_params` echo one another when nothing overrode."""
    assert _nodes()['predict'].base_model == 'family/big'


def test_disagreeing_columns_refuse_to_resolve():
    row = dict(ROW, **{'resolved_params.predict.base_model': 'family/small'})
    with pytest.raises(AttributeError, match='disagree'):
        _nodes(row)['predict'].base_model


def test_a_disagreeing_name_still_lists_and_names_its_columns():
    """Refusing to guess is only useful if it says what the choices were."""
    row = dict(ROW, **{'resolved_params.predict.base_model': 'family/small'})
    predict = _nodes(row)['predict']
    assert 'base_model' in predict.keys()
    with pytest.raises(AttributeError, match='resolved_params.predict.base_model'):
        predict.base_model


def test_provenance_columns_never_reach_a_node_view():
    """`specified.params.*` is always 1 and would fake a disagreement."""
    assert _nodes()['predict'].base_model == 'family/big'


def test_a_node_view_records_the_qualified_column_it_came_from():
    """Evidence stays traceable no matter which spelling the claim used."""
    namespace = ClaimResultNamespace(ROW)
    nodes = namespace.bind_nodes()
    nodes['compare'].gap
    nodes['predict'].host
    assert namespace.accessed == {
        'metrics.compare.gap', 'machine.predict.host',
    }


def test_missing_name_reports_what_is_available():
    with pytest.raises(AttributeError, match="no 'rmse' under 'compare'"):
        _nodes()['compare'].rmse


def test_a_row_leaf_is_bound_as_a_plain_value():
    assert _namespaces()['node'] == 'compare'


def test_introspection_lists_children_not_dotted_keys():
    metrics = _namespaces()['metrics']
    assert metrics._children() == ['compare', 'predict']
    assert 'compare' in dir(metrics)
    assert metrics.keys() == [
        'compare.gap', 'compare.threshold', 'predict.base_score',
    ]


def test_introspection_does_not_count_as_consuming_evidence():
    """A repr in a debugger must not enlarge what the card claims it used."""
    namespace = ClaimResultNamespace(ROW)
    metrics = namespace.bind()['metrics']
    repr(metrics)
    dir(metrics)
    metrics.keys()
    assert namespace.accessed == set()


def test_items_and_values_do_count_as_consuming_evidence():
    namespace = ClaimResultNamespace(ROW)
    dict(namespace.bind()['metrics'].items())
    assert 'metrics.compare.gap' in namespace.accessed


def test_a_view_reads_like_a_mapping():
    metrics = _namespaces()['metrics']
    assert metrics['compare.gap'] == 0.15
    assert 'compare.gap' in metrics
    assert 'compare' in metrics
    assert len(metrics) == 3
    assert sorted(metrics) == metrics.keys()


def test_repr_says_where_it_is_and_what_is_under_it():
    assert repr(_namespaces()['metrics']) == (
        '<ClaimResultNamespace metrics: compare, predict>'
    )
    assert repr(_nodes()['compare']) == (
        '<ClaimResultNamespace compare: gap, threshold>'
    )


def _verdict(claim, row=None, symbols=None):
    from magnet.evaluation import Symbols
    from magnet.evaluation_new import _evaluate_claim_cell
    return _evaluate_claim_cell(
        claim,
        Symbols.decompose_symbol_defs(symbols or {})[0],
        row or ROW,
        'compare_id_abc',
        set(),
    )


def test_a_claim_can_use_the_node_view():
    result = _verdict('assert compare.gap > compare.threshold')
    assert result.status == 'VERIFIED'
    assert result.consumed == [
        'metrics.compare.gap', 'metrics.compare.threshold',
    ]


def test_a_claim_can_still_use_the_qualified_path():
    result = _verdict('assert metrics.compare.gap > metrics.compare.threshold')
    assert result.status == 'VERIFIED'
    assert result.consumed == [
        'metrics.compare.gap', 'metrics.compare.threshold',
    ]


def test_a_declared_symbol_outranks_a_node_of_the_same_name():
    """The node view is a convenience; it must not break an existing card."""
    result = _verdict(
        'assert compare == 42',
        symbols={'compare': {'value': 42}},
    )
    assert result.status == 'VERIFIED'


def test_a_declared_symbol_still_collides_with_a_namespace():
    with pytest.raises(ValueError, match='collides'):
        _verdict('assert True', symbols={'metrics': {'value': 1}})
