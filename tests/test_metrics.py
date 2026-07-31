from types import SimpleNamespace

import pytest

from magnet.evaluation import Metric, Symbols, _calculate_metrics


def _metric_metadata(
    *,
    objective='maximize',
    display_name=None,
    strategy='mean',
):
    aggregation_strategy = {'type': strategy}
    return {
        'score': {
            'display_name': display_name,
            'define_metric': {
                'objective': objective,
                'aggregation_strategy': aggregation_strategy,
            },
        }
    }


def test_mean_metric_uses_named_reducer():
    metadata = _metric_metadata(strategy='mean')
    metric = Metric.build_metrics_from_symbol_metadata(metadata)[0]

    assert metric.reducer.__name__ == 'fmean'
    assert metric.aggregate_calculate([1.0, 3.0]) == 2.0


def test_missing_metric_value_does_not_compute_partial_aggregate():
    metadata = _metric_metadata(strategy='mean')
    metrics = Metric.build_metrics_from_symbol_metadata(metadata)
    evaluation_sets = [
        [
            SimpleNamespace(symbols=Symbols({'score': {'value': None}})),
            SimpleNamespace(symbols=Symbols({'score': {'value': 1.0}})),
        ],
        [
            SimpleNamespace(symbols=Symbols({'score': {'value': 1.0}})),
            SimpleNamespace(symbols=Symbols({'score': {'value': None}})),
        ],
    ]

    for evaluations in evaluation_sets:
        calculated = _calculate_metrics(metrics, evaluations, metadata)
        assert calculated == {}


def test_metric_results_keep_original_display_name_shape():
    metadata = _metric_metadata(
        strategy='mean',
        display_name='Average Score',
        objective='maximize',
    )
    metrics = Metric.build_metrics_from_symbol_metadata(metadata)
    evaluations = [
        SimpleNamespace(symbols=Symbols({'score': {'value': 1.0}})),
        SimpleNamespace(symbols=Symbols({'score': {'value': 3.0}})),
    ]

    calculated = _calculate_metrics(metrics, evaluations, metadata)

    assert calculated == {'Average Score': 2.0}


def test_custom_metric_remains_not_implemented():
    metadata = _metric_metadata(strategy='custom')

    with pytest.raises(NotImplementedError):
        Metric.build_metrics_from_symbol_metadata(metadata)
