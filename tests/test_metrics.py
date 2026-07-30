from types import SimpleNamespace

import pytest

from magnet.evaluation import Metric, Symbols, _calculate_metrics


def _metric_metadata(
    *,
    threshold=None,
    objective='maximize',
    display_name=None,
    strategy='threshold',
):
    aggregation_strategy = {'type': strategy}
    if threshold is not None:
        aggregation_strategy['parameters'] = {'threshold': threshold}
    return {
        'score': {
            'display_name': display_name,
            'define_metric': {
                'objective': objective,
                'aggregation_strategy': aggregation_strategy,
            },
        }
    }


def test_threshold_metrics_bind_their_own_thresholds():
    metadata = {
        'loose': {
            'define_metric': {
                'objective': 'maximize',
                'aggregation_strategy': {
                    'type': 'threshold',
                    'parameters': {'threshold': 0.5},
                },
            },
        },
        'strict': {
            'define_metric': {
                'objective': 'maximize',
                'aggregation_strategy': {
                    'type': 'threshold',
                    'parameters': {'threshold': 0.9},
                },
            },
        },
    }
    metrics = {
        metric.name: metric
        for metric in Metric.build_metrics_from_symbol_metadata(metadata)
    }

    assert metrics['loose'].calculate([0.7]) is True
    assert metrics['strict'].calculate([0.7]) is False


def test_threshold_direction_uses_objective():
    minimize_metadata = _metric_metadata(
        threshold=0.5,
        objective='minimize',
    )
    minimize_metric = Metric.build_metrics_from_symbol_metadata(
        minimize_metadata
    )[0]
    assert minimize_metric.calculate([0.4, 0.3]) is True
    assert minimize_metric.calculate([0.4, 0.6]) is False

    maximize_metadata = _metric_metadata(
        threshold=0.5,
        objective='maximize',
    )
    maximize_metric = Metric.build_metrics_from_symbol_metadata(
        maximize_metadata
    )[0]
    assert maximize_metric.calculate([0.6, 0.7]) is True
    assert maximize_metric.calculate([0.6, 0.4]) is False


def test_mean_metric_uses_named_reducer():
    metadata = _metric_metadata(strategy='mean')
    metric = Metric.build_metrics_from_symbol_metadata(metadata)[0]

    assert metric.reducer.__name__ == 'fmean'
    assert metric.calculate([1.0, 3.0]) == 2.0


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
