from types import SimpleNamespace

from magnet.evaluation import Metric, Symbols, _calculate_metrics


def _metric_metadata(
    *,
    threshold=None,
    lower_is_better=False,
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
                'lower_is_better': lower_is_better,
                'aggregation_strategy': aggregation_strategy,
            },
        }
    }


def test_threshold_metrics_bind_their_own_thresholds():
    metadata = {
        'loose': {
            'define_metric': {
                'lower_is_better': False,
                'aggregation_strategy': {
                    'type': 'threshold',
                    'parameters': {'threshold': 0.5},
                },
            },
        },
        'strict': {
            'define_metric': {
                'lower_is_better': False,
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


def test_threshold_direction_uses_lower_is_better():
    lower_metadata = _metric_metadata(
        threshold=0.5,
        lower_is_better=True,
    )
    lower_metric = Metric.build_metrics_from_symbol_metadata(lower_metadata)[0]
    assert lower_metric.calculate([0.4, 0.3]) is True
    assert lower_metric.calculate([0.4, 0.6]) is False

    higher_metadata = _metric_metadata(
        threshold=0.5,
        lower_is_better=False,
    )
    higher_metric = Metric.build_metrics_from_symbol_metadata(higher_metadata)[0]
    assert higher_metric.calculate([0.6, 0.7]) is True
    assert higher_metric.calculate([0.6, 0.4]) is False


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


def test_metric_results_use_stable_symbol_keys():
    metadata = _metric_metadata(
        strategy='mean',
        display_name='Average Score',
        lower_is_better=False,
    )
    metrics = Metric.build_metrics_from_symbol_metadata(metadata)
    evaluations = [
        SimpleNamespace(symbols=Symbols({'score': {'value': 1.0}})),
        SimpleNamespace(symbols=Symbols({'score': {'value': 3.0}})),
    ]

    calculated = _calculate_metrics(metrics, evaluations, metadata)

    assert calculated == {
        'score': {
            'display_name': 'Average Score',
            'value': 2.0,
            'lower_is_better': False,
            'aggregation_strategy': {'type': 'mean'},
        }
    }
