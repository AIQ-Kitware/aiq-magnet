import pytest
import yaml
from importlib.resources import files
from pydantic import ValidationError
from magnet.schema import EvaluationCardSchema


@pytest.fixture(scope='module')
def simple_card():
    card_path = files('magnet') / 'cards' / 'simple.yaml'
    with card_path.open('r') as f:
        return yaml.safe_load(f)


def test_valid_card_passes_validation(simple_card):
    EvaluationCardSchema.model_validate(simple_card)


@pytest.mark.parametrize('broken_card', [
    pytest.param({'title': None}, id='null-required-field'),
    pytest.param({'claim': None}, id='missing-claim'),
    pytest.param({'submitter': {'name': 'No Email'}}, id='invalid-nested-schema'),
    pytest.param({'symbols': None}, id='no-backend-or-symbols'),
    pytest.param({'symbols': {'x': {}}}, id='symbol-missing-resolution'),
])
def test_invalid_card_fails_validation(simple_card, broken_card):
    card = {**simple_card, **broken_card}
    with pytest.raises(ValidationError):
        EvaluationCardSchema.model_validate(card)


def test_metric_objective_is_an_enum(simple_card):
    card = {
        **simple_card,
        'symbols': {
            'score': {
                'type': 'float',
                'value': 0.5,
                'metadata': {
                    'define_metric': {
                        'objective': 'increase',
                        'aggregation_strategy': {'type': 'mean'},
                    }
                },
            }
        },
    }
    with pytest.raises(ValidationError):
        EvaluationCardSchema.model_validate(card)


def test_custom_metric_strategy_still_validates(simple_card):
    card = {
        **simple_card,
        'symbols': {
            'score': {
                'type': 'float',
                'value': 0.5,
                'metadata': {
                    'define_metric': {
                        'objective': 'maximize',
                        'aggregation_strategy': {'type': 'custom'},
                    }
                },
            }
        },
    }
    EvaluationCardSchema.model_validate(card)
