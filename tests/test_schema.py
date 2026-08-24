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


@pytest.mark.parametrize('dependency_key', ['depends_on', 'depends'])
def test_symbol_dependency_aliases_validate(simple_card, dependency_key):
    card = {
        **simple_card,
        'symbols': {
            'x': {'value': 1},
            'y': {
                'python': 'y = x + 1',
                dependency_key: ['x'],
            },
        },
    }
    validated = EvaluationCardSchema.model_validate(card)
    assert validated.symbols['y'].depends_on == ['x']


def test_symbol_dependency_aliases_may_agree(simple_card):
    card = {
        **simple_card,
        'symbols': {
            'x': {'value': 1},
            'y': {
                'python': 'y = x + 1',
                'depends_on': ['x'],
                'depends': ['x'],
            },
        },
    }
    validated = EvaluationCardSchema.model_validate(card)
    assert validated.symbols['y'].depends_on == ['x']


def test_symbol_dependency_aliases_must_not_disagree(simple_card):
    card = {
        **simple_card,
        'symbols': {
            'x': {'value': 1},
            'z': {'value': 2},
            'y': {
                'python': 'y = x + 1',
                'depends_on': ['x'],
                'depends': ['z'],
            },
        },
    }
    with pytest.raises(
        ValidationError,
        match='`depends_on` and `depends` must agree',
    ):
        EvaluationCardSchema.model_validate(card)


@pytest.fixture(scope='module')
def kwdagger_card():
    card_path = (files('magnet') / 'examples' / 'llama_consistency' /
                 'llama_kwdagger.yaml')
    with card_path.open('r') as f:
        return yaml.safe_load(f)


def test_a_kwdagger_card_validates(kwdagger_card):
    card = EvaluationCardSchema.model_validate(kwdagger_card)
    assert card.kwdagger.result_node == 'llama_compare'


def test_llama_kwdagger_pipeline_is_inline_and_wired(kwdagger_card):
    from kwdagger.pipeline import coerce_pipeline

    pipeline = kwdagger_card['kwdagger']['pipeline']
    assert isinstance(pipeline, dict)
    dag = coerce_pipeline(pipeline)
    assert sorted(dag.node_dict) == ['llama_compare', 'llama_predict']
    assert dag.node_dict['llama_compare'].inputs['scores_fpath'].pred
    assert 'metrics.llama_compare' in kwdagger_card['claim']['python']


def test_shared_schema_allows_legacy_kwdagger_without_result_node(kwdagger_card):
    kwdagger = {
        k: v for k, v in kwdagger_card['kwdagger'].items()
        if k != 'result_node'
    }
    card = EvaluationCardSchema.model_validate(
        {**kwdagger_card, 'kwdagger': kwdagger}
    )
    assert card.kwdagger.result_node is None


def test_kwdagger_keys_magnet_does_not_read_are_passed_through(kwdagger_card):
    kwdagger = {**kwdagger_card['kwdagger'], 'skip_existing': False}
    card = EvaluationCardSchema.model_validate(
        {**kwdagger_card, 'kwdagger': kwdagger})
    assert card.kwdagger.skip_existing is False
