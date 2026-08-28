from importlib.resources import files

import pytest

from magnet.demo.helm_demodata import ensure_helm_llama_fixture_outputs
from magnet.evaluation import EvaluationCard


LLAMA_MODELS = [
    'meta/llama-2-13b',
    'meta/llama-2-70b',
    'meta/llama-2-7b',
    'meta/llama-3-70b',
    'meta/llama-3-8b',
    'meta/llama-65b',
]

# Keep end-to-end pipeline execution representative rather than paying for all
# 36 subprocesses in every card implementation. This 2x2 matrix contains both
# self-comparisons and a deliberately falsifying cross-family comparison.
FAST_LLAMA_MODELS = [
    'meta/llama-2-7b',
    'meta/llama-3-70b',
]

CARD_NAMES = [
    'llama.yaml',
    'llama_pipeline.yaml',
    'llama_kwdagger.yaml',
]


@pytest.mark.parametrize('card_name', CARD_NAMES)
def test_llama_card_declares_full_matrix(tmp_path, card_name):
    """The shipped examples still declare the full 6x6 model sweep."""
    card_path = files('magnet') / 'cards' / card_name
    card = EvaluationCard(card_path, tmp_path / 'results')

    base_models, comp_models = _card_model_matrix(card)
    assert base_models == LLAMA_MODELS
    assert comp_models == LLAMA_MODELS
    assert len(base_models) * len(comp_models) == 36


@pytest.mark.parametrize('card_name', CARD_NAMES)
def test_llama_card(llama_helm_data, tmp_path, card_name):
    data_path = llama_helm_data
    results_path = f'{tmp_path}/results'
    card_path = files('magnet') / 'cards' / card_name

    card = EvaluationCard(card_path, results_path)
    override_path(card, str(data_path / 'lite' / 'benchmark_output'))
    _limit_model_matrix(card, FAST_LLAMA_MODELS)

    assert card.evaluate() == 'FALSIFIED'
    assert len(card.evaluations) == 4


def _card_model_matrix(card):
    if card.has_pipeline:
        params = card.pipeline['llama_predict']['algo_params']
        return params['base_model'], params['comp_model']
    elif card.has_kwdagger:
        matrix = card.kwdagger['matrix']
        return (
            matrix['llama_predict.base_model'],
            matrix['llama_predict.comp_model'],
        )
    else:
        return (
            card.symbols['base_model']['sweep'],
            card.symbols['comp_model']['sweep'],
        )


def _limit_model_matrix(card, models):
    """Shrink expensive execution while retaining multi-axis sweep coverage."""
    models = list(models)
    if card.has_pipeline:
        params = card.pipeline['llama_predict']['algo_params']
        params['base_model'] = models
        params['comp_model'] = models
    elif card.has_kwdagger:
        matrix = card.kwdagger['matrix']
        matrix['llama_predict.base_model'] = models
        matrix['llama_predict.comp_model'] = models
    else:
        card.replace({'base_model': models, 'comp_model': models})


def override_path(card, corrected_path):
    """
    manually replace data input path depending on definition
    """
    if card.has_pipeline:
        card.pipeline['llama_predict']['algo_params']['helm_runs_path'] = (
            corrected_path
        )

        # replace script with module call to avoid searching for path root
        python_script = card.pipeline['llama_predict']['executable'][:-3]
        python_module = ' -m '.join(python_script.replace('/', '.').split())

        card.pipeline['llama_predict']['executable'] = python_module
    elif card.has_kwdagger:
        card.kwdagger['matrix']['llama_predict.helm_runs_path'] = corrected_path
    else:
        card.replace({'helm_runs_path': corrected_path})


@pytest.fixture(scope='session')
def llama_helm_data():
    """Small local HELM Lite fixture; no GCS access or dataset download."""
    return ensure_helm_llama_fixture_outputs()
