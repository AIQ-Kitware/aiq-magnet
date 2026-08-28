from importlib.resources import files

import pytest

from magnet.demo.helm_demodata import ensure_helm_llama_fixture_outputs
from magnet.evaluation import EvaluationCard


@pytest.mark.parametrize(
    'card_name',
    [
        'llama.yaml',
        'llama_pipeline.yaml',
        'llama_kwdagger.yaml',
    ],
)
def test_llama_card(llama_helm_data, tmp_path, card_name):
    data_path = llama_helm_data
    results_path = f'{tmp_path}/results'
    card_path = files('magnet') / 'cards' / card_name

    card = EvaluationCard(card_path, results_path)
    override_path(card, str(data_path / 'lite' / 'benchmark_output'))

    assert card.evaluate() == 'FALSIFIED'
    assert len(card.evaluations) == 36


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
