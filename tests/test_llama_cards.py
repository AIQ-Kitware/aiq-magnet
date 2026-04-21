import pytest

pytest.importorskip('gcsfs')

from importlib.resources import files

from magnet.backends.helm.cli import download_helm_results
from magnet.evaluation import EvaluationCard


@pytest.mark.parametrize(
    'card_name',
    [
        'llama.yaml',
        'llama_pipeline.yaml',
        'llama_kwdagger.yaml',
    ],
)
def test_llama_card(run_download, tmp_path, card_name):
    data_path = run_download
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
def run_download(tmp_path_factory):
    """
    Follow README download script (HELM lite v1.0.0) and collect llama-3 results from HELM lite v1.2.0
    """
    tmp_path = tmp_path_factory.mktemp('helm_data')
    helm_dir = tmp_path / 'data' / 'crfm-helm-public'
    helm_dir.mkdir(parents=True, exist_ok=True)

    download_helm_results.main(
        argv=False,
        download_dir=helm_dir,
        benchmark='lite',
        version='v1.0.0',
        runs='regex:mmlu.*model=.*llama.*',
    )
    download_helm_results.main(
        argv=False,
        download_dir=helm_dir,
        benchmark='lite',
        version='v1.2.0',
        runs='regex:mmlu.*model=.*llama.*',
    )

    return helm_dir
