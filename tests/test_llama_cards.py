import json
from importlib.resources import files

import pytest
import ubelt as ub

from magnet.demo.helm_demodata import ensure_helm_llama_fixture_outputs
from magnet.evaluation import EvaluationCard
from magnet.evaluation_new import NewEvaluationRecipe

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

# The kwdagger recipe is an example rather than a card, and `evaluate_new`
# runs it; the other two remain legacy cards run by `magnet evaluate`.
CARDS = [
    ('cards/llama.yaml', False),
    ('cards/llama_pipeline.yaml', False),
    ('examples/llama_consistency/llama_kwdagger.yaml', True),
]


def _load(card_relpath, use_new_evaluator, results_path):
    card_path = files('magnet').joinpath(*card_relpath.split('/'))
    card_cls = NewEvaluationRecipe if use_new_evaluator else EvaluationCard
    return card_cls(card_path, results_path)


@pytest.mark.parametrize('card_relpath,use_new_evaluator', CARDS)
def test_llama_card_declares_full_matrix(
        tmp_path, card_relpath, use_new_evaluator):
    """The shipped examples still declare the full 6x6 model sweep."""
    card = _load(card_relpath, use_new_evaluator, tmp_path / 'results')

    base_models, comp_models = _card_model_matrix(card)
    assert base_models == LLAMA_MODELS
    assert comp_models == LLAMA_MODELS
    assert len(base_models) * len(comp_models) == 36


@pytest.mark.parametrize('card_relpath,use_new_evaluator', CARDS)
def test_llama_card(
        llama_helm_data, tmp_path, card_relpath, use_new_evaluator):
    data_path = llama_helm_data
    results_path = f'{tmp_path}/results'

    card = _load(card_relpath, use_new_evaluator, results_path)
    if use_new_evaluator:
        assert card.evidence_scope == 'requested'
    override_path(card, str(data_path / 'lite' / 'benchmark_output'))
    _limit_model_matrix(card, FAST_LLAMA_MODELS)

    expected_cells = len(FAST_LLAMA_MODELS) ** 2
    if use_new_evaluator:
        assert card.evaluate(backend='serial').result == 'FALSIFIED'
        assert len(card.result_card.cell_results) == expected_cells
        # The kwdagger recipe carries the same symbol metadata the legacy
        # card does, so the dashboard contract is identical on both routes.
        # The names differ: a legacy card has only a bare `base_score`, while
        # a recipe says which node's, and qualifies by namespace as well where
        # that is the meaning.
        written = sorted(ub.Path(results_path).glob('*/symbol_metadata.json'))
        assert len(written) == 1
        assert json.loads(written[0].read_text()) == {
            'llama_evaluate.base_score': {
                'display_name': 'Average Exact Match',
                'display': True,
                'define_metric': {
                    'objective': 'maximize',
                    'aggregation_strategy': {'type': 'mean'},
                },
            },
            'llama_evaluate.gathered_runs': {
                'display_name': 'HELM runs gathered',
                'display': True,
            },
        }
        assert 'Average Exact Match' in card.result_card.metrics
    else:
        assert card.evaluate() == 'FALSIFIED'
        assert len(card.evaluations) == expected_cells


def _card_model_matrix(card):
    if card.has_pipeline:
        params = card.pipeline['llama_predict']['algo_params']
        return params['base_model'], params['comp_model']
    elif card.has_kwdagger:
        matrix = card.kwdagger['matrix']
        return (
            matrix['llama_evaluate.base_model'],
            matrix['llama_evaluate.comp_model'],
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
        matrix['llama_evaluate.base_model'] = models
        matrix['llama_evaluate.comp_model'] = models
        # Only materialize what the comparison actually needs.
        matrix['materialize_run.model'] = models
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
        # The recipe materializes runs out of a precomputed HELM root, so it
        # takes the cache root rather than a benchmark_output directory.
        matrix = card.kwdagger['matrix']
        matrix['materialize_run.precomputed_root'] = str(
            ub.Path(corrected_path).parent.parent
        )
        # The hermetic fixture carries these two subjects.
        matrix['materialize_run.subject'] = ['abstract_algebra', 'anatomy']
    else:
        card.replace({'helm_runs_path': corrected_path})


@pytest.fixture(scope='session')
def llama_helm_data():
    """Small local HELM Lite fixture; no GCS access or dataset download."""
    return ensure_helm_llama_fixture_outputs()
