"""The shipped aiq-magnet-evals recipes (integration plan M7).

Static checks run everywhere. End-to-end runs need the engine workers named in
``aiq_evals_support`` and run every recipe through ``evaluate_new``.
"""
import importlib.resources
import json

import pytest
import yaml
from aiq_evals_support import (
    HELM_PYTHON,
    INSPECT_PYTHON,
    OLMO_PYTHON,
    evaluate,
    latest_run_dir,
    needs_helm,
    needs_inspect,
    needs_olmo,
    run_verdicts,
)

RECIPES = ('helm_generation', 'inspect_generation', 'inspect_agent', 'olmo_generation',
           'olmo_agent', 'mixed_engines')


def recipe_path(name):
    return importlib.resources.files('magnet') / 'examples' / 'aiq_evals' / f'{name}.yaml'


def test_recipes_ship_with_the_package():
    folder = importlib.resources.files('magnet') / 'examples' / 'aiq_evals'
    shipped = {p.name for p in folder.iterdir() if p.name.endswith(('.yaml', '.md'))}
    assert shipped == {f'{name}.yaml' for name in RECIPES} | {'README.md'}


@pytest.mark.parametrize('name', RECIPES)
def test_recipes_use_installed_example_modules(name):
    text = recipe_path(name).read_text()
    assert 'tests.' not in text  # no aiq-magnet-evals source checkout needed
    card = yaml.safe_load(text)
    assert [link['url'] for link in card['links']] == ['https://github.com/Erotemic/aiq-magnet-evals']
    assert card['kwdagger']['result_node'] == 'evaluate'


@pytest.mark.parametrize('name', RECIPES)
def test_recipes_compile_in_a_dry_run_without_engines(name, tmp_path, monkeypatch):
    from magnet.backends.aiq_evals import pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError('a dry run must not resolve')

    monkeypatch.setattr(pipeline, 'run_preflight', forbidden)
    _, card = evaluate(recipe_path(name), tmp_path / 'out', dry_run=True)
    assert card.result == 'NOT_EVALUATED'


def _run(name, tmp_path, workers):
    params = {'matrix': {f'{node}.worker_python': python for node, python in workers.items()}}
    return _run_with(name, tmp_path, params)


def _run_with(name, tmp_path, params):
    out = tmp_path / 'out'
    _, card = evaluate(recipe_path(name), out, params=params)
    verdicts = run_verdicts(latest_run_dir(out))
    return card, verdicts


@needs_helm
def test_helm_generation_recipe(tmp_path):
    card, verdicts = _run('helm_generation', tmp_path, {'evaluate': HELM_PYTHON})
    assert card.result == 'VERIFIED' and len(verdicts) == 1


@needs_inspect
@pytest.mark.parametrize('name', ['inspect_generation', 'inspect_agent'])
def test_inspect_recipes(name, tmp_path):
    card, verdicts = _run(name, tmp_path, {'evaluate': INSPECT_PYTHON})
    assert card.result == 'VERIFIED', verdicts
    assert verdicts[0]['symbols']['metrics.evaluate.score'] == 1.0


@needs_olmo
def test_olmo_generation_recipe(tmp_path):
    card, verdicts = _run('olmo_generation', tmp_path, {'evaluate': OLMO_PYTHON})
    assert card.result == 'VERIFIED', verdicts


@needs_olmo
def test_olmo_agent_recipe_against_the_example_endpoint(tmp_path, monkeypatch):
    from magnet_evals.examples.chat_server import chat_server

    card_data = yaml.safe_load(recipe_path('olmo_agent').read_text())
    models = card_data['kwdagger']['pipeline']['nodes']['evaluate']['algo_params']['models']
    monkeypatch.setenv('OPENAI_API_KEY', 'example-local-key')
    with chat_server() as port:
        models[0]['provider_options']['base_url'] = f'http://127.0.0.1:{port}/v1'
        params = {
            'pipeline': {'nodes': {'evaluate': {'algo_params': {'models': models}}}},
            'matrix': {'evaluate.worker_python': OLMO_PYTHON},
        }
        card, verdicts = _run_with('olmo_agent', tmp_path, params)
    assert card.result == 'VERIFIED', verdicts
    for path in (tmp_path / 'out').rglob('*.json'):
        assert 'example-local-key' not in path.read_text(errors='ignore'), path


@needs_helm
@needs_inspect
def test_mixed_engine_comparison_recipe(tmp_path):
    card, verdicts = _run('mixed_engines', tmp_path, {'helm': HELM_PYTHON, 'inspect': INSPECT_PYTHON})
    assert card.result == 'VERIFIED', verdicts
    (verdict,) = verdicts
    symbols = verdict['symbols']
    assert symbols['metrics.evaluate.comparable'] is True
    # Two engines, two distinct measurement identities: no cross-engine collision.
    evidence = verdict['evidence']
    identities = {evidence[f'metrics.evaluate.{side}.measurement_identity'] for side in ('left', 'right')}
    engines = {evidence[f'metrics.evaluate.{side}.engine'] for side in ('left', 'right')}
    assert len(identities) == 2 and engines == {'helm', 'inspect_ai'}
    comparison = next((tmp_path / 'out' / '_kwdagger' / 'evaluate').glob('*/comparison.json'))
    assert json.loads(comparison.read_text())['mapping'].startswith('both are exact-match')
