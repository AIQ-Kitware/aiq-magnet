"""Tests for the replacement kwdagger-native evaluation API."""
import sys
import textwrap

import pytest
import ubelt as ub
import yaml

import magnet.evaluation_new as evaluation_new
from magnet.evaluation import main as legacy_main
from magnet.evaluation_new import (
    NewEvaluationCLI,
    NewEvaluationRecipe,
    NewEvaluationResultCard,
)


SCRIPT = """
import json, pathlib, sys
args = dict(a.lstrip('-').split('=', 1) for a in sys.argv[1:])
out = pathlib.Path(args['results_fpath'])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'score': float(args['seed']) / 10}))
"""


@pytest.fixture
def kwdagger_recipe_fpath(tmp_path):
    dpath = ub.Path(tmp_path)
    script = dpath / 'emit.py'
    script.write_text(textwrap.dedent(SCRIPT))
    fpath = dpath / 'recipe.yaml'
    fpath.write_text(yaml.safe_dump({
        'title': 'new evaluator probe',
        'description': 'new evaluator probe',
        'version': '1.0',
        'organizations': ['Kitware'],
        'submitter': {'name': 't', 'email': 't@example.com'},
        'links': [],
        'tags': ['test'],
        'claim': {'python': 'assert metrics.emit.score < 100'},
        'kwdagger': {
            'result_node': 'emit',
            'pipeline': {'nodes': {'emit': {
                'executable': f'{sys.executable} {script}',
                'algo_params': {'seed': 1},
                'out_paths': {'results_fpath': 'results.json'},
            }}},
            'matrix': {'emit.seed': [1, 2]},
        },
    }, sort_keys=False))
    return fpath


def test_new_cli_does_not_expose_legacy_execution_options():
    keys = set(NewEvaluationCLI().keys())
    assert {
        'backend', 'tmux_workers', 'skip_existing', 'cache', 'max_configs'
    } <= keys
    assert not {
        'override', 'jobs', 'parallel_backend', 'queue_backend', 'workers'
    } & keys


def test_new_api_uses_recipe_and_result_names():
    assert 'main' not in vars(evaluation_new)
    assert not hasattr(evaluation_new, 'NewEvaluationCard')
    assert not hasattr(evaluation_new, 'NewEvaluationTask')
    assert not hasattr(evaluation_new, 'Results')
    assert callable(NewEvaluationCLI.main)


def test_new_cli_passes_execution_config_directly(
        kwdagger_recipe_fpath, tmp_path, monkeypatch):
    monkeypatch.setenv('MAGNET_QUEUE_BACKEND', 'definitely-not-a-backend')
    monkeypatch.setenv('MAGNET_TMUX_WORKERS', '999')

    output_path = ub.Path(tmp_path) / 'out'
    result_card = NewEvaluationCLI.main(argv=[
        str(kwdagger_recipe_fpath),
        '--output_path', str(output_path),
        '--params', 'matrix: {emit.seed: [7]}',
        '--backend', 'serial',
        '--skip_existing=0',
        '--cache=1',
        '--max_configs=1',
    ])

    assert isinstance(result_card, NewEvaluationResultCard)
    artifacts = sorted(
        (output_path / '_kwdagger' / 'emit').glob('*/results.json')
    )
    assert len(artifacts) == 1


def test_new_recipe_rejects_legacy_symbol_sweeps(
        kwdagger_recipe_fpath, tmp_path):
    data = yaml.safe_load(ub.Path(kwdagger_recipe_fpath).read_text())
    data['symbols'] = {'legacy_axis': {'sweep': [1, 2]}}
    ub.Path(kwdagger_recipe_fpath).write_text(
        yaml.safe_dump(data, sort_keys=False)
    )

    with pytest.raises(ValueError, match='symbol sweeps'):
        NewEvaluationRecipe(
            kwdagger_recipe_fpath,
            ub.Path(tmp_path) / 'out',
            validate='off',
        )


def test_new_recipe_rejects_legacy_pipeline(tmp_path):
    fpath = ub.Path(tmp_path) / 'legacy.yaml'
    fpath.write_text(yaml.safe_dump({
        'claim': {'python': 'assert True'},
        'pipeline': {
            'old_node': {
                'executable': 'echo',
                'out_paths': {'results_fpath': 'results.json'},
            },
        },
    }, sort_keys=False))

    with pytest.raises(ValueError, match='requires a kwdagger recipe'):
        NewEvaluationRecipe(
            fpath,
            ub.Path(tmp_path) / 'out',
            validate='off',
        )


def test_legacy_evaluator_points_kwdagger_cards_to_evaluate_new(
        kwdagger_recipe_fpath, tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        legacy_main(argv=[
            str(kwdagger_recipe_fpath),
            '--output_path', str(ub.Path(tmp_path) / 'out'),
            '--validate', 'off',
        ])
    message = str(exc_info.value)
    assert 'declares a `kwdagger:` pipeline' in message
    assert '`magnet evaluate`' in message
    assert 'magnet evaluate_new' in message
