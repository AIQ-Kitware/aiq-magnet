"""
A recipe's kwdagger matrix is a default that ``--params`` may override.

``--params`` speaks the same language as ``kwdagger schedule --params``, so one
recipe can run against models it does not name rather than being forked per
matrix variation.
"""

import textwrap

import pytest
import ubelt as ub
import yaml

from magnet.evaluation_new import NewEvaluationCLI, NewEvaluationRecipe

SCRIPT = """
import json, sys, pathlib
args = dict(a.lstrip('-').split('=', 1) for a in sys.argv[1:])
out = pathlib.Path(args['results_fpath'])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'score': float(args['seed']) / 10}))
"""


@pytest.fixture
def recipe_fpath(tmp_path):
    dpath = ub.Path(tmp_path)
    script = dpath / 'emit.py'
    script.write_text(textwrap.dedent(SCRIPT))
    fpath = dpath / 'recipe.yaml'
    fpath.write_text(yaml.safe_dump({
        'title': 'probe',
        'description': 'probe',
        'version': '1.0',
        'organizations': ['Kitware'],
        'submitter': {'name': 't', 'email': 't@example.com'},
        'links': [],
        'tags': ['test'],
        'claim': {'python': 'assert metrics.emit.score < 100'},
        'kwdagger': {
            'result_node': 'emit',
            'pipeline': {'nodes': {'emit': {
                'executable': f'python {script}',
                'algo_params': {'seed': 1},
                'out_paths': {'results_fpath': 'results.json'},
            }}},
            'matrix': {'emit.seed': [1, 2]},
        },
    }, sort_keys=False))
    return fpath


def _scores(result_card):
    return sorted(
        cell.result_values['metrics.emit.score']
        for cell in result_card.cell_results
    )


def test_params_replaces_an_axis_of_the_grid(recipe_fpath, tmp_path):
    recipe = NewEvaluationRecipe(recipe_fpath, ub.Path(tmp_path) / 'out')
    recipe.apply_params('matrix: {emit.seed: [7, 8, 9]}')
    result_card = recipe.evaluate(backend='serial')

    assert _scores(result_card) == [0.7, 0.8, 0.9]


def test_params_leaves_the_rest_of_the_block_alone(recipe_fpath, tmp_path):
    recipe = NewEvaluationRecipe(recipe_fpath, ub.Path(tmp_path) / 'out')
    recipe.apply_params('matrix: {emit.seed: [7]}')

    assert recipe.kwdagger['result_node'] == 'emit'
    assert 'nodes' in recipe.kwdagger['pipeline']


def test_params_may_be_a_file(recipe_fpath, tmp_path):
    params_fpath = ub.Path(tmp_path) / 'grid.yaml'
    params_fpath.write_text('matrix: {emit.seed: [7]}')

    recipe = NewEvaluationRecipe(recipe_fpath, ub.Path(tmp_path) / 'out')
    recipe.apply_params(str(params_fpath))
    result_card = recipe.evaluate(backend='serial')

    assert _scores(result_card) == [0.7]


def test_the_run_records_the_grid_that_ran(recipe_fpath, tmp_path):
    output_path = ub.Path(tmp_path) / 'out'
    recipe = NewEvaluationRecipe(recipe_fpath, output_path)
    recipe.apply_params('matrix: {emit.seed: [7]}')
    recipe.evaluate(backend='serial')

    written = yaml.safe_load(
        (output_path / recipe._run_hash / 'card.yaml').read_text())
    assert written['kwdagger']['matrix']['emit.seed'] == [7]


def test_a_different_grid_is_a_different_run(recipe_fpath, tmp_path):
    output_path = ub.Path(tmp_path) / 'out'

    first = NewEvaluationRecipe(recipe_fpath, output_path)
    first.apply_params('matrix: {emit.seed: [7]}')
    second = NewEvaluationRecipe(recipe_fpath, output_path)
    second.apply_params('matrix: {emit.seed: [8]}')

    assert first._recipe_hash != second._recipe_hash


def test_params_reaches_the_cli(recipe_fpath, tmp_path):
    output_path = ub.Path(tmp_path) / 'out'
    result_card = NewEvaluationCLI.main(argv=[
        str(recipe_fpath),
        '--output_path', str(output_path),
        '--params', 'matrix: {emit.seed: [7]}',
        '--backend', 'serial',
    ])

    assert result_card is not None
    assert _scores(result_card) == [0.7]
    artifacts = sorted((output_path / '_kwdagger' / 'emit').glob('*/results.json'))
    assert len(artifacts) == 1
