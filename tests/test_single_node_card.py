"""
The smallest useful kwdagger card: one node, declared inline.
"""

import os
import textwrap
from collections.abc import Iterable

import pytest
import ubelt as ub
import yaml

from magnet.evaluation_new import NewEvaluationRecipe

SCRIPT = """
import json, sys, pathlib
args = dict(a.lstrip('-').split('=', 1) for a in sys.argv[1:])
out = pathlib.Path(args['results_fpath'])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'result': {'metrics': {'score': float(args['seed']) / 10}}}))
"""


def _card_data(
    script: str | os.PathLike[str],
    seeds: Iterable[int],
    result_node: str | None = 'emit',
) -> dict[str, object]:
    kwdagger: dict[str, object] = {
        'pipeline': {
            'nodes': {
                'emit': {
                    'executable': f'python {script}',
                    'algo_params': {'seed': 1},
                    'out_paths': {'results_fpath': 'results.json'},
                    'primary_out_key': 'results_fpath',
                },
            },
        },
        'matrix': {'emit.seed': list(seeds)},
    }
    if result_node is not None:
        kwdagger['result_node'] = result_node
    return {
        'name': 'probe',
        'title': 'probe',
        'description': 'probe',
        'version': '1.0',
        'organizations': ['Kitware'],
        'submitter': {'name': 't', 'email': 't@example.com'},
        'links': [],
        'tags': ['test'],
        'claim': {'python': 'assert metrics.emit.score < 100'},
        'kwdagger': kwdagger,
    }



@pytest.fixture
def write_card(tmp_path):
    dpath = ub.Path(tmp_path)
    script = dpath / 'emit.py'
    script.write_text(textwrap.dedent(SCRIPT))

    def make(seeds, result_node='emit'):
        fpath = dpath / 'card.yaml'
        fpath.write_text(yaml.safe_dump(
            _card_data(script, seeds, result_node), sort_keys=False))
        return fpath

    return make


def test_a_one_node_pipeline_needs_no_python(write_card, tmp_path):
    # An inline `nodes:` mapping, a matrix, and the node's name. No module
    # path, no pipeline function, no claim node.
    recipe = NewEvaluationRecipe(write_card([1, 2]), ub.Path(tmp_path) / 'out')
    result_card = recipe.evaluate(backend='serial')
    assert result_card.result == 'VERIFIED'
    assert len(result_card.cell_results) == 2
    assert {e.cell_key for e in result_card.cell_results} != {None}


def test_a_kwdagger_card_must_name_its_result_node(write_card, tmp_path):
    with pytest.raises(ValueError, match='result_node'):
        NewEvaluationRecipe(
            write_card([1], result_node=None), ub.Path(tmp_path) / 'out')
