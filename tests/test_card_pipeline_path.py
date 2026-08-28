"""
Constructing a card resolves the paths written inside it.

A card may name a pipeline YAML file rather than inline the DAG or name a
callable. That path is relative to the card, so the card has to know which
directory it came from.
"""

import ubelt as ub
import yaml

from magnet.evaluation_new import NewEvaluationRecipe


def _write_card(dpath, pipeline):
    card = {
        'name': 'doubling',
        'title': 'Doubling',
        'description': 'Doubling a seed doubles it',
        'version': '1.0',
        'organizations': ['Kitware'],
        'submitter': {'name': 'Kitware TA2 Team', 'email': 'x@kitware.com'},
        'tags': ['test'],
        'links': [{
            'title': 'MAGNET',
            'url': 'https://github.com/AIQ-Kitware/aiq-magnet',
            'type': 'software',
        }],
        'claim': {'python': 'assert doubled == seed * 2'},
        'kwdagger': {
            'pipeline': pipeline,
            'result_node': 'demo_node',
            'matrix': {'demo_node.seed': [1]},
        },
    }
    fpath = ub.Path(dpath) / 'card.yaml'
    fpath.write_text(yaml.safe_dump(card, sort_keys=False))
    return fpath


def test_a_relative_pipeline_file_resolves_against_the_card(tmp_path):
    dpath = ub.Path(tmp_path) / 'cards'
    dpath.ensuredir()
    card = NewEvaluationRecipe(_write_card(dpath, 'dag.yaml'), tmp_path)
    # Not './dag.yaml' relative to the shell's cwd.
    assert card.kwdagger['pipeline'] == str(dpath / 'dag.yaml')


def test_a_pipeline_callable_is_left_alone(tmp_path):
    # Only file paths are resolved; an import path is not one.
    card = NewEvaluationRecipe(_write_card(tmp_path, 'some.module.pipeline()'),
                          tmp_path)
    assert card.kwdagger['pipeline'] == 'some.module.pipeline()'
