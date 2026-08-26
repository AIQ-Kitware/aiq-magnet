"""
A recipe must name itself.

``title`` is a sentence for a human. Machinery needs a short identifier it can
use as a tmux session name, a path component or a log prefix, and deriving one
from prose guesses. A recipe therefore declares ``name`` outright.
"""

import pytest
import yaml

from magnet.evaluation_new import NewEvaluationRecipe

_RECIPE = {
    'name': 'probe',
    'title': 'A readable sentence about the probe',
    'description': 'd',
    'version': 1.0,
    'organizations': ['Kitware'],
    'submitter': {'name': 'Kitware', 'email': 'aiq-ta2@kitware.com'},
    'tags': ['example'],
    'links': [{'title': 'MAGNET', 'url': 'https://example.invalid',
               'type': 'software'}],
    'claim': {'python': 'assert True'},
    'kwdagger': {
        'pipeline': {'emit': {'command': 'true'}},
        'result_node': 'emit',
    },
}


def _write(tmp_path, **overrides):
    card = dict(_RECIPE)
    card.update(overrides)
    card.pop('_drop', None)
    for key in overrides.get('_drop', ()):
        card.pop(key, None)
    card.pop('_drop', None)
    fpath = tmp_path / 'card.yaml'
    fpath.write_text(yaml.safe_dump(card))
    return fpath


def test_a_recipe_declares_its_name(tmp_path):
    recipe = NewEvaluationRecipe(_write(tmp_path), tmp_path / 'out')
    assert recipe.name == 'probe'
    # The prose title is a separate thing and stays untouched.
    assert recipe.title == 'A readable sentence about the probe'


def test_a_recipe_without_a_name_is_rejected(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        NewEvaluationRecipe(_write(tmp_path, _drop=('name',)), tmp_path / 'out')
    message = str(excinfo.value)
    assert '`name`' in message
    # The error has to say what to do, not just what is missing.
    assert 'title' in message


@pytest.mark.parametrize(
    'bad',
    [
        'has spaces',
        'has/slash',
        'has.dot',      # tmux treats a dot specially in session names
        'has:colon',    # and a colon separates window from session
        '',
    ],
)
def test_a_name_that_could_not_be_used_as_one_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError) as excinfo:
        NewEvaluationRecipe(_write(tmp_path, name=bad), tmp_path / 'out')
    assert '`name`' in str(excinfo.value)


def test_the_requirement_does_not_reach_legacy_cards(tmp_path):
    """Legacy cards keep working; the requirement is on recipes only."""
    from magnet.evaluation import EvaluationCard

    card = dict(_RECIPE)
    del card['name']
    del card['kwdagger']
    card['pipeline'] = {'emit': {'command': 'true'}}
    fpath = tmp_path / 'legacy.yaml'
    fpath.write_text(yaml.safe_dump(card))

    legacy = EvaluationCard(fpath, tmp_path / 'out', validate='off')
    assert legacy.title == 'A readable sentence about the probe'
