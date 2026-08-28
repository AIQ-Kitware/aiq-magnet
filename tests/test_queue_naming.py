"""
A card's execution queue is named after the card.

cmd_queue's tmux backend matches sessions on the queue name to decide which
ones belong to this queue. With no name every card on the machine shared one
namespace, so starting one card reported an unrelated card's sessions as
conflicts and offered to kill them.
"""

import yaml

from magnet.evaluation_new import NewEvaluationRecipe

_RECIPE = {
    'name': 'incubilate_lift',
    'title': 'Lift under scale-up',
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


def _recipe(tmp_path, **overrides):
    card = dict(_RECIPE, **overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)
    fpath = tmp_path / 'card.yaml'
    fpath.write_text(yaml.safe_dump(card))
    return NewEvaluationRecipe(fpath, tmp_path / 'out')


def test_the_queue_is_named_after_the_card(tmp_path):
    assert _recipe(tmp_path).queue_name == 'schedule-incubilate_lift'


def test_different_cards_do_not_share_a_namespace(tmp_path):
    a = _recipe(tmp_path / 'a', name='princeton_drag')
    b = _recipe(tmp_path / 'b', name='incubilate_lift')
    assert a.queue_name != b.queue_name


def test_two_runs_of_one_card_do_share_a_name(tmp_path):
    """That collision is real and is meant to be reported as one."""
    a = _recipe(tmp_path / 'a', name='same_card')
    b = _recipe(tmp_path / 'b', name='same_card')
    assert a.queue_name == b.queue_name


def test_the_name_reaches_the_scheduler(tmp_path, monkeypatch):
    """The derived name is what kwdagger is actually asked for."""
    import magnet.evaluation_new as mod

    seen = {}

    class _Processor:
        def __init__(self, *args, **kwargs):
            pass

        def schedule(self, **options):
            seen.update(options)
            raise _Stop()

    class _Stop(Exception):
        pass

    monkeypatch.setattr(mod, 'KWDaggerProcessor', _Processor)
    recipe = _recipe(tmp_path)
    try:
        recipe.evaluate(backend='serial')
    except _Stop:
        pass

    assert seen['queue_name'] == 'schedule-incubilate_lift'


def test_an_explicit_queue_name_still_wins(tmp_path, monkeypatch):
    import magnet.evaluation_new as mod

    seen = {}

    class _Stop(Exception):
        pass

    class _Processor:
        def __init__(self, *args, **kwargs):
            pass

        def schedule(self, **options):
            seen.update(options)
            raise _Stop()

    monkeypatch.setattr(mod, 'KWDaggerProcessor', _Processor)
    recipe = _recipe(tmp_path)
    try:
        recipe.evaluate(backend='serial', queue_name='schedule-mine')
    except _Stop:
        pass

    assert seen['queue_name'] == 'schedule-mine'
