"""
The three demo cards, end to end.

Each runs offline in a fraction of a second, reaches a verdict, and leaves a
theory.json naming the relation its code declares.
"""
import json

import pytest
import ubelt as ub
from importlib.resources import files

from magnet.evaluation import EvaluationCard

CARDS = [
    ('theory_coin_flip_exact.yaml', 'tests', 'Examples.CoinFlip.Binomial'),
    ('theory_monte_carlo.yaml', 'approximates', 'Examples.Circle.AreaRatio'),
    ('theory_training_order.yaml', 'motivates', 'Examples.TrainingOrder.Why'),
]


def _run(card_name, output_path):
    card_fpath = files('magnet') / 'cards' / card_name
    card = EvaluationCard(card_fpath, output_path)
    status = card.evaluate()
    run_dpath = ub.Path(next(iter(ub.Path(output_path).iterdir())))
    return status, run_dpath


@pytest.mark.parametrize('card_name,relation,ref', CARDS)
def test_a_demo_card_verifies_and_records_its_relation(
        card_name, relation, ref, tmp_path):
    status, run_dpath = _run(card_name, tmp_path / 'runs')
    assert status == 'VERIFIED'

    theory = json.loads((run_dpath / 'theory.json').read_text())
    assert [link['relation'] for link in theory['links']] == [relation]
    assert theory['links'][0]['ref'] == ref

    # The entry the link points at travels with it, so the artifact reads on
    # its own without the index beside it.
    assert [entry['id'] for entry in theory['entries']] == [ref]
    assert theory['entries'][0]['statement']
    assert 'unresolved' not in theory


def test_the_link_names_the_code_that_declares_it(tmp_path):
    _, run_dpath = _run('theory_training_order.yaml', tmp_path / 'runs')
    link = json.loads((run_dpath / 'theory.json').read_text())['links'][0]
    assert link['qualname'] == 'training_order_sensitivity'
    assert link['file'].endswith('training_order/experiment.py')
    assert '..' not in link['file']
    assert link['line'] > 0


def test_the_question_is_carried_as_a_question(tmp_path):
    _, run_dpath = _run('theory_training_order.yaml', tmp_path / 'runs')
    entry = json.loads((run_dpath / 'theory.json').read_text())['entries'][0]
    assert entry['kind'] == 'question'


def test_a_card_without_a_theory_block_writes_no_artifact(tmp_path):
    _, run_dpath = _run('simple.yaml', tmp_path / 'runs')
    assert not (run_dpath / 'theory.json').exists()


def test_a_reference_with_no_entry_is_an_error(tmp_path):
    (tmp_path / 'demo.py').write_text(ub.codeblock(
        '''
        import magnet.theory as theory

        @theory.tests('Nope.Missing')
        def experiment():
            pass
        '''))
    (tmp_path / 'theory.yaml').write_text('entries: []\n')
    (tmp_path / 'card.yaml').write_text(ub.codeblock(
        '''
        title: unresolved
        description: names something no index defines
        claim:
          python: |
            assert True
        symbols:
          x:
            type: int
            value: 1
        theory:
          sources: [demo.py]
          indexes: [theory.yaml]
        '''))
    card = EvaluationCard(tmp_path / 'card.yaml', tmp_path / 'runs',
                          validate='off')
    with pytest.raises(ValueError, match='Nope.Missing'):
        card.evaluate()
