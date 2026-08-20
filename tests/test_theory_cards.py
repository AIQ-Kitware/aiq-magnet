"""
The three demo cards, end to end.

Each runs offline in a fraction of a second, reaches a verdict, and leaves a
theory.json naming the relation its code declares.
"""
import json
from importlib.resources import files

import pytest
import ubelt as ub
from pydantic import ValidationError

from magnet.evaluation import EvaluationCard
from magnet.schema import TheorySchema

CARDS = [
    ('coin_flip', 'tests', 'Examples.CoinFlip.Binomial'),
    ('monte_carlo', 'approximates', 'Examples.Circle.AreaRatio'),
    ('training_order', 'motivates', 'Examples.TrainingOrder.Why'),
]

EXAMPLES = files('magnet') / 'examples' / 'theory_links'


def _run(example, output_path):
    card_fpath = (files('magnet') / 'cards' / example if example.endswith('.yaml')
                  else EXAMPLES / example / 'card.yaml')
    card = EvaluationCard(card_fpath, output_path)
    status = card.evaluate()
    run_dpath = ub.Path(next(iter(ub.Path(output_path).iterdir())))
    return status, run_dpath


@pytest.mark.parametrize('example,relation,ref', CARDS)
def test_a_demo_card_verifies_and_records_its_relation(
        example, relation, ref, tmp_path):
    status, run_dpath = _run(example, tmp_path / 'runs')
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
    _, run_dpath = _run('training_order', tmp_path / 'runs')
    link = json.loads((run_dpath / 'theory.json').read_text())['links'][0]
    assert link['qualname'] == 'training_order_sensitivity'
    assert link['file'].endswith('training_order/experiment.py')
    assert '..' not in link['file']
    assert link['line'] > 0


def test_the_question_is_carried_as_a_question(tmp_path):
    _, run_dpath = _run('training_order', tmp_path / 'runs')
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
    (tmp_path / 'card.yaml').write_text(ub.codeblock(
        '''
        title: unresolved
        description: names something the card does not define
        claim:
          python: |
            assert True
        symbols:
          x:
            type: int
            value: 1
        theory:
          sources: [demo.py]
          entries: []
        '''))
    card = EvaluationCard(tmp_path / 'card.yaml', tmp_path / 'runs',
                          validate='off')
    with pytest.raises(ValueError, match='Nope.Missing'):
        card.evaluate()


def test_card_and_source_links_share_one_report(tmp_path):
    # A real card can put its claim-level relation in YAML while source
    # annotations identify a finer-grained implementation relationship.
    (tmp_path / 'demo.py').write_text(ub.codeblock(
        '''
        import magnet.theory as theory

        @theory.approximates('From.File', note='finite implementation proxy')
        def one():
            pass
        '''))
    (tmp_path / 'shared.yaml').write_text(
        'entries:\n  - id: From.File\n    kind: theorem\n')
    (tmp_path / 'card.yaml').write_text(ub.codeblock(
        '''
        title: both
        description: one entry from a file, one written out here
        claim:
          python: |
            assert True
        symbols:
          x:
            type: int
            value: 1
        theory:
          links:
            - relation: tests
              ref: From.Card
              note: the card claim directly evaluates this finite proposition
          sources: [demo.py]
          indexes: [shared.yaml]
          entries:
            - id: From.Card
              kind: definition
        '''))
    card = EvaluationCard(tmp_path / 'card.yaml', tmp_path / 'runs',
                          validate='off')
    assert card.evaluate() == 'VERIFIED'
    run_dpath = ub.Path(next(iter((tmp_path / 'runs').iterdir())))
    theory = json.loads((run_dpath / 'theory.json').read_text())
    assert sorted(e['id'] for e in theory['entries']) == ['From.Card', 'From.File']
    assert [link['relation'] for link in theory['links']] == ['tests', 'approximates']
    card_link, source_link = theory['links']
    assert card_link['note'].startswith('the card claim')
    assert 'file' not in card_link
    assert source_link['note'] == 'finite implementation proxy'
    assert source_link['file'].endswith('demo.py')


def test_an_unresolved_card_level_link_is_an_error(tmp_path):
    card = {
        'theory': {
            'links': [{'relation': 'tests', 'ref': 'Nope.Missing'}],
        }
    }
    from magnet.theory.cards import report_from_card
    with pytest.raises(ValueError, match='Nope.Missing'):
        report_from_card(card, tmp_path)


def test_theory_schema_rejects_silent_extra_keys():
    with pytest.raises(ValidationError, match='ledger'):
        TheorySchema.model_validate({'ledger': 'old-theory-ledger.json'})
