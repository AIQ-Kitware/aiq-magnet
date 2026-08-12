"""Tests for a card declaring what its claim is grounded on."""
import json

import pytest
import yaml

from magnet.evaluation import EvaluationCard
from magnet.theory import approximates, satisfies
from magnet.theory.cards import basis_from_card

THEOREMS = {
    'project': {'name': 'Example', 'commit': '0' * 40},
    'theorems': [
        {
            'declaration': 'Probability.WeakLaw.tendsto_sampleMean_mean',
            'informal': 'The sample mean converges to the population mean.',
            'hypotheses': [
                {'name': 'hiid', 'informal': 'the draws are i.i.d.'},
                {'name': 'hvar', 'informal': 'the population has finite variance'},
                {'name': 'hlim', 'informal': 'the sample size tends to infinity'},
            ],
            'axioms': ['propext', 'Classical.choice', 'Quot.sound'],
        }
    ],
}

CARD = {
    'title': 'Sample mean is close to the population mean',
    'description': 'A Monte-Carlo mean lands within a tolerance',
    'version': '1.0',
    'organizations': ['Kitware'],
    'submitter': {'name': 'Kitware TA2 Team', 'email': 'aiq-ta2@kitware.com'},
    'tags': ['example'],
    'links': [],
    'claim': {'python': 'assert abs(sample_mean - 50.0) < tolerance\n'},
    'symbols': {
        'tolerance': {'type': 'float', 'value': 25.0},
        'sample_mean': {'type': 'float', 'value': 50.0},
    },
    'theory': {
        'formalizations': ['theorems.yaml'],
        'grounds': [{'declaration': 'Probability.WeakLaw.tendsto_sampleMean_mean'}],
    },
}


@pytest.fixture
def card_root(tmp_path):
    (tmp_path / 'theorems.yaml').write_text(yaml.safe_dump(THEOREMS))
    (tmp_path / 'card.yaml').write_text(yaml.safe_dump(CARD))
    return tmp_path


def test_no_theory_block_means_no_basis():
    assert basis_from_card({'title': 'nothing declared'}) is None
    assert basis_from_card({'theory': {'formalizations': []}}) is None


def test_relaxations_come_from_the_code_not_the_card(card_root):
    # Annotated where the departure happens; the card never mentions these.
    # An audit ledger is what supplies them in practice.
    edges = [
        satisfies('Probability.WeakLaw.tendsto_sampleMean_mean::hvar'),
        approximates('Probability.WeakLaw.tendsto_sampleMean_mean::hlim', 'high'),
    ]

    coverage = basis_from_card(CARD, root=card_root, edges=edges).coverage()

    assert not coverage.is_complete
    assert [h.name for h in coverage.unaccounted] == ['hiid']


def test_a_grounded_card_writes_theory_json(card_root):
    card = EvaluationCard(card_root / 'card.yaml', card_root / 'runs')
    card.evaluate()

    run_dpath = next(iter((card_root / 'runs').iterdir()))
    written = json.loads((run_dpath / 'theory.json').read_text())

    assert written['coverage']['complete'] is False
    assert [h['name'] for h in written['coverage']['statements'][0]['unaccounted']] == [
        'hiid',
        'hvar',
        'hlim',
    ]
    assert written['formalizations'][0]['commit'] == '0' * 40
    # Beside the files the runner already writes, not instead of them.
    assert (run_dpath / 'verdict.json').exists()
    assert (run_dpath / 'card.yaml').exists()


def test_an_ungrounded_card_writes_no_theory_json(tmp_path):
    plain = {k: v for k, v in CARD.items() if k != 'theory'}
    (tmp_path / 'card.yaml').write_text(yaml.safe_dump(plain))

    card = EvaluationCard(tmp_path / 'card.yaml', tmp_path / 'runs')
    card.evaluate()

    run_dpath = next(iter((tmp_path / 'runs').iterdir()))
    assert card.basis is None
    assert not (run_dpath / 'theory.json').exists()
