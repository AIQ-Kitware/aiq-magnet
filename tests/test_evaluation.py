import json

import pytest
import yaml

from magnet.evaluation import EvaluationCard, Symbol, Symbols


TEST_CARD_TEXT = """
claim:
  python: |
    assert score >= 0

symbols:
  x:
    sweep: [1.0, 3.0]

  score:
    metadata:
      display_name: "Average Score"
      define_metric:
        objective: maximize
        aggregation_strategy:
          type: mean
    type: float
    depends_on:
      - x
    python: |
      score = x
"""


def test_evaluation_preserves_metrics(tmp_path):
    card_fpath = tmp_path / 'card.yaml'
    card_fpath.write_text(TEST_CARD_TEXT)

    output_path = tmp_path / 'results'

    card = EvaluationCard(
        card_fpath,
        output_path,
        validate='off',
    )

    assert card.evaluate(
        jobs=1,
    ) == 'VERIFIED'

    run_dpath = next(output_path.iterdir())

    aggregate = json.loads(
        (run_dpath / 'verdict.json').read_text()
    )

    assert aggregate['metrics'] == {
        'Average Score': 2.0,
    }

    # Also catches the unresolved-parent execution-hash bug.
    for claim_hash in aggregate['claims']:
        assert (
            run_dpath
            / 'results'
            / claim_hash
            / 'verdict.json'
        ).exists()


def test_parallel_evaluation_preserves_metrics(tmp_path):
    card_fpath = tmp_path / 'card.yaml'
    card_fpath.write_text(TEST_CARD_TEXT)

    output_path = tmp_path / 'results'

    card = EvaluationCard(
        card_fpath,
        output_path,
        validate='off',
    )

    assert card.evaluate(
        jobs=2,
        parallel_backend='loky',
    ) == 'VERIFIED'

    run_dpath = next(output_path.iterdir())

    aggregate = json.loads(
        (run_dpath / 'verdict.json').read_text()
    )

    assert aggregate['metrics'] == {
        'Average Score': 2.0,
    }

    # Also catches the unresolved-parent execution-hash bug.
    for claim_hash in aggregate['claims']:
        assert (
            run_dpath
            / 'results'
            / claim_hash
            / 'verdict.json'
        ).exists()


@pytest.mark.parametrize('dependency_key', ['depends_on', 'depends'])
def test_symbol_dependency_alias_orders_resolution(dependency_key):
    """
    Check that symbol are ordered topologically by the dependency graph.
    """
    symbols = Symbols({
        'y': {
            'type': 'int',
            'python': 'y = x + 1',
            dependency_key: ['x'],
        },
        'x': {'type': 'int', 'value': 1},
    })

    symbols.resolve()

    assert symbols()['y'] == 2


def test_symbol_dependency_aliases_must_agree():
    with pytest.raises(
        ValueError,
        match='`depends_on` and `depends` disagree',
    ):
        Symbol('y', {'depends_on': ['x'], 'depends': ['z']})


def test_symbol_override_uses_plain_yaml_data(tmp_path):
    card_fpath = tmp_path / 'card.yaml'
    card_fpath.write_text("""
claim:
  python: assert True
symbols:
  models:
    sweep: [old]
  label:
    value: old
""")
    card = EvaluationCard(card_fpath, tmp_path / 'results', validate='off')

    card.replace("models: [alpha, beta]\nlabel: 'a:b=c'")

    assert type(card.symbols['models']['sweep']) is list
    assert type(card.symbols['label']['value']) is str
    assert card.symbols['models']['sweep'] == ['alpha', 'beta']
    assert card.symbols['label']['value'] == 'a:b=c'
    yaml.safe_dump(card.original_card)
