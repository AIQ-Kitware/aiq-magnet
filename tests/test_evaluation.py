import json

from magnet.evaluation import EvaluationCard


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
