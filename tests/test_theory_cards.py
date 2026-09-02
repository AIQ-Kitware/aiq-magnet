"""Theory cards resolve static links and compute premise coverage."""
import json
import textwrap
from importlib.resources import files
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from magnet.schema import TheorySchema

SIMPLE_CARDS = [
    ('coin_flip', 'tests', 'Examples.CoinFlip.Binomial'),
]

EXAMPLES = files('magnet') / 'examples' / 'theory_links'


def _run(example, output_path):
    """Evaluate a demo card and return its status and run directory.

    The theory_links examples are kwdagger recipes; the ones named by file are
    legacy cards. Both write `theory.json` into the run directory, which is the
    point: the theory report is a property of the card, not of the executor.
    """
    if example.endswith('.yaml'):
        from magnet.evaluation import EvaluationCard
        card = EvaluationCard(files('magnet') / 'cards' / example, output_path)
        status = card.evaluate()
    else:
        from magnet.evaluation_new import NewEvaluationRecipe
        recipe = NewEvaluationRecipe(
            EXAMPLES / example / 'card.yaml', output_path
        )
        status = recipe.evaluate(backend='serial').result
    # The new evaluator also puts its shared kwdagger store here.
    run_dpath = next(
        path for path in Path(output_path).iterdir()
        if path.is_dir() and path.name != '_kwdagger'
    )
    return status, run_dpath


@pytest.mark.parametrize('example,relation,ref', SIMPLE_CARDS)
def test_demo_cards_write_versioned_statement_links(example, relation, ref, tmp_path):
    status, run_dpath = _run(example, tmp_path / 'runs')
    assert status == 'VERIFIED'

    report = json.loads((run_dpath / 'theory.json').read_text())
    assert report['schema_version'] == 1
    assert [link['relation'] for link in report['statement_links']] == [relation]
    assert report['statement_links'][0]['ref'] == ref
    assert report['premise_links'] == []
    assert report['premise_coverage'] == []
    assert [entry['id'] for entry in report['entries']] == [ref]
    assert report['entries'][0]['statement']
    assert 'unresolved' not in report


def test_monte_carlo_example_demonstrates_static_premise_accounting():
    from magnet.theory.cards import report_from_card

    root = EXAMPLES / 'monte_carlo'
    card = yaml.safe_load((root / 'card.yaml').read_text())
    report = report_from_card(card, root).to_dict()
    assert report['schema_version'] == 1
    assert [
        (link['relation'], link['ref']) for link in report['statement_links']
    ] == [
        ('approximates', 'Examples.Circle.MonteCarloConsistency'),
        ('approximates', 'Examples.Circle.AreaRatio'),
    ]

    premise_links = {
        link['ref']: link for link in report['premise_links']
    }
    assert premise_links[
        'Examples.Circle.MonteCarloConsistency::hindicator'
    ]['relation'] == 'satisfies'
    assert premise_links[
        'Examples.Circle.MonteCarloConsistency::hmeas'
    ]['relation'] == 'assumes'
    assert premise_links[
        'Examples.Circle.MonteCarloConsistency::huniform'
    ]['relation'] == 'substitutes'

    assert len(report['premise_coverage']) == 1
    coverage = report['premise_coverage'][0]
    assert coverage['ref'] == 'Examples.Circle.MonteCarloConsistency'
    assert coverage['premise_count'] == 4
    assert coverage['accounted_count'] == 3
    assert coverage['complete'] is False
    assert coverage['unaccounted'] == ['hiid']

    entries = {entry['id']: entry for entry in report['entries']}
    sampling = entries['Examples.Circle.MonteCarloConsistency']
    assert sampling['declaration'] == (
        'MagnetExamples.Circle.monteCarloEstimator_consistent'
    )
    assert [premise['id'] for premise in sampling['premises']] == [
        'hindicator', 'hmeas', 'hiid', 'huniform'
    ]


def test_source_paths_are_portable_and_formalization_is_structured():
    from magnet.theory.cards import report_from_card

    root = EXAMPLES / 'coin_flip'
    card = yaml.safe_load((root / 'card.yaml').read_text())
    report = report_from_card(card, root).to_dict()
    link = report['statement_links'][0]
    assert link['qualname'] == 'enumerated_head_counts'
    assert link['file'] == 'experiment.py'
    assert not link['file'].startswith('/')
    entry = report['entries'][0]
    assert entry['formalization'] == {'system': 'lean4'}
    assert entry['source_path'] == 'CoinFlip.lean'


def test_fibonacci_performance_example_separates_question_from_explanation(tmp_path):
    from magnet.theory.cards import report_from_card

    root = EXAMPLES / 'fibonacci_performance'
    card = yaml.safe_load((root / 'card.yaml').read_text())
    report = report_from_card(card, root).to_dict()
    assert [
        (link['relation'], link['ref']) for link in report['statement_links']
    ] == [
        ('motivates', 'Examples.FibonacciPerformance.Why'),
        ('approximates', 'Examples.FibonacciPerformance.RecursiveCallGapAt28'),
    ]
    assert report['premise_links'] == []
    assert report['premise_coverage'] == []

    entries = {entry['id']: entry for entry in report['entries']}
    question = entries['Examples.FibonacciPerformance.Why']
    assert question['kind'] == 'question'
    assert 'formalization' not in question
    assert 'source_path' not in question

    explanation = entries['Examples.FibonacciPerformance.RecursiveCallGapAt28']
    assert explanation['kind'] == 'theorem'
    assert explanation['formalization'] == {'system': 'lean4'}
    assert explanation['source_path'] == 'FibonacciCost.lean'
    assert explanation['declaration'] == (
        'MagnetExamples.FibonacciPerformance.recursiveCalls_28_costGap'
    )

    status, run_dpath = _run('fibonacci_performance', tmp_path / 'runs')
    assert status == 'VERIFIED'
    written = json.loads((run_dpath / 'theory.json').read_text())
    assert written['statement_links'] == report['statement_links']


def test_a_card_without_a_theory_block_writes_no_artifact(tmp_path):
    _, run_dpath = _run('simple.yaml', tmp_path / 'runs')
    assert not (run_dpath / 'theory.json').exists()


def test_broken_theory_fails_before_the_run_directory_is_created(tmp_path):
    (tmp_path / 'demo.py').write_text(
        textwrap.dedent(
            '''
            import magnet.theory as theory

            @theory.tests('Nope.Missing')
            def experiment():
                pass
            '''
        )
    )
    card_data = textwrap.dedent(
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
          empirical_sources: [demo.py]
          entries: []
        '''
    )
    (tmp_path / 'card.yaml').write_text(card_data)
    output_path = tmp_path / 'runs'
    from magnet.evaluation import EvaluationCard
    card = EvaluationCard(tmp_path / 'card.yaml', output_path, validate='off')
    with pytest.raises(ValueError, match='Nope.Missing'):
        card.evaluate()
    assert not output_path.exists()


def test_card_and_source_statement_links_share_one_report(tmp_path):
    (tmp_path / 'demo.py').write_text(
        textwrap.dedent(
            '''
            import magnet.theory as theory

            @theory.approximates('From.File', note='finite implementation proxy')
            def one():
                pass
            '''
        )
    )
    (tmp_path / 'shared.yaml').write_text(
        textwrap.dedent(
            '''
            schema_version: 1
            entries:
              - id: From.File
                kind: theorem
                statement: shared theorem
            '''
        )
    )
    card = {
        'theory': {
            'links': [
                {
                    'relation': 'tests',
                    'ref': 'From.Card',
                    'note': 'the card claim directly evaluates this proposition',
                }
            ],
            'empirical_sources': ['demo.py'],
            'indexes': ['shared.yaml'],
            'entries': [
                {'id': 'From.Card', 'kind': 'definition', 'statement': 'finite claim'}
            ],
        }
    }
    from magnet.theory.cards import report_from_card

    report = report_from_card(card, tmp_path).to_dict()
    assert [entry['id'] for entry in report['entries']] == ['From.Card', 'From.File']
    assert [link['relation'] for link in report['statement_links']] == [
        'tests', 'approximates'
    ]
    card_link, source_link = report['statement_links']
    assert 'file' not in card_link
    assert source_link['file'] == 'demo.py'


def test_static_premise_coverage_is_computed_from_index_and_source(tmp_path):
    (tmp_path / 'experiment.py').write_text(
        textwrap.dedent(
            '''
            import magnet.theory as theory

            @theory.tests('Example.Stability')
            @theory.satisfies(
                'Example.Stability::hbounded',
                note='input validation establishes the bounded domain')
            def evaluate():
                with theory.assumes(
                        'Example.Stability::hiid',
                        note='the sampling procedure is treated as IID'):
                    return True
            '''
        )
    )
    (tmp_path / 'theory.yaml').write_text(
        textwrap.dedent(
            '''
            schema_version: 1
            formalization:
              system: lean4
              repository: https://example.invalid/formalization.git
              revision: synthetic-revision
            entries:
              - id: Example.Stability
                kind: theorem
                declaration: Example.stability
                source_path: Example/Stability.lean
                statement: synthetic stability theorem
                premises:
                  - id: hbounded
                    type: Bounded xs
                  - id: hiid
                    type: IID samples
                  - id: hunique
                    type: Unique optimum
            '''
        )
    )
    card = {
        'theory': {
            'empirical_sources': ['experiment.py'],
            'indexes': ['theory.yaml'],
        }
    }
    from magnet.theory.cards import report_from_card

    report = report_from_card(card, tmp_path).to_dict()
    assert [link['relation'] for link in report['statement_links']] == ['tests']
    assert [link['relation'] for link in report['premise_links']] == [
        'satisfies', 'assumes'
    ]
    coverage = report['premise_coverage'][0]
    assert coverage['ref'] == 'Example.Stability'
    assert coverage['premise_count'] == 3
    assert coverage['accounted_count'] == 2
    assert coverage['complete'] is False
    assert coverage['unaccounted'] == ['hunique']
    by_id = {premise['id']: premise for premise in coverage['premises']}
    assert by_id['hbounded']['links'][0]['relation'] == 'satisfies'
    assert by_id['hiid']['links'][0]['relation'] == 'assumes'
    assert by_id['hunique']['accounted'] is False


def test_unknown_premise_is_a_static_resolution_error(tmp_path):
    (tmp_path / 'experiment.py').write_text(
        textwrap.dedent(
            '''
            import magnet.theory as theory

            @theory.tests('Example.Stability')
            @theory.assumes('Example.Stability::hmissing')
            def evaluate():
                pass
            '''
        )
    )
    card = {
        'theory': {
            'empirical_sources': ['experiment.py'],
            'entries': [
                {
                    'id': 'Example.Stability',
                    'kind': 'theorem',
                    'statement': 'demo',
                    'premises': [{'id': 'hexists'}],
                }
            ],
        }
    }
    from magnet.theory.cards import report_from_card

    with pytest.raises(ValueError, match='hmissing'):
        report_from_card(card, tmp_path)


def test_card_level_premise_links_are_rejected():
    with pytest.raises(ValidationError, match='cannot target premises'):
        TheorySchema.model_validate(
            {
                'links': [
                    {'relation': 'approximates', 'ref': 'Example.Stability::hiid'}
                ]
            }
        )


def test_theory_schema_uses_empirical_sources_and_rejects_old_name():
    parsed = TheorySchema.model_validate({'empirical_sources': ['experiment.py']})
    assert parsed.empirical_sources == ['experiment.py']
    with pytest.raises(ValidationError, match='sources'):
        TheorySchema.model_validate({'sources': ['experiment.py']})
    with pytest.raises(ValidationError, match='ledger'):
        TheorySchema.model_validate({'ledger': 'old-theory-ledger.json'})


def test_motivates_does_not_create_premise_coverage_obligation(tmp_path):
    (tmp_path / 'experiment.py').write_text(
        textwrap.dedent(
            '''
            import magnet.theory as theory

            @theory.motivates('Example.Question')
            @theory.assumes('Example.Question::hcontext')
            def observe():
                pass
            '''
        )
    )
    card = {
        'theory': {
            'empirical_sources': ['experiment.py'],
            'entries': [
                {
                    'id': 'Example.Question',
                    'kind': 'question',
                    'statement': 'why does the phenomenon occur?',
                    'premises': [{'id': 'hcontext'}],
                }
            ],
        }
    }
    from magnet.theory.cards import report_from_card

    report = report_from_card(card, tmp_path).to_dict()
    assert report['premise_coverage'] == []
    assert report['unattached_premise_links'] == []
