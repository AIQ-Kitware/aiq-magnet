"""Tests for the replacement kwdagger-native evaluation API."""
import sys
import textwrap

import pytest
import ubelt as ub
import yaml

import magnet.evaluation_new as evaluation_new
from magnet.evaluation import main as legacy_main
from magnet.evaluation_new import (
    NewEvaluationCLI,
    NewEvaluationRecipe,
    NewEvaluationResultCard,
)


SCRIPT = """
import json, pathlib, sys
args = dict(a.lstrip('-').split('=', 1) for a in sys.argv[1:])
out = pathlib.Path(args['results_fpath'])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'result': {'metrics': {'score': float(args['seed']) / 10}}}))
"""


@pytest.fixture
def kwdagger_recipe_fpath(tmp_path):
    dpath = ub.Path(tmp_path)
    script = dpath / 'emit.py'
    script.write_text(textwrap.dedent(SCRIPT))
    fpath = dpath / 'recipe.yaml'
    fpath.write_text(yaml.safe_dump({
        'title': 'new evaluator probe',
        'description': 'new evaluator probe',
        'version': '1.0',
        'organizations': ['Kitware'],
        'submitter': {'name': 't', 'email': 't@example.com'},
        'links': [],
        'tags': ['test'],
        'claim': {'python': 'assert metrics.emit.score < 100'},
        'kwdagger': {
            'result_node': 'emit',
            'pipeline': {'nodes': {'emit': {
                'executable': f'{sys.executable} {script}',
                'algo_params': {'seed': 1},
                'out_paths': {'results_fpath': 'results.json'},
                'primary_out_key': 'results_fpath',
            }}},
            'matrix': {'emit.seed': [1, 2]},
        },
    }, sort_keys=False))
    return fpath


def test_new_cli_does_not_expose_legacy_execution_options():
    keys = set(NewEvaluationCLI().keys())
    assert {
        'backend', 'tmux_workers', 'skip_existing', 'cache', 'max_configs'
    } <= keys
    assert not {
        'override', 'jobs', 'parallel_backend', 'queue_backend', 'workers'
    } & keys


def test_new_api_uses_recipe_and_result_names():
    assert 'main' not in vars(evaluation_new)
    assert not hasattr(evaluation_new, 'NewEvaluationCard')
    assert not hasattr(evaluation_new, 'NewEvaluationTask')
    assert not hasattr(evaluation_new, 'Results')
    assert callable(NewEvaluationCLI.main)


def test_new_cli_passes_execution_config_directly(
        kwdagger_recipe_fpath, tmp_path, monkeypatch):
    monkeypatch.setenv('MAGNET_QUEUE_BACKEND', 'definitely-not-a-backend')
    monkeypatch.setenv('MAGNET_TMUX_WORKERS', '999')

    output_path = ub.Path(tmp_path) / 'out'
    result_card = NewEvaluationCLI.main(argv=[
        str(kwdagger_recipe_fpath),
        '--output_path', str(output_path),
        '--params', 'matrix: {emit.seed: [7]}',
        '--backend', 'serial',
        '--skip_existing=0',
        '--cache=1',
        '--max_configs=1',
    ])

    assert isinstance(result_card, NewEvaluationResultCard)
    artifacts = sorted(
        (output_path / '_kwdagger' / 'emit').glob('*/results.json')
    )
    assert len(artifacts) == 1


def test_new_recipe_rejects_legacy_symbol_sweeps(
        kwdagger_recipe_fpath, tmp_path):
    data = yaml.safe_load(ub.Path(kwdagger_recipe_fpath).read_text())
    data['symbols'] = {'legacy_axis': {'sweep': [1, 2]}}
    ub.Path(kwdagger_recipe_fpath).write_text(
        yaml.safe_dump(data, sort_keys=False)
    )

    with pytest.raises(ValueError, match='symbol sweeps'):
        NewEvaluationRecipe(
            kwdagger_recipe_fpath,
            ub.Path(tmp_path) / 'out',
            validate='off',
        )


def test_new_recipe_rejects_legacy_pipeline(tmp_path):
    fpath = ub.Path(tmp_path) / 'legacy.yaml'
    fpath.write_text(yaml.safe_dump({
        'claim': {'python': 'assert True'},
        'pipeline': {
            'old_node': {
                'executable': 'echo',
                'out_paths': {'results_fpath': 'results.json'},
                'primary_out_key': 'results_fpath',
            },
        },
    }, sort_keys=False))

    with pytest.raises(ValueError, match='requires a kwdagger recipe'):
        NewEvaluationRecipe(
            fpath,
            ub.Path(tmp_path) / 'out',
            validate='off',
        )


def test_legacy_evaluator_points_kwdagger_cards_to_evaluate_new(
        kwdagger_recipe_fpath, tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        legacy_main(argv=[
            str(kwdagger_recipe_fpath),
            '--output_path', str(ub.Path(tmp_path) / 'out'),
            '--validate', 'off',
        ])
    message = str(exc_info.value)
    assert 'declares a `kwdagger:` pipeline' in message
    assert '`magnet evaluate`' in message
    assert 'magnet evaluate_new' in message


def _scores(result_card):
    return sorted(
        cell.evidence_row['metrics.emit.score']
        for cell in result_card.cell_results
    )


def test_evidence_accumulates_across_independent_schedule_requests(
        kwdagger_recipe_fpath, tmp_path):
    """The current finite request does not define the available evidence set."""
    output_path = ub.Path(tmp_path) / 'out'

    first = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    first.apply_params('matrix: {emit.seed: [1]}')
    first_result = first.evaluate(backend='serial')
    assert _scores(first_result) == [0.1]

    second = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    second.apply_params('matrix: {emit.seed: [2]}')
    second_result = second.evaluate(backend='serial')

    # Only seed 2 was requested by the second schedule, but seed 1 remains
    # available evidence in the shared KWDagger result store.
    assert _scores(second_result) == [0.1, 0.2]
    assert second_result.requested_work['processes'] == 1


def test_identical_recipe_invocations_keep_distinct_magnet_run_records(
        kwdagger_recipe_fpath, tmp_path):
    output_path = ub.Path(tmp_path) / 'out'
    first = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    second = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)

    assert first._recipe_hash == second._recipe_hash
    assert first._run_hash != second._run_hash


def test_failed_current_attempt_does_not_become_claim_evidence(
        kwdagger_recipe_fpath, tmp_path, monkeypatch):
    """Execution failure is reported separately from claim truth."""

    class FakeProcessor:
        def __init__(self, spec, root_dpath):
            self.root_dpath = ub.Path(root_dpath).ensuredir()

        def schedule(self, **schedule_options):
            pass

        def inspect_requested_runs(self):
            return [{
                'process_id': 'new-failed-process',
                'node': 'emit',
                'schedule_status': 'new_submission',
                'attempt_status': 'failed',
                'returncode': 3,
                'output_available': False,
                'enabled': True,
            }]

        def load_available_result_rows(self):
            artifact = self.root_dpath / 'emit' / 'old-process' / 'results.json'
            return [{
                'key': 'old-process',
                'artifact': str(artifact),
                'row': {'metrics.emit.score': 0.1},
            }]

    monkeypatch.setattr(evaluation_new, 'KWDaggerProcessor', FakeProcessor)

    recipe = NewEvaluationRecipe(
        kwdagger_recipe_fpath, ub.Path(tmp_path) / 'out'
    )
    result_card = recipe.evaluate(backend='serial')

    assert result_card.result == 'VERIFIED'
    assert _scores(result_card) == [0.1]
    assert result_card.requested_work['attempt_status'] == {'failed': 1}
    assert result_card.requested_work['outputs_available'] == 0
