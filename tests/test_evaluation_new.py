"""Tests for the replacement kwdagger-native evaluation API."""
import json
import sys
import textwrap

import pytest
import ubelt as ub
import yaml
from pydantic import ValidationError

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
    NewEvaluationCLI.main(argv=[
        str(kwdagger_recipe_fpath),
        '--output_path', str(output_path),
        '--params', 'matrix: {emit.seed: [7]}',
        '--backend', 'serial',
        '--skip_existing=0',
        '--cache=1',
        '--max_configs=1',
    ])

    artifacts = sorted(
        (output_path / '_kwdagger' / 'emit').glob('*/results.json')
    )
    assert len(artifacts) == 1


def test_cli_main_returns_an_exit_status_not_a_result_card(
        kwdagger_recipe_fpath, tmp_path):
    """
    The console script is `sys.exit(MagnetCLI.main())`. `sys.exit` of a
    non-integer prints that object and exits 1, so a `main` that returned the
    result card made every successful run dump the card and report failure.
    """
    argv = [
        'evaluate_new',
        str(kwdagger_recipe_fpath),
        '--output_path', str(ub.Path(tmp_path) / 'out'),
        '--backend', 'serial',
        '--max_configs=1',
    ]
    from magnet.cli.main import MagnetCLI

    assert NewEvaluationCLI.main(argv=argv[1:]) is None
    status = MagnetCLI.main(argv=argv)
    assert isinstance(status, int) and status == 0

    # The card is still available from the library API and on disk.
    recipe = NewEvaluationRecipe(
        kwdagger_recipe_fpath, ub.Path(tmp_path) / 'lib', validate='off'
    )
    assert isinstance(
        recipe.evaluate(backend='serial', max_configs=1),
        NewEvaluationResultCard,
    )


def test_declared_symbol_metadata_reaches_the_dashboard_contract(
        kwdagger_recipe_fpath, tmp_path):
    """
    A kwdagger recipe declares metric symbols without computing them: the
    pipeline produced the value, and the declaration supplies the name,
    display flag, and metric definition the dashboard and the BAA metric
    report need. Each symbol is named for the evidence column it describes.
    """
    data = yaml.safe_load(ub.Path(kwdagger_recipe_fpath).read_text())
    data['symbols'] = {
        'metrics.emit.score': {
            'metadata': {
                'display_name': 'Average Exact Match',
                'display': True,
                'define_metric': {
                    'objective': 'maximize',
                    'aggregation_strategy': {'type': 'mean'},
                },
            },
        },
        'params.emit.seed': {
            'metadata': {'display_name': 'Seed', 'display': True},
        },
    }
    ub.Path(kwdagger_recipe_fpath).write_text(
        yaml.safe_dump(data, sort_keys=False)
    )

    output_path = ub.Path(tmp_path) / 'out'
    recipe = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    result_card = recipe.evaluate(backend='serial')

    written = sorted(output_path.glob('*/symbol_metadata.json'))
    assert len(written) == 1
    assert json.loads(written[0].read_text()) == {
        'metrics.emit.score': {
            'display_name': 'Average Exact Match',
            'display': True,
            'define_metric': {
                'objective': 'maximize',
                'aggregation_strategy': {'type': 'mean'},
            },
        },
        'params.emit.seed': {'display_name': 'Seed', 'display': True},
    }

    # `params.emit.seed` carries display metadata only, so it names no metric.
    # `metrics.emit.score` is filled from that column on each evidence row and
    # reduced across them under its declared display name.
    assert set(result_card.metrics) == {'Average Exact Match'}
    assert result_card.metrics['Average Exact Match'] == pytest.approx(0.15)
    for cell in result_card.cell_results:
        assert (cell.symbols['metrics.emit.score']
                == cell.evidence_row['metrics.emit.score'])


def test_a_qualified_symbol_names_exactly_one_evidence_column():
    """
    `seed` alone matches two columns holding different values. Naming the
    column says which is meant, and the provenance flag is reachable only by
    asking for it outright.
    """
    row = {
        'metrics.emit.score': 0.1,
        'params.emit.seed': 7,
        'specified.params.emit.seed': 1,
    }
    filled, measured = evaluation_new._fill_declared_symbols(
        {'params.emit.seed': {'metadata': {'display_name': 'Seed'}}}, row
    )
    assert measured == {'params.emit.seed'}
    assert filled['params.emit.seed']['value'] == 7

    filled, _ = evaluation_new._fill_declared_symbols(
        {'specified.params.emit.seed': {'type': 'int'}}, row
    )
    assert filled['specified.params.emit.seed']['value'] == 1


def test_unqualified_symbol_names_still_match_the_last_segment():
    """Legacy cards name a bare symbol; that keeps resolving."""
    row = {'metrics.emit.score': 0.1, 'params.emit.seed': 7}
    filled, measured = evaluation_new._fill_declared_symbols(
        {'score': {'type': 'float'}}, row
    )
    assert measured == {'score'}
    assert filled['score']['value'] == 0.1


def test_declared_symbols_ignore_kwdagger_provenance_flags():
    """
    `specified.params.<node>.<param>` is kwdagger's "was requested" flag and
    is always 1. Matching a bare symbol on the last dotted segment must not
    let that flag stand in for the parameter's value.
    """
    row = {
        'specified.params.emit.seed': 1,
        'params.emit.seed': 7,
        'metrics.emit.score': 0.1,
    }
    filled, measured = evaluation_new._fill_declared_symbols(
        {'seed': {'type': 'int'}}, row
    )
    assert measured == {'seed'}
    assert filled['seed']['value'] == 7


def test_a_declaration_only_symbol_needs_no_resolution(tmp_path):
    """
    A symbol that only labels a pipeline output has nothing to resolve, so
    the schema must not demand a `value`, `sweep`, or `python` from it --
    with or without a metric definition.
    """
    from magnet.schema import SymbolSchema

    SymbolSchema.model_validate(
        {'metadata': {'display_name': 'HELM Data Path', 'display': True}}
    )
    with pytest.raises(ValidationError):
        SymbolSchema.model_validate({})


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


def _set_evidence_scope(recipe_fpath, scope):
    data = yaml.safe_load(ub.Path(recipe_fpath).read_text())
    data['evidence'] = {'scope': scope}
    ub.Path(recipe_fpath).write_text(yaml.safe_dump(data, sort_keys=False))


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


def test_requested_scope_filters_accumulated_evidence(
        kwdagger_recipe_fpath, tmp_path):
    output_path = ub.Path(tmp_path) / 'out'

    first = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    first.apply_params('matrix: {emit.seed: [1]}')
    first.evaluate(backend='serial')

    _set_evidence_scope(kwdagger_recipe_fpath, 'requested')
    second = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    second.apply_params('matrix: {emit.seed: [2]}')
    second_result = second.evaluate(backend='serial')

    assert second_result.evidence_scope == 'requested'
    assert second_result.evidence_discovered == 2
    assert _scores(second_result) == [0.2]
    assert second_result.as_record()['evidence'] == {
        'scope': 'requested',
        'available': 1,
        'discovered': 2,
    }


def test_requested_scope_includes_requested_cached_output(
        kwdagger_recipe_fpath, tmp_path):
    output_path = ub.Path(tmp_path) / 'out'

    first = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    first.apply_params('matrix: {emit.seed: [1]}')
    first.evaluate(backend='serial')

    _set_evidence_scope(kwdagger_recipe_fpath, 'requested')
    second = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    second.apply_params('matrix: {emit.seed: [1]}')
    second_result = second.evaluate(backend='serial', skip_existing=True)

    assert _scores(second_result) == [0.1]
    assert second_result.requested_work['schedule_status']['skipped'] >= 1


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
            self.result_node = 'emit'

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



def _parse_with_existing_dashboard_contract(root_path):
    """Small dependency-free mirror of eval-card-viz's local upload parser."""
    parsed = []
    for run_dpath in ub.Path(root_path).iterdir():
        if not run_dpath.is_dir():
            continue
        results_dpath = run_dpath / 'results'
        if not results_dpath.exists():
            continue

        runs = []
        for result_dpath in results_dpath.iterdir():
            verdict_fpath = result_dpath / 'verdict.json'
            if verdict_fpath.is_file():
                record = yaml.safe_load(verdict_fpath.read_text())
                record['id'] = result_dpath.name
                runs.append(record)

        card_fpath = run_dpath / 'card.yaml'
        verdict_fpath = run_dpath / 'verdict.json'
        if not card_fpath.is_file() or not verdict_fpath.is_file():
            continue
        card = yaml.safe_load(card_fpath.read_text())
        verdict = yaml.safe_load(verdict_fpath.read_text())
        parsed.append({
            'card': card,
            'result': verdict['result'],
            'claim_aggregation_strategy': (
                verdict['claim_aggregation_strategy']
            ),
            'runs': runs,
        })
    return parsed


def test_new_run_bundle_matches_dashboard_contract(
        kwdagger_recipe_fpath, tmp_path):
    output_path = ub.Path(tmp_path) / 'out'
    recipe = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    recipe.apply_params('matrix: {emit.seed: [1]}')
    result_card = recipe.evaluate(backend='serial')

    run_dpath = output_path / recipe._run_hash
    assert (run_dpath / 'card.yaml').is_file()
    assert (run_dpath / 'log').is_file()
    assert (run_dpath / 'results').is_dir()
    assert (run_dpath / 'verdict.json').is_file()

    aggregate = yaml.safe_load((run_dpath / 'verdict.json').read_text())
    assert aggregate['result'] == result_card.result
    assert 'claim_aggregation_strategy' in aggregate
    assert aggregate['evidence']['scope'] == 'all'

    result_fpaths = list((run_dpath / 'results').glob('*/verdict.json'))
    assert len(result_fpaths) == 1
    cell = yaml.safe_load(result_fpaths[0].read_text())
    # These four fields are the existing dashboard's per-run read contract.
    assert {'status', 'output', 'symbols', 'timestamp'} <= set(cell)
    assert cell['symbols'] == {'metrics.emit.score': 0.1}
    assert cell['evidence']['metrics.emit.score'] == 0.1

    # The dashboard's local upload path receives the output root, skips the
    # shared _kwdagger directory, and discovers the MAGNET run directory.
    dashboard_cards = _parse_with_existing_dashboard_contract(output_path)
    assert len(dashboard_cards) == 1
    assert dashboard_cards[0]['result'] == result_card.result
    assert dashboard_cards[0]['runs'][0]['symbols'] == {
        'metrics.emit.score': 0.1
    }


def test_empty_evidence_still_writes_dashboard_run_bundle(
        kwdagger_recipe_fpath, tmp_path, monkeypatch):
    class EmptyProcessor:
        def __init__(self, spec, root_dpath):
            self.root_dpath = ub.Path(root_dpath).ensuredir()
            self.result_node = 'emit'

        def schedule(self, **schedule_options):
            pass

        def inspect_requested_runs(self):
            return []

        def load_available_result_rows(self):
            return []

    monkeypatch.setattr(evaluation_new, 'KWDaggerProcessor', EmptyProcessor)

    output_path = ub.Path(tmp_path) / 'out'
    recipe = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    result_card = recipe.evaluate(backend='serial')
    run_dpath = output_path / recipe._run_hash

    assert result_card.result == 'INCONCLUSIVE'
    assert (run_dpath / 'results').is_dir()
    aggregate = yaml.safe_load((run_dpath / 'verdict.json').read_text())
    assert aggregate['result'] == 'INCONCLUSIVE'
    assert aggregate['evidence'] == {
        'scope': 'all',
        'available': 0,
        'discovered': 0,
    }


def test_dry_run_compiles_the_campaign_without_running_it(
        kwdagger_recipe_fpath, tmp_path):
    """`--dry_run` schedules KWDagger with run=0.

    The whole matrix still compiles, so the request can be reported in full --
    which is the point of asking. Nothing is submitted.
    """
    output_path = ub.Path(tmp_path) / 'out'
    recipe = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    result = recipe.evaluate(backend='serial', dry_run=True)

    requested = result.requested_work
    assert requested['processes'] == 2
    assert requested['attempt_status'] == {'not_started': 2}
    assert requested['outputs_available'] == 0

    assert not sorted((output_path / '_kwdagger' / 'emit').glob('*/results.json'))


def test_a_real_run_of_the_same_recipe_does_produce_artifacts(
        kwdagger_recipe_fpath, tmp_path):
    """The contrast that makes the dry run mean something."""
    output_path = ub.Path(tmp_path) / 'out'
    recipe = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    result = recipe.evaluate(backend='serial')

    assert result.requested_work['outputs_available'] == 2
    assert len(
        sorted((output_path / '_kwdagger' / 'emit').glob('*/results.json'))
    ) == 2


def test_dry_run_still_judges_evidence_that_already_exists(
        kwdagger_recipe_fpath, tmp_path):
    """A dry run does not pretend the result store is empty.

    Scheduling and evidence discovery are separate: not submitting work says
    nothing about what has already been computed.
    """
    output_path = ub.Path(tmp_path) / 'out'
    NewEvaluationRecipe(kwdagger_recipe_fpath, output_path).evaluate(
        backend='serial')

    second = NewEvaluationRecipe(kwdagger_recipe_fpath, output_path)
    result = second.evaluate(backend='serial', dry_run=True)
    assert len(result.cell_results) == 2
    assert result.result == 'VERIFIED'


def test_dry_run_is_off_by_default(kwdagger_recipe_fpath, tmp_path):
    assert NewEvaluationCLI()['dry_run'] is False
