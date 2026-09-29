"""aiq-magnet-evals integration (aiq-magnet-evals docs/planning/aiq-magnet-integration-plan.md).

End-to-end tests run real engines through ``magnet_evals`` workers; see
``aiq_evals_support`` for the environment variables that select them. With
``MAGNET_REQUIRE_AIQ_EVALS=1`` a missing prerequisite fails instead of
skipping, which is how the dedicated CI job runs this file.
"""
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import ubelt as ub
from aiq_evals_support import (
    HELM_ALGO,
    HELM_PYTHON,
    HELM_SELECT,
    INSPECT_PYTHON,
    OLMO_PYTHON,
    evaluate,
    evaluation_node,
    evaluations,
    latest_run_dir,
    needs,
    needs_helm,
    needs_inspect,
    needs_olmo,
    needs_repo,
    recipe_rows,
    repo,
    repo_on_worker_path,
    require_magnet_evals,
    run_verdicts,
    store_attempts,
    write_recipe,
)

magnet_evals = require_magnet_evals()

from magnet.backends.aiq_evals import (  # noqa: E402
    EvaluationNode,
    preflight_scope,
)
from magnet.backends.aiq_evals import pipeline as node_mod  # noqa: E402
from magnet.backends.aiq_evals.projection import (  # noqa: E402
    evidence_view,
    flat_metrics,
    select_metric,
)


def helm_recipe(dpath, *, select=HELM_SELECT, algo=None, **kwargs):
    algo = {**HELM_ALGO, **(algo or {}), 'select': select}
    return write_recipe(dpath, {'evaluate': evaluation_node(algo)}, **kwargs)


# --- M5 + dashboards: one HELM evaluation -> one claim row ----------------------

@needs_helm
def test_one_helm_evaluation_becomes_one_claim_row(tmp_path):
    recipe, card = evaluate(helm_recipe(tmp_path), tmp_path / 'out')
    rows = recipe_rows(recipe)
    assert len(rows) == 1
    row = rows[0]['row']
    assert row['metrics.evaluate.eligible'] is True
    assert row['metrics.evaluate.engine'] == 'helm'
    assert row['metrics.evaluate.selected.metric'] == 'exact_match'
    assert row['metrics.evaluate.score'] == row['metrics.evaluate.selected.value']
    assert len(row['metrics.evaluate.measurement_identity']) == 64
    assert card.result == 'VERIFIED'

    # The legacy dashboard contract (eval-card-viz upload): card.yaml, log,
    # results/*/verdict.json, verdict.json with concrete claim symbols.
    run_dir = latest_run_dir(tmp_path / 'out')
    for name in ('card.yaml', 'verdict.json', 'requested_runs.json'):
        assert (run_dir / name).is_file(), name
    assert any(run_dir.glob('*log*'))
    (verdict,) = run_verdicts(run_dir)
    assert verdict['status'] == 'VERIFIED'
    assert verdict['consumed'] == ['metrics.evaluate.score']
    assert verdict['symbols']['metrics.evaluate.score'] == row['metrics.evaluate.score']
    top = json.loads((run_dir / 'verdict.json').read_text())
    assert top['result'] == 'VERIFIED' and top['evidence']['available'] == 1


# --- M6: cardinality acceptance experiment -----------------------------------
# Representative native multi-result fixtures committed in aiq-magnet-evals
# (tests/fixtures/*-native/multi), imported through real engine workers, loaded
# by real kwdagger build_tables + KWDaggerProcessor, and judged through
# ClaimResultNamespace by the real evaluate_new flow.

def run_cardinality(tmp_path, *, engine_params, worker, selects, claim):
    fpath = write_recipe(
        tmp_path, {'evaluate': evaluation_node(engine_params, worker=worker)}, claim=claim,
        matrix={'evaluate.select': [json.dumps(s, sort_keys=True) for s in selects]}, name='cardinality',
    )
    recipe, _ = evaluate(fpath, tmp_path / 'out')
    rows = recipe_rows(recipe)
    verdicts = run_verdicts(latest_run_dir(tmp_path / 'out'))
    by_select = {json.dumps(json.loads(r['row']['params.evaluate.select']), sort_keys=True): r for r in rows}
    return rows, verdicts, by_select


@needs_repo
@needs_inspect
def test_cardinality_inspect_multi_log_epochs(tmp_path, monkeypatch):
    repo_on_worker_path(monkeypatch)
    fixture = repo() / 'tests' / 'fixtures' / 'inspect-native' / 'multi'
    engine_params = {
        'engine': 'inspect_ai',
        'task': str(repo() / 'tests' / 'native' / 'inspect_fixture.py'),
        'data_revision': 'fixture-v1',
        'models': [
            {'role': 'primary', 'model': 'local', 'provider': 'fixture', 'revision': 'local-v1'},
            {'role': 'grader', 'model': 'grader', 'provider': 'fixture', 'revision': 'local-v1'},
        ],
        'engine_options': {'registration_modules': ['tests.native.inspect_fixture'], 'eval_options': {'epochs': 2}},
        'import_source': str(fixture),
    }
    unique = {'task': 'role_task', 'scorer': 'match', 'metric': 'accuracy'}
    ambiguous = {'scorer': 'match', 'metric': 'accuracy'}   # three task logs match
    nothing = {'task': 'role_task', 'metric': 'f1'}
    rows, verdicts, by_select = run_cardinality(
        tmp_path, engine_params=engine_params, worker=INSPECT_PYTHON,
        selects=[unique, ambiguous, nothing], claim='assert metrics.evaluate.score == 1.0',
    )
    # One artifact -> one row -> one verdict, for three logs x two epochs each.
    assert len(rows) == len(verdicts) == 3
    assert len({r['artifact'] for r in rows}) == 3
    row = by_select[json.dumps(unique, sort_keys=True)]['row']
    assert row['metrics.evaluate.score'] == 1.0 and row['metrics.evaluate.candidates'] == 1
    assert row['metrics.evaluate.selected.task'] == 'role_task'
    # One import, reused by the other two projections (single-flight or not).
    assert sorted(r['row']['metrics.evaluate.action'] for r in rows) == ['imported', 'reused', 'reused']
    # Rejected selections expose no score: nothing is averaged across tasks.
    for select, count in ((ambiguous, 3), (nothing, 0)):
        rejected = by_select[json.dumps(select, sort_keys=True)]['row']
        score = rejected.get('metrics.evaluate.score')
        assert score is None or score != score
        assert rejected['metrics.evaluate.eligible'] is False
        assert rejected['metrics.evaluate.candidates'] == count
    statuses = sorted(v['status'] for v in verdicts)
    assert statuses.count('VERIFIED') == 1, statuses

    # Duplicate sample IDs across task logs and repeated epochs stay inside the
    # run; the claim-facing value is the selected task's native metric.
    run = magnet_evals.load_run(row['metrics.evaluate.run_path'])
    ids = [(s.task, s.sample_id) for s in run.result.samples if s.native.get('kind') == 'sample']
    assert len({sid for _, sid in ids}) < len({t for t, _ in ids}) * 2
    assert {s.epoch for s in run.result.samples if s.epoch} == {1, 2}
    native = [m.value for m in magnet_evals.outputs.select_metrics(run, task='role_task', scorer='match', metric='accuracy')]
    assert native == [row['metrics.evaluate.score']]


@needs_repo
@needs_olmo
def test_cardinality_olmo_suite_prefix_overlapping_tasks(tmp_path, monkeypatch):
    repo_on_worker_path(monkeypatch)
    engine_params = {
        'engine': 'olmo_eval',
        'task': 'aiq_p1_multi',
        'task_revision': 'fixture-v1',
        'data_revision': 'fixture-v1',
        'models': [{'role': 'primary', 'model': 'mock', 'provider': 'mock', 'revision': 'local-v1'}],
        'engine_options': {
            'upstream_revision': '73ade80e24f796af55caeb8fd7b75a7f3fd607fd',
            'task_modules': ['tests.native.olmo_fixture'],
        },
        'import_source': str(repo() / 'tests' / 'fixtures' / 'olmo-native' / 'multi'),
    }
    unique = {'task': 'aiq_p1_local', 'metric': 'contains_42'}
    ambiguous = {'metric': 'contains_42'}
    rows, verdicts, by_select = run_cardinality(
        tmp_path, engine_params=engine_params, worker=OLMO_PYTHON,
        selects=[unique, ambiguous], claim='assert metrics.evaluate.score == 1.0',
    )
    assert len(rows) == len(verdicts) == 2
    assert by_select[json.dumps(unique, sort_keys=True)]['row']['metrics.evaluate.selected.task'] == 'aiq_p1_local'
    assert by_select[json.dumps(ambiguous, sort_keys=True)]['row']['metrics.evaluate.candidates'] == 2
    assert sorted(v['status'] for v in verdicts).count('VERIFIED') == 1


# --- M3: identity, reuse, stale markers ---------------------------------------

@needs_helm
def test_reuse_selector_change_and_stale_marker(tmp_path):
    out = tmp_path / 'out'
    fpath = helm_recipe(tmp_path)
    recipe, _ = evaluate(fpath, out)
    assert len(evaluations(out)) == 1 and len(store_attempts(out)) == 1

    # Same request: kwdagger skips the node (it still validates as done).
    evaluate(fpath, out)
    assert len(evaluations(out)) == 1 and len(store_attempts(out)) == 1

    # A different evidence selector reruns the node, not the native evaluation.
    evaluate(helm_recipe(tmp_path, select={'metric': 'quasi_exact_match', 'group': 'test', 'score': None}), out)
    records = [json.loads(p.read_text()) for p in evaluations(out)]
    assert len(records) == 2
    assert sorted(e['action'] for e in records) == ['executed', 'reused']
    assert len({e['measurement_identity']['digest'] for e in records}) == 1
    assert len(store_attempts(out)) == 1

    # A stale marker cannot hide a broken run: tamper with the canonical run.
    run_path = Path(records[0]['run_path'])
    (run_path / 'native' / 'injected.txt').write_text('tamper\n')
    evaluate(fpath, out)
    assert len(store_attempts(out)) == 2
    assert len(recipe_rows(recipe)) == 2


@needs_helm
def test_non_reusable_identity_always_executes(tmp_path):
    out = tmp_path / 'out'
    fpath = helm_recipe(tmp_path, algo={'data_revision': None})
    evaluate(fpath, out)
    evaluate(fpath, out)
    records = [json.loads(p.read_text()) for p in evaluations(out)]
    assert len(records) == 2  # a nonce in the node id: never skipped as done
    assert all(e['preflight_identity'].startswith('unresolved-') for e in records)
    assert all(not e['measurement_identity']['reusable'] for e in records)


TASK_SOURCE = """
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate


@task
def edited_task():
    # revision marker: {marker}
    return Task(dataset=[Sample(input="Two plus two?", target="4")], solver=generate(), scorer=match())
"""


@needs_inspect
def test_task_code_change_reruns_the_node(tmp_path):
    # M3: kwdagger's node identity follows the resolved measurement, so editing
    # the task file (not the recipe) schedules a new native evaluation.
    task_file = tmp_path / 'edited_task.py'
    task_file.write_text(TASK_SOURCE.format(marker='v1'))
    algo = {
        'engine': 'inspect_ai',
        'task': str(task_file),
        'data_revision': 'example-v1',
        'models': [{'role': 'primary', 'model': 'local', 'provider': 'aiq_example', 'revision': 'local-v1'}],
        'engine_options': {'registration_modules': ['magnet_evals.examples.inspect_tasks']},
        'select': {'scorer': 'match', 'metric': 'accuracy'},
    }
    fpath = write_recipe(tmp_path, {'evaluate': evaluation_node(algo, worker=INSPECT_PYTHON)},
                         claim='assert metrics.evaluate.score == 1.0')
    out = tmp_path / 'out'
    evaluate(fpath, out)
    evaluate(fpath, out)
    assert len(evaluations(out)) == 1 and len(store_attempts(out)) == 1
    task_file.write_text(TASK_SOURCE.format(marker='v2'))
    _, card = evaluate(fpath, out)
    records = [json.loads(p.read_text()) for p in evaluations(out)]
    assert card.result == 'VERIFIED'
    assert len(records) == 2 and len(store_attempts(out)) == 2
    assert len({r['measurement_identity']['digest'] for r in records}) == 2
    assert all(r['action'] == 'executed' for r in records)


def test_dry_run_resolves_nothing(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('dry run must not resolve (would run task code)')

    monkeypatch.setattr(node_mod, 'run_preflight', forbidden)
    _, card = evaluate(helm_recipe(tmp_path), tmp_path / 'out', dry_run=True)
    assert card.result == 'NOT_EVALUATED'
    assert not (tmp_path / 'out' / '_kwdagger' / '_aiq_evals_store').exists()
    assert not evaluations(tmp_path / 'out')


def test_static_errors_surface_in_a_dry_run(tmp_path):
    with pytest.raises(ValueError, match='unknown evidence selector'):
        evaluate(helm_recipe(tmp_path, select={'metrik': 'exact_match'}), tmp_path / 'out', dry_run=True)
    fpath = helm_recipe(tmp_path, algo={'coverage_policy': 'most'})
    with pytest.raises(ValueError, match='coverage_policy'):
        evaluate(fpath, tmp_path / 'out2', dry_run=True)


def _node(**algo):
    base = {'engine': 'helm', 'task': 'simple_mcqa', 'models': [{'model': 'simple/model1', 'revision': 'r'}]}
    node = EvaluationNode(name='evaluate', algo_params={**base, **algo})
    node.configure({})
    return node


def test_measurement_identity_is_computed_never_configured(monkeypatch):
    # A recipe cannot supply it ...
    with pytest.raises(ValueError, match='computed by preflight'):
        _node(measurement_identity='f' * 64)
    with pytest.raises(ValueError, match='computed by preflight'):
        _node(import_identity='e' * 64)

    # ... and a value arriving through configuration (e.g. a matrix) is replaced.
    node = EvaluationNode(name='evaluate', algo_params={
        'engine': 'helm', 'task': 'simple_mcqa', 'models': [{'model': 'simple/model1', 'revision': 'r'}],
    })
    node.configure({'measurement_identity': 'f' * 64, 'import_identity': 'e' * 64})
    assert node.final_algo_config['measurement_identity'] == node_mod.DRY_RUN_IDENTITY
    assert node.final_algo_config['import_identity'] is None

    calls = []

    def fake_preflight(command, timeout=None):
        calls.append(command)
        return {'measurement_identity': {'digest': 'a' * 64, 'reusable': True, 'unknown_reasons': []},
                'import_identity': None}

    monkeypatch.setattr(node_mod, 'run_preflight', fake_preflight)
    with preflight_scope(True):
        assert node.final_algo_config['measurement_identity'] == 'a' * 64
        assert node.final_algo_config['measurement_identity'] == 'a' * 64
    assert len(calls) == 1  # memoized within one schedule
    with preflight_scope(True):
        node.final_algo_config
    assert len(calls) == 2  # every schedule re-resolves


def test_preflight_runs_through_the_node_container(monkeypatch):
    commands = []

    def fake_preflight(command, timeout=None):
        commands.append(command)
        return {'measurement_identity': {'digest': 'b' * 64, 'reusable': True, 'unknown_reasons': []},
                'import_identity': None}

    monkeypatch.setattr(node_mod, 'run_preflight', fake_preflight)
    node = _node()
    node.config['worker_python'] = '/opt/engine/bin/python'
    node.container_image = 'engine-image:latest'
    with preflight_scope(True):
        node.final_algo_config
    (command,) = commands
    # Same wrapper as the node's own command: resolution sees the container's
    # engine and the container-only worker interpreter.
    assert command.startswith('docker run')
    assert 'engine-image:latest' in command
    assert 'magnet.backends.aiq_evals.cli.resolve_node' in command
    assert '--worker_python=/opt/engine/bin/python' in command
    assert node.command.startswith('docker run')


# --- M4: evidence is recomputed from the validated run, never trusted ---------

def _publish(store_root, value, *, task='t', digest='c' * 64, other=None, coverage='complete'):
    from magnet_evals.artifacts import publish_run
    from magnet_evals.contracts import (
        CoverageFacts,
        EvaluationRequest,
        EvaluationResult,
        ExecutionContext,
        MeasurementIdentity,
        MetricRecord,
        ModelBinding,
        ResolvedEvaluation,
        ResultRecord,
    )
    from magnet_evals.store import ResultStore

    identity = MeasurementIdentity(algorithm='t', digest=digest, reusable=True)
    request = EvaluationRequest(engine='helm', task=task, models=(ModelBinding(role='primary', model='m'),))
    resolved = ResolvedEvaluation(request=request, adapter_version='a', engine_version=None,
                                  native_config={}, identity=identity)
    metrics = [MetricRecord(task=task, model_role='primary', metric='acc', value=value, denominator=4)]
    if other is not None:
        metrics.append(MetricRecord(task=task, model_role='primary', metric='other', value=other, denominator=4))
    result = EvaluationResult(engine='helm', identity=identity, status='succeeded', records=(
        ResultRecord(task=task, model_role='primary', metrics=tuple(metrics),
                     coverage=CoverageFacts(status=coverage, expected=4, processed=4 if coverage == 'complete' else 3)),
    ))
    path = ResultStore(store_root).run_path(digest)
    return publish_run(path, resolved=resolved, result=result, context=ExecutionContext(output_dir=path))


def _schedule(node_dir, run, *, select=None, policy='complete', store=None, import_identity=None):
    """Write the invoke.sh kwdagger would render for a node scheduled like this."""
    import shlex

    from magnet.backends.aiq_evals.projection import normalize_selector

    node_dir.mkdir(parents=True, exist_ok=True)
    expected = {'select': normalize_selector(select), 'coverage_policy': policy,
                'measurement_identity': run.resolved.identity.digest}
    if import_identity:
        expected['import_identity'] = import_identity
    request = json.dumps(run.resolved.request.to_dict(), sort_keys=True)
    store = store or run.path.parent.parent.parent  # <store>/runs/<dd>/<digest>
    (node_dir / 'invoke.sh').write_text(
        '#!/bin/bash\n# Root node\n'
        f'python -m magnet.backends.aiq_evals.cli.check_done {node_dir}/evaluation.json '
        f'--expected={shlex.quote(json.dumps(expected, sort_keys=True))} || \\\n'
        'python -m magnet.backends.aiq_evals.cli.run_node \\\n'
        f'    --request={shlex.quote(request)} \\\n'
        f'    --store_dpath={store} \\\n'
        f'    --out_dpath={node_dir} \\\n'
        f'    --coverage_policy={policy}\n'
    )


def _evaluation_json(node_dir, run, *, select=None, policy='complete', schedule=True):
    from magnet.backends.aiq_evals.projection import normalize_selector

    node_dir.mkdir(parents=True, exist_ok=True)
    if schedule:
        _schedule(node_dir, run, select=select, policy=policy)
    fpath = node_dir / 'evaluation.json'
    fpath.write_text(json.dumps({
        'schema': node_mod.EVALUATION_SCHEMA,
        'action': 'executed',
        'run_path': str(run.path),
        'measurement_identity': run.resolved.identity.to_dict(),
        'normalized_artifact_identity': run.manifest['normalized_artifact_identity'],
        'import_identity': None,
        'select': normalize_selector(select),
        'coverage_policy': policy,
        'request': run.resolved.request.to_dict(),
    }))
    return fpath


def _row(fpath):
    """Load a row as kwdagger does (the node's own scheduling record is invoke.sh)."""
    node = SimpleNamespace(name='evaluate', out_paths={'evaluation_fname': 'evaluation.json'},
                           primary_out_key='evaluation_fname')
    return dict(node_mod.load_evaluation_row(node, fpath.parent))


def _assert_invalid(row, reason=''):
    assert row['metrics.evaluate.eligible'] is False
    assert 'metrics.evaluate.score' not in row
    assert row['metrics.evaluate.ineligible_reasons'].startswith('invalid evaluation')
    assert reason in row['metrics.evaluate.ineligible_reasons']


def test_edited_evaluation_json_cannot_change_the_evidence(tmp_path):
    run = _publish(tmp_path / 'store', 0.5)
    fpath = _evaluation_json(tmp_path / 'node', run)
    assert _row(fpath)['metrics.evaluate.score'] == 0.5

    # Claim-facing values written into evaluation.json are ignored: the row is
    # recomputed from the validated run.
    summary = json.loads(fpath.read_text())
    summary['evidence'] = {'selected': {'value': 0.99}, 'eligible': True, 'denominator': 1000}
    summary['score'] = 0.99
    fpath.write_text(json.dumps(summary))
    row = _row(fpath)
    assert row['metrics.evaluate.score'] == 0.5 and row['metrics.evaluate.denominator'] == 4
    assert node_mod.evaluation_is_valid(fpath)

    # A changed run reference or identity makes the node invalid: it is not
    # done (and reruns), and a load reports an ineligible row, not a score.
    other = _publish(tmp_path / 'store', 0.9, digest='d' * 64)
    for key, value in (
        ('run_path', str(other.path)),
        ('normalized_artifact_identity', 'f' * 64),
        ('measurement_identity', {'digest': 'e' * 64}),
    ):
        tampered = {**json.loads(_evaluation_json(tmp_path / 'node', run).read_text()), key: value}
        _schedule(tmp_path / 'node', run)
        fpath.write_text(json.dumps(tampered))
        assert not node_mod.evaluation_is_valid(fpath), key
        _assert_invalid(_row(fpath))


def test_edited_projection_is_rejected_when_rows_load(tmp_path):
    # The reviewer's case: partial coverage, scheduled as select=acc/complete.
    # Rewriting the recorded selector and policy must not make it eligible.
    run = _publish(tmp_path / 'store', 0.2, other=0.99, coverage='partial')
    fpath = _evaluation_json(tmp_path / 'node', run, select={'metric': 'acc'})
    row = _row(fpath)
    assert row['metrics.evaluate.eligible'] is False and 'metrics.evaluate.score' not in row
    edited = {**json.loads(fpath.read_text()), 'select': {'metric': 'other'}, 'coverage_policy': 'any'}
    fpath.write_text(json.dumps(edited))
    _assert_invalid(_row(fpath), 'differs from what was scheduled')
    # A node genuinely scheduled that way is eligible.
    genuine = _evaluation_json(tmp_path / 'node2', run, select={'metric': 'other'}, policy='any')
    assert _row(genuine)['metrics.evaluate.score'] == 0.99
    # Without kwdagger's scheduling record, the row cannot be trusted.
    orphan = _evaluation_json(tmp_path / 'node3', run, schedule=False)
    _assert_invalid(_row(orphan), 'no scheduling record')


def test_a_run_of_another_request_is_rejected_when_rows_load(tmp_path):
    run = _publish(tmp_path / 'store', 0.5)
    other = _publish(tmp_path / 'store', 0.9, task='another-task', digest='d' * 64)
    fpath = _evaluation_json(tmp_path / 'node', run)
    # Point the node at a valid run of a different request, with consistent identities.
    swapped = {**json.loads(fpath.read_text()), 'run_path': str(other.path),
               'normalized_artifact_identity': other.manifest['normalized_artifact_identity'],
               'measurement_identity': other.resolved.identity.to_dict(),
               'request': other.resolved.request.to_dict()}
    fpath.write_text(json.dumps(swapped))
    _assert_invalid(_row(fpath))


def test_evaluation_json_cannot_redirect_to_another_run_of_the_measurement(tmp_path):
    # A measurement can have several valid bundles (earlier attempts, other
    # imports). A non-import node must load the store's canonical run.
    from magnet_evals.artifacts import publish_run
    from magnet_evals.contracts import ExecutionContext

    canonical = _publish(tmp_path / 'store', 0.5)
    other_path = tmp_path / 'store' / 'attempts' / 'cc' / ('c' * 64) / 'older-attempt'
    other_result = type(canonical.result).from_dict({
        **canonical.result.to_dict(),
        'records': [{**canonical.result.records[0].to_dict(),
                     'metrics': [{**canonical.result.records[0].metrics[0].to_dict(), 'value': 0.9}]}],
    })
    other = publish_run(other_path, resolved=canonical.resolved, result=other_result,
                        context=ExecutionContext(output_dir=other_path))
    fpath = _evaluation_json(tmp_path / 'node', canonical)
    assert _row(fpath)['metrics.evaluate.score'] == 0.5
    redirected = {**json.loads(fpath.read_text()), 'run_path': str(other.path),
                  'normalized_artifact_identity': other.manifest['normalized_artifact_identity']}
    fpath.write_text(json.dumps(redirected))
    assert node_mod.evaluation_is_valid(fpath)  # the bundle itself is valid ...
    _assert_invalid(_row(fpath), 'acquisition slot')  # ... but not this node's slot


def test_import_nodes_must_load_their_scheduled_import_slot(tmp_path):
    from magnet_evals.store import ResultStore

    canonical = _publish(tmp_path / 'store', 0.5)
    store = ResultStore(tmp_path / 'store')
    native = 'ab' * 32
    slot = store.import_path('c' * 64, native)
    slot.parent.mkdir(parents=True)
    import shutil as _shutil
    _shutil.copytree(canonical.path, slot)
    fpath = _evaluation_json(tmp_path / 'node', canonical, schedule=False)
    _schedule(tmp_path / 'node', canonical, import_identity=native)
    summary = {**json.loads(fpath.read_text()), 'import_identity': native}
    # The canonical run is valid but is not the scheduled import's slot.
    fpath.write_text(json.dumps(summary))
    _assert_invalid(_row(fpath))


def test_comparison_rows_follow_the_scheduled_inputs(tmp_path):
    import shlex

    from magnet.backends.aiq_evals.cli import compare

    root = tmp_path / '_kwdagger'
    left = _evaluation_json(root / 'helm' / 'h1', _publish(tmp_path / 's1', 0.25))
    right = _evaluation_json(root / 'inspect' / 'i1', _publish(tmp_path / 's2', 0.75))
    decoy = _evaluation_json(root / 'inspect' / 'i2', _publish(tmp_path / 's3', 0.95))
    node_dir = root / 'evaluate' / 'e1'
    node_dir.mkdir(parents=True)
    compare.main([f'--left_fpath={left}', f'--right_fpath={right}', '--mapping=same task',
                  f'--out_fpath={node_dir / "comparison.json"}'])
    (node_dir / 'invoke.sh').write_text(
        '#!/bin/bash\npython -m magnet.backends.aiq_evals.cli.compare '
        f'--left_fpath={left} --right_fpath={right} --mapping={shlex.quote("same task")} '
        f'--out_fpath={node_dir / "comparison.json"}\n'
    )
    node = SimpleNamespace(name='evaluate', out_paths={'out_fpath': 'comparison.json'}, primary_out_key='out_fpath')
    row = dict(compare.load_kwdagger_result(node, node_dir))
    assert row['metrics.evaluate.difference'] == 0.5
    # Redirect the recorded comparison to the decoy and change the mapping: ignored.
    recorded = json.loads((node_dir / 'comparison.json').read_text())
    recorded['right']['fpath'] = str(decoy)
    recorded['mapping'] = 'anything'
    (node_dir / 'comparison.json').write_text(json.dumps(recorded))
    row = dict(compare.load_kwdagger_result(node, node_dir))
    assert row['metrics.evaluate.difference'] == 0.5 and row['metrics.evaluate.mapping'] == 'same task'
    # Without a scheduling record nothing is comparable.
    (node_dir / 'invoke.sh').unlink()
    assert dict(compare.load_kwdagger_result(node, node_dir))['metrics.evaluate.comparable'] is False


def test_preflight_is_bounded_and_kills_its_process_group(tmp_path):
    marker = tmp_path / 'pid'
    start = time.monotonic()
    with pytest.raises(node_mod.PreflightError, match='timed out'):
        node_mod.run_preflight(f'sleep 60 & echo $! > {marker}; wait', timeout=1)
    assert time.monotonic() - start < 30
    pid = int(marker.read_text())
    stat = Path(f'/proc/{pid}/stat')
    assert not stat.exists() or stat.read_text().split()[2] == 'Z'


@needs_helm
def test_a_stale_schedule_stops_before_any_engine_work(tmp_path):
    # Preflight said one identity; the request now resolves to another. The
    # node must ask to be rescheduled without executing anything.
    request = {**HELM_ALGO, 'schema_version': 1}
    proc = subprocess.run(
        [sys.executable, '-m', 'magnet.backends.aiq_evals.cli.run_node', f'--request={json.dumps(request)}',
         f'--store_dpath={tmp_path / "store"}', f'--out_dpath={tmp_path / "node"}',
         f'--measurement_identity={"0" * 64}', f'--worker_python={HELM_PYTHON}'],
        capture_output=True, text=True,
    )
    assert proc.returncode == 3, proc.stderr
    summary = json.loads((tmp_path / 'node' / 'attempt_summary.json').read_text())
    assert summary['status'] == 'not-run' and 'reschedule' in summary['error']
    assert not (tmp_path / 'store' / 'attempts').exists()


def test_edited_run_payload_invalidates_the_node(tmp_path):
    run = _publish(tmp_path / 'store', 0.5)
    fpath = _evaluation_json(tmp_path / 'node', run)
    results = run.path / 'results.json'
    payload = json.loads(results.read_text())
    payload['records'][0]['metrics'][0]['value'] = 0.99
    results.write_text(json.dumps(payload))
    assert not node_mod.evaluation_is_valid(fpath)
    _assert_invalid(_row(fpath), 'does not validate')


def test_done_check_pins_the_scheduled_projection(tmp_path):
    run = _publish(tmp_path / 'store', 0.5)
    fpath = _evaluation_json(tmp_path / 'node', run, select={'metric': 'acc'})
    expected = {'select': {'metric': 'acc'}, 'coverage_policy': 'complete', 'measurement_identity': 'c' * 64}
    assert node_mod.evaluation_is_valid(fpath, expected)
    # Someone edits the recorded selector or policy: the node is not done.
    for key, value in (('select', {'metric': 'other'}), ('coverage_policy', 'any')):
        edited = {**json.loads(fpath.read_text()), key: value}
        other = tmp_path / 'edited' / key / 'evaluation.json'
        other.parent.mkdir(parents=True)
        other.write_text(json.dumps(edited))
        assert not node_mod.evaluation_is_valid(other, expected), key
    assert not node_mod.evaluation_is_valid(fpath, {**expected, 'measurement_identity': 'd' * 64})


@needs_helm
def test_rows_are_never_served_from_a_stale_cache(tmp_path):
    # kwdagger caches resolved rows keyed on evaluation.json's mtime. Loading
    # without rescheduling (as `evidence.scope: all` does for older nodes) must
    # still notice that the run changed, and must not abort other rows.
    recipe, _ = evaluate(helm_recipe(tmp_path), tmp_path / 'out')
    (row,) = recipe_rows(recipe)
    assert row['row']['metrics.evaluate.eligible'] is True
    run_path = Path(row['row']['metrics.evaluate.run_path'])
    (run_path / 'native' / 'injected.txt').write_text('tamper\n')
    (again,) = recipe_rows(recipe)
    assert again['row']['metrics.evaluate.eligible'] is False
    assert 'metrics.evaluate.score' not in again['row']
    assert 'invalid evaluation' in again['row']['metrics.evaluate.ineligible_reasons']


@needs_helm
def test_preflight_does_not_need_secret_values(tmp_path, monkeypatch):
    # A leased node's key exists only inside its lease; scheduling resolves
    # before that, and identity never needs the value.
    from magnet.backends.aiq_evals.cli import resolve_node

    monkeypatch.delenv('AIQ_LEASED_KEY', raising=False)
    request = {**HELM_ALGO, 'schema_version': 1, 'engine_options': {'required_secrets': ['AIQ_LEASED_KEY']}}
    payload = resolve_node.resolve(request, HELM_PYTHON, None, False)
    assert payload['measurement_identity']['reusable']


# --- ADR-0011: imports keyed by content ----------------------------------------

def _helm_native_run(tmp_path):
    """Execute HELM once and return a copy of its native run directory."""
    outcome = magnet_evals.ensure_evaluation(
        magnet_evals.EvaluationRequest.from_dict({**HELM_ALGO, 'schema_version': 1}),
        tmp_path / 'seed-store', worker_python=HELM_PYTHON,
    )
    assert outcome.run.result.status == 'succeeded'
    (run_dir,) = (outcome.run.path / 'native').rglob('run_spec.json')
    source = tmp_path / 'helm-import'
    shutil.copytree(run_dir.parent, source / run_dir.parent.name)
    return source


def _set_helm_exact_match(source, value):
    (stats_fpath,) = source.rglob('stats.json')
    stats = json.loads(stats_fpath.read_text())
    for stat in stats:
        name = stat['name']
        if name['name'] == 'exact_match' and name.get('split') == 'test' and not name.get('perturbation'):
            stat['mean'] = value
    stats_fpath.write_text(json.dumps(stats, indent=2))


@needs_helm
def test_editing_imported_native_files_in_place_reruns_the_node(tmp_path):
    source = _helm_native_run(tmp_path)
    _set_helm_exact_match(source, 0.25)
    out = tmp_path / 'out'
    fpath = helm_recipe(tmp_path, algo={'import_source': str(source)})
    recipe, _ = evaluate(fpath, out)
    (first,) = recipe_rows(recipe)
    assert first['row']['metrics.evaluate.action'] == 'imported'
    assert first['row']['metrics.evaluate.score'] == 0.25

    # Unchanged content: kwdagger skips the node.
    evaluate(fpath, out)
    assert len(evaluations(out)) == 1

    # Same path, edited content: new import identity -> new node -> new value.
    _set_helm_exact_match(source, 0.75)
    recipe, card = evaluate(fpath, out, )
    rows = sorted(recipe_rows(recipe), key=lambda r: r['row']['metrics.evaluate.score'])
    assert [r['row']['metrics.evaluate.score'] for r in rows] == [0.25, 0.75]
    assert rows[1]['row']['metrics.evaluate.action'] == 'imported'
    assert len({r['row']['metrics.evaluate.import_identity'] for r in rows}) == 2
    # The requested (current) evidence is the new content only.
    (verdict,) = run_verdicts(latest_run_dir(out))
    assert verdict['symbols']['metrics.evaluate.score'] == 0.75


# --- ADR-0011: concurrent projections of one measurement execute it once ------

@needs_helm
@needs(shutil.which('tmux') is not None, 'needs tmux for concurrent scheduling')
def test_concurrent_selector_nodes_execute_the_native_evaluation_once(tmp_path):
    selects = [
        {'metric': 'exact_match', 'group': 'test', 'score': None},
        {'metric': 'quasi_exact_match', 'group': 'test', 'score': None},
        {'metric': 'prefix_exact_match', 'group': 'test', 'score': None},
    ]
    fpath = write_recipe(
        tmp_path, {'evaluate': evaluation_node(HELM_ALGO)},
        matrix={'evaluate.select': [json.dumps(s, sort_keys=True) for s in selects]}, name='concurrent',
    )
    out = tmp_path / 'out'
    recipe, card = evaluate(fpath, out, backend='tmux', tmux_workers=3)
    records = [json.loads(p.read_text()) for p in evaluations(out)]
    assert len(records) == 3
    assert sorted(r['action'] for r in records) == ['executed', 'reused', 'reused']
    assert len(store_attempts(out)) == 1
    assert len(recipe_rows(recipe)) == 3


# --- M10: evidence scopes and failure provenance -------------------------------

@needs_helm
def test_requested_versus_accumulated_evidence(tmp_path):
    out = tmp_path / 'out'
    other = {'metric': 'quasi_exact_match', 'group': 'test', 'score': None}
    evaluate(helm_recipe(tmp_path), out)
    _, requested = evaluate(helm_recipe(tmp_path, select=other), out)
    assert len(requested.cell_results) == 1
    assert requested.cell_results[0].evidence_row['metrics.evaluate.selected.metric'] == 'quasi_exact_match'
    _, accumulated = evaluate(helm_recipe(tmp_path, select=other, scope='all'), out)
    metrics = sorted(c.evidence_row['metrics.evaluate.selected.metric'] for c in accumulated.cell_results)
    assert metrics == ['exact_match', 'quasi_exact_match']


@needs_repo
@needs_helm
def test_native_failure_is_provenance_not_a_verdict(tmp_path, monkeypatch):
    repo_on_worker_path(monkeypatch)
    fpath = helm_recipe(tmp_path, algo={
        'task': 'aiq_p5_fail', 'task_revision': 'x', 'data_revision': 'x',
        'engine_options': {'plugins': ['tests.native.helm_plugin_fixture']},
    }, claim='assert metrics.evaluate.score > 0.5')
    out = tmp_path / 'out'
    recipe, card = evaluate(fpath, out)
    # No evidence row, so the claim is not falsified by an execution error.
    assert card.result != 'FALSIFIED' and not card.cell_results
    assert card.requested_work['attempt_status'] == {'failed': 1}
    (node_dir,) = (out / '_kwdagger' / 'evaluate').iterdir()
    summary = json.loads((node_dir / 'attempt_summary.json').read_text())
    assert summary['status'] == 'failed' and not (node_dir / 'evaluation.json').exists()
    (attempt,) = store_attempts(out)
    assert attempt.read_text().strip() == 'failed'


# --- M8: leasing, duplicate startup, cancellation ------------------------------

def test_lease_runtime_maps_lease_env_and_refuses_mismatch():
    from magnet.backends.aiq_evals.cli.run_node import lease_runtime

    request = {'models': [{'role': 'primary', 'model': 'smol-135'}, {'role': 'grader', 'model': 'judge-7'}]}
    env = {'OPENAI_BASE_URL': 'http://gw/v1', 'OPENAI_API_KEY': 'lease-key',
           'INFER_STACK_ENDPOINT_SMOL_135': 'smol-135', 'INFER_STACK_ENDPOINT_JUDGE_ALIAS': 'judge-7'}
    assert lease_runtime('smol-135', request, env) == ({'primary': 'http://gw/v1'}, {'OPENAI_API_KEY': 'lease-key'})
    assert lease_runtime(None, request, env) == ({}, {})
    # Several roles, one lease: every alias sits behind the lease's base URL.
    both = {'primary': 'smol-135', 'grader': 'judge-alias'}
    assert lease_runtime(both, request, env)[0] == {'primary': 'http://gw/v1', 'grader': 'http://gw/v1'}
    with pytest.raises(SystemExit, match='no lease is active'):
        lease_runtime('smol-135', request, {})
    with pytest.raises(SystemExit, match='serves'):
        lease_runtime('smol-135', request, {**env, 'INFER_STACK_ENDPOINT_SMOL_135': 'other-name'})
    with pytest.raises(SystemExit, match='does not bind'):
        lease_runtime({'critic': 'judge-alias'}, request, env)
    # Without the lease's served-name variable nothing can be verified.
    with pytest.raises(SystemExit, match='exports no INFER_STACK_ENDPOINT_SMOL_135'):
        lease_runtime('smol-135', request, {'OPENAI_BASE_URL': 'http://gw/v1'})


def _leased_node(store, digest, monkeypatch, *, perf=None, **algo):
    from magnet.leasing import LeaseSettings

    monkeypatch.setattr(node_mod, 'run_preflight', lambda command, timeout=None: {
        'measurement_identity': {'digest': digest, 'reusable': True, 'unknown_reasons': []},
        'import_identity': 'f' * 64 if algo.get('import_source') else None,
    })
    node = EvaluationNode(
        name='evaluate',
        algo_params={'engine': 'inspect_ai', 'task': 't', 'models': [{'model': 'smol-135', 'revision': 'r'}], **algo},
        perf_params={'endpoint': 'smol-135', 'store_dpath': str(store), **(perf or {})},
    )
    node.apply_lease_settings(LeaseSettings(enabled=True, allowed_gpus=False))
    node.configure({})
    return node


def test_leasing_is_decided_by_a_gate_when_the_node_runs(tmp_path, monkeypatch):
    import shlex

    monkeypatch.delenv('INFER_STACK_LEASE_ID', raising=False)
    node = _leased_node(tmp_path / 'store', 'c' * 64, monkeypatch,
                        perf={'endpoints': json.dumps({'grader': 'judge-alias'})})
    with preflight_scope(True):
        command = node.command
    # The outer command is the host-side gate; the lease wraps only the child.
    assert not command.startswith('infer-stack')
    assert 'magnet.backends.aiq_evals.cli.run_node' in command
    tokens = shlex.split(command.replace('\\\n', ' '))
    (child,) = [tok[len('--leased_command='):] for tok in tokens if tok.startswith('--leased_command=')]
    assert child.startswith('infer-stack run --endpoint smol-135,judge-alias')
    assert '--lock_held=True' in child
    assert node.lease_roles() == {'primary': 'smol-135', 'grader': 'judge-alias'}
    # In a container the lease's served-name variables are forwarded by name.
    node.container_image = 'engine-image:latest'
    with preflight_scope(True):
        contained = node.command
    tokens = shlex.split(contained.replace('\\\n', ' '))
    (child,) = [tok[len('--leased_command='):] for tok in tokens if tok.startswith('--leased_command=')]
    for name in ('OPENAI_BASE_URL', 'OPENAI_API_KEY', 'INFER_STACK_ENDPOINT_SMOL_135',
                 'INFER_STACK_ENDPOINT_JUDGE_ALIAS'):
        assert f'-e {name} ' in child, name
    # An import runs no model: no gate and no lease.
    importer = _leased_node(tmp_path / 'store', 'c' * 64, monkeypatch, import_source=str(tmp_path))
    with preflight_scope(True):
        assert 'infer-stack' not in importer.command and '--leased_command' not in importer.command


GATE_CHILD = """
import json, sys, time
from pathlib import Path
sys.path.insert(0, {tests!r})
from test_aiq_evals_integration import _publish
store, digest, counter = sys.argv[1], sys.argv[2], Path(sys.argv[3])
with counter.open('a') as fh:
    fh.write('child\\n')
time.sleep(1.0)  # a slow leased evaluation
_publish(store, 0.5, digest=digest)
"""


def test_concurrent_gates_start_one_leased_child(tmp_path):
    # Two nodes (e.g. two selectors) need one missing measurement. The gates
    # serialize on the store's acquisition lock: the first runs its leased
    # child, the second then finds the run and records reuse without a lease.
    digest = 'c' * 64
    store = tmp_path / 'store'
    counter = tmp_path / 'children.txt'
    script = tmp_path / 'child.py'
    script.write_text(GATE_CHILD.format(tests=str(Path(__file__).parent)))
    request = json.dumps(_publish(tmp_path / 'probe-store', 0.1, digest=digest).resolved.request.to_dict())
    procs = []
    for name, select in (('a', {'metric': 'acc'}), ('b', {'metric': 'acc'})):
        child = f'{shlex_quote(sys.executable)} {shlex_quote(str(script))} {store} {digest} {counter}'
        procs.append(subprocess.Popen([
            sys.executable, '-m', 'magnet.backends.aiq_evals.cli.run_node',
            f'--request={request}', f'--store_dpath={store}', f'--out_dpath={tmp_path / name}',
            '--evaluation_fname=evaluation.json', f'--measurement_identity={digest}',
            f'--select={json.dumps(select)}', f'--leased_command={child}',
        ]))
    assert [p.wait(timeout=120) for p in procs] == [0, 0]
    assert counter.read_text().splitlines() == ['child']
    # The stub child writes no summary; the gate that waited recorded reuse.
    records = [json.loads(p.read_text()) for p in sorted(tmp_path.glob('[ab]/evaluation.json'))]
    assert [(r['action'], r['waited']) for r in records] == [('reused', True)]


def shlex_quote(text):
    import shlex

    return shlex.quote(text)


@needs_repo
@needs_helm
def test_sigterm_cancels_the_engine_worker(tmp_path, monkeypatch):
    repo_on_worker_path(monkeypatch)
    pid_file = tmp_path / 'child.pid'
    request = {
        'engine': 'helm', 'task': 'aiq_p5_slow', 'task_revision': 'x', 'data_revision': 'x',
        'models': [{'role': 'primary', 'model': 'simple/model1', 'revision': 'local-v1'}],
        'task_options': {'max_eval_instances': 1},
        'engine_options': {'plugins': ['tests.native.helm_plugin_fixture']},
    }
    env = dict(os.environ, AIQ_P5_CHILD_PID_FILE=str(pid_file))
    proc = subprocess.Popen(
        [sys.executable, '-m', 'magnet.backends.aiq_evals.cli.run_node', '--request', json.dumps(request),
         '--store_dpath', str(tmp_path / 'store'), '--out_dpath', str(tmp_path / 'node'),
         '--worker_python', HELM_PYTHON],
        env=env,
    )
    try:
        for _ in range(600):
            if pid_file.exists() and pid_file.read_text():
                break
            time.sleep(0.1)
        else:
            pytest.fail('native HELM task never started')
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    child = ub.Path(f'/proc/{int(pid_file.read_text())}/stat')
    assert not child.exists() or child.read_text().split()[2] == 'Z'
    attempts = list((tmp_path / 'store' / 'attempts').rglob('ATTEMPT_TERMINAL'))
    assert [p.read_text().strip() for p in attempts] == ['cancelled']
    assert not (tmp_path / 'node' / 'evaluation.json').exists()


# --- engine-free projection -----------------------------------------------------

def _metric(**kw):
    base = dict(task='t', model_role='primary', metric='acc', scorer=None, score=None,
                group=None, reducer=None, value=1.0, denominator=4)
    base.update(kw)
    return SimpleNamespace(**base)


def _bundle(metrics, status='succeeded', coverage='complete'):
    records = {}
    for m in metrics:
        records.setdefault(m.task, []).append(m)
    result = SimpleNamespace(
        engine='fake', status=status,
        records=[
            SimpleNamespace(task=task, model_role='primary', metrics=ms,
                            coverage=SimpleNamespace(status=coverage, to_dict=lambda c=coverage: {'status': c}))
            for task, ms in records.items()
        ],
    )
    return SimpleNamespace(
        result=result, path='/run',
        resolved=SimpleNamespace(identity=SimpleNamespace(digest='d' * 64, reusable=True)),
        manifest={'normalized_artifact_identity': 'n' * 64},
    )


def test_selection_is_exact_or_rejected():
    metrics = [_metric(task='a'), _metric(task='b'), _metric(task='a', metric='f1')]
    result = _bundle(metrics).result
    assert select_metric(result, {'task': 'a', 'metric': 'acc'}).metric is metrics[0]
    assert select_metric(result, {'metric': 'acc'}).candidates == 2
    assert select_metric(result, {'metric': 'acc'}).metric is None
    assert select_metric(result, {'task': 'z'}).candidates == 0
    # An explicit null requires the field unset (HELM's unperturbed statistic).
    perturbed = [_metric(score='perturbation:robustness'), _metric()]
    assert select_metric(_bundle(perturbed).result, {'score': None}).metric is perturbed[1]
    with pytest.raises(ValueError, match='unknown evidence selector'):
        select_metric(result, {'metrik': 'acc'})


@pytest.mark.parametrize('status,coverage,value,policy,eligible', [
    ('succeeded', 'complete', 0.5, 'complete', True),
    ('succeeded', 'partial', 0.5, 'complete', False),
    ('succeeded', 'partial', 0.5, 'any', True),
    ('failed', 'complete', 0.5, 'complete', False),
    ('succeeded', 'complete', float('nan'), 'complete', False),
])
def test_eligibility_policy_and_score_exposure(status, coverage, value, policy, eligible):
    view = evidence_view(_bundle([_metric(value=value)], status, coverage), coverage_policy=policy)
    assert view.eligible is eligible
    flat = flat_metrics(view.to_dict())
    assert ('score' in flat) is eligible
    assert flat['selected.value'] == value or value != value
    assert flat['denominator'] == 4


def test_single_flight_store_is_shared_by_concurrent_node_processes(tmp_path):
    # Engine-free: several run_node-like processes contend for one measurement
    # in one store; exactly one executes (ADR-0011). Uses the store API directly
    # so the property is checked without an engine.
    from magnet_evals.store import ResultStore

    store = ResultStore(tmp_path / 'store')
    digest = 'ab' * 32
    order = []

    async def worker(name):
        async with store.acquisition_lock(digest) as lock:
            order.append((name, lock.waited))
            await asyncio.sleep(0.1)

    async def main():
        await asyncio.gather(*(worker(i) for i in range(3)))

    asyncio.run(main())
    assert [waited for _, waited in order] == [False, True, True]
