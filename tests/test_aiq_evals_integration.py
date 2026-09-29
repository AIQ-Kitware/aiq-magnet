"""aiq-evals integration (aiq-evals docs/planning/aiq-magnet-integration-plan.md).

The end-to-end tests run a real engine through ``magnet_evals``: HELM's local
simple model, in this interpreter or ``$AIQ_EVALS_HELM_PYTHON``. They skip when
``magnet_evals`` or ``helm`` is unavailable. The projection tests are engine-free.
"""
import json
import os
import sys

import pytest
import ubelt as ub
import yaml

magnet_evals = pytest.importorskip('magnet_evals')

from types import SimpleNamespace  # noqa: E402

from magnet.backends.aiq_evals import EvaluationNode  # noqa: E402
from magnet.backends.aiq_evals import pipeline as node_mod  # noqa: E402
from magnet.backends.aiq_evals.projection import evidence_view, flat_metrics, select_metric  # noqa: E402
from magnet.evaluation_new import NewEvaluationRecipe  # noqa: E402

HELM_PYTHON = os.environ.get('AIQ_EVALS_HELM_PYTHON', sys.executable)


def _helm_available() -> bool:
    import subprocess

    return subprocess.run([HELM_PYTHON, '-c', 'import helm, magnet_evals'], capture_output=True).returncode == 0


needs_helm = pytest.mark.skipif(not _helm_available(), reason='needs crfm-helm + magnet_evals in a worker')

SELECT = {'metric': 'exact_match', 'group': 'test', 'score': None}


def write_recipe(dpath, *, claim='assert metrics.evaluate.score >= 0', select=SELECT, max_eval=1,
                 data_revision='helm-builtin'):
    fpath = ub.Path(dpath) / 'recipe.yaml'
    fpath.write_text(yaml.safe_dump({
        'name': 'aiq_evals_probe',
        'title': 'aiq-evals probe',
        'description': 'one HELM evaluation through aiq-evals',
        'version': '1.0',
        'organizations': ['Kitware'],
        'submitter': {'name': 't', 'email': 't@example.com'},
        'links': [],
        'tags': ['test'],
        'claim': {'python': claim},
        'kwdagger': {
            'result_node': 'evaluate',
            'pipeline': {'nodes': {'evaluate': {
                'class': 'magnet.backends.aiq_evals.EvaluationNode',
                'algo_params': {
                    'engine': 'helm',
                    'task': 'simple_mcqa',
                    'task_revision': 'helm-builtin',
                    'data_revision': data_revision,
                    'models': [{'role': 'primary', 'model': 'simple/model1', 'revision': 'local-v1'}],
                    'task_options': {'max_eval_instances': max_eval},
                    'select': select,
                },
                'perf_params': {'worker_python': HELM_PYTHON},
            }}},
        },
    }, sort_keys=False))
    return fpath


@needs_helm
def test_one_helm_evaluation_becomes_one_claim_row(tmp_path):
    recipe = NewEvaluationRecipe(write_recipe(tmp_path), ub.Path(tmp_path) / 'out', validate='off')
    card = recipe.evaluate(backend='serial')
    rows = recipe_rows(recipe)
    assert len(rows) == 1
    row = rows[0]['row']
    assert row['metrics.evaluate.eligible'] is True
    assert row['metrics.evaluate.engine'] == 'helm'
    assert row['metrics.evaluate.selected.metric'] == 'exact_match'
    assert row['metrics.evaluate.score'] == row['metrics.evaluate.selected.value']
    assert len(row['metrics.evaluate.measurement_identity']) == 64
    assert card is not None
    verdicts = list((ub.Path(tmp_path) / 'out').glob('*/results/*/verdict.json'))
    assert len(verdicts) == 1
    verdict = json.loads(verdicts[0].read_text())
    assert verdict['status'] == 'VERIFIED'
    assert verdict['consumed'] == ['metrics.evaluate.score']


def recipe_rows(recipe):
    from magnet._kwdagger import KWDaggerProcessor

    processor = KWDaggerProcessor(recipe.kwdagger, root_dpath=recipe.kwdagger_dpath)
    return processor.load_available_result_rows()


# --- M6: cardinality acceptance experiment -----------------------------------
# Representative native multi-result fixtures from aiq-evals (committed there;
# see aiq-evals docs/planning/phase1-evidence.md), imported through real engine
# workers, loaded by real kwdagger build_tables + KWDaggerProcessor, and judged
# through ClaimResultNamespace by the real evaluate_new flow.

def _aiq_evals_repo():
    import pathlib

    root = pathlib.Path(os.environ.get('AIQ_EVALS_REPO', pathlib.Path(magnet_evals.__file__).parents[1]))
    return root if (root / 'tests' / 'fixtures').is_dir() else None


def _worker(var):
    python = os.environ.get(var)
    return python if python and os.path.exists(python) else None


INSPECT_PYTHON = _worker('AIQ_EVALS_INSPECT_PYTHON')
OLMO_PYTHON = _worker('AIQ_EVALS_OLMO_PYTHON')
REPO = _aiq_evals_repo()


def multi_recipe(dpath, *, engine_params, worker, selects, claim):
    fpath = ub.Path(dpath) / 'recipe.yaml'
    fpath.write_text(yaml.safe_dump({
        'name': 'cardinality', 'title': 'cardinality', 'description': 'M6 cardinality',
        'version': '1.0', 'organizations': ['Kitware'],
        'submitter': {'name': 't', 'email': 't@example.com'}, 'links': [], 'tags': ['test'],
        'claim': {'python': claim},
        'kwdagger': {
            'result_node': 'evaluate',
            'pipeline': {'nodes': {'evaluate': {
                'class': 'magnet.backends.aiq_evals.EvaluationNode',
                'algo_params': engine_params,
                'perf_params': {'worker_python': worker},
            }}},
            'matrix': {'evaluate.select': [json.dumps(s, sort_keys=True) for s in selects]},
        },
    }, sort_keys=False))
    return fpath


def run_cardinality(tmp_path, **kwargs):
    recipe = NewEvaluationRecipe(multi_recipe(tmp_path, **kwargs), ub.Path(tmp_path) / 'out', validate='off')
    recipe.evaluate(backend='serial')
    rows = recipe_rows(recipe)
    verdicts = [
        json.loads(p.read_text())
        for p in (ub.Path(tmp_path) / 'out').glob('*/results/*/verdict.json')
    ]
    by_select = {json.dumps(json.loads(r['row']['params.evaluate.select']), sort_keys=True): r for r in rows}
    return rows, verdicts, by_select


@pytest.mark.skipif(not (INSPECT_PYTHON and REPO), reason='needs AIQ_EVALS_INSPECT_PYTHON and aiq-evals fixtures')
def test_cardinality_inspect_multi_log_epochs(tmp_path):
    fixture = REPO / 'tests' / 'fixtures' / 'inspect-native' / 'multi'
    engine_params = {
        'engine': 'inspect_ai',
        'task': str(REPO / 'tests' / 'native' / 'inspect_fixture.py'),
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
    # Rejected selections expose no score: nothing is averaged across tasks.
    for select, count in ((ambiguous, 3), (nothing, 0)):
        rejected = by_select[json.dumps(select, sort_keys=True)]['row']
        assert 'metrics.evaluate.score' not in rejected or rejected['metrics.evaluate.score'] != rejected['metrics.evaluate.score']
        assert rejected['metrics.evaluate.eligible'] is False
        assert rejected['metrics.evaluate.candidates'] == count
    statuses = sorted(v['status'] for v in verdicts)
    assert statuses.count('VERIFIED') == 1, statuses

    # Duplicate sample IDs across task logs and repeated epochs stay inside the
    # aiq-evals run; the claim-facing value is the selected task's native metric.
    run = magnet_evals.load_run(row['metrics.evaluate.run_path'])
    ids = [(s.task, s.sample_id) for s in run.result.samples if s.native.get('kind') == 'sample']
    assert len({sid for _, sid in ids}) < len({t for t, _ in ids}) * 2
    assert {s.epoch for s in run.result.samples if s.epoch} == {1, 2}
    native = [m.value for m in magnet_evals.outputs.select_metrics(run, task='role_task', scorer='match', metric='accuracy')]
    assert native == [row['metrics.evaluate.score']]


@pytest.mark.skipif(not (OLMO_PYTHON and REPO), reason='needs AIQ_EVALS_OLMO_PYTHON and aiq-evals fixtures')
def test_cardinality_olmo_suite_prefix_overlapping_tasks(tmp_path):
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
        'import_source': str(REPO / 'tests' / 'fixtures' / 'olmo-native' / 'multi'),
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


# --- M3/M4/M9: identity, reuse, dry run, projection ---------------------------


def _evaluations(tmp_path):
    return sorted((ub.Path(tmp_path) / 'out' / '_kwdagger' / 'evaluate').glob('*/evaluation.json'))


def _store_attempts(tmp_path):
    return sorted((ub.Path(tmp_path) / 'out' / '_kwdagger' / '_aiq_evals_store' / 'attempts').rglob('attempt.json'))


@needs_helm
def test_reuse_selector_change_and_stale_marker(tmp_path):
    out = ub.Path(tmp_path) / 'out'
    recipe = NewEvaluationRecipe(write_recipe(tmp_path), out, validate='off')
    recipe.evaluate(backend='serial')
    assert len(_evaluations(tmp_path)) == 1 and len(_store_attempts(tmp_path)) == 1

    # Same request: kwdagger skips the node (it still validates as done).
    recipe.evaluate(backend='serial')
    assert len(_evaluations(tmp_path)) == 1 and len(_store_attempts(tmp_path)) == 1

    # A different evidence selector reruns the node, not the native evaluation.
    reselect = NewEvaluationRecipe(
        write_recipe(tmp_path, select={'metric': 'quasi_exact_match', 'group': 'test', 'score': None}),
        out, validate='off',
    )
    reselect.evaluate(backend='serial')
    evaluations = [json.loads(p.read_text()) for p in _evaluations(tmp_path)]
    assert len(evaluations) == 2
    assert sorted(e['action'] for e in evaluations) == ['executed', 'reused']
    assert len({e['measurement_identity']['digest'] for e in evaluations}) == 1
    assert len(_store_attempts(tmp_path)) == 1

    # A stale marker cannot hide a broken run: tamper with the canonical run.
    run_path = ub.Path(evaluations[0]['run_path'])
    (run_path / 'native' / 'injected.txt').write_text('tamper\n')
    recipe.evaluate(backend='serial')
    assert len(_store_attempts(tmp_path)) == 2
    assert len(recipe_rows(recipe)) == 2


@needs_helm
def test_non_reusable_identity_always_executes(tmp_path):
    fpath = write_recipe(tmp_path, data_revision=None)
    recipe = NewEvaluationRecipe(fpath, ub.Path(tmp_path) / 'out', validate='off')
    recipe.evaluate(backend='serial')
    recipe.evaluate(backend='serial')
    evaluations = [json.loads(p.read_text()) for p in _evaluations(tmp_path)]
    assert len(evaluations) == 2  # a nonce in the node id: never skipped as done
    assert all(e['preflight_identity'].startswith('unresolved-') for e in evaluations)
    assert all(not e['measurement_identity']['reusable'] for e in evaluations)


def test_dry_run_resolves_nothing(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('dry run must not resolve (would run task code)')

    monkeypatch.setattr(node_mod, '_preflight_digest', forbidden)
    recipe = NewEvaluationRecipe(write_recipe(tmp_path), ub.Path(tmp_path) / 'out', validate='off')
    recipe.evaluate(backend='serial', dry_run=True)
    assert not (ub.Path(tmp_path) / 'out' / '_kwdagger' / '_aiq_evals_store').exists()
    assert not _evaluations(tmp_path)


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


# --- M8: leasing, duplicate startup, cancellation ------------------------------

def test_lease_runtime_maps_lease_env_and_refuses_mismatch():
    from magnet.backends.aiq_evals.cli.run_node import lease_runtime

    request = {'models': [{'role': 'primary', 'model': 'smol-135'}]}
    env = {'OPENAI_BASE_URL': 'http://gw/v1', 'OPENAI_API_KEY': 'lease-key'}
    assert lease_runtime('smol-135', request, env) == ({'primary': 'http://gw/v1'}, {'OPENAI_API_KEY': 'lease-key'})
    assert lease_runtime(None, request, env) == ({}, {})
    with pytest.raises(SystemExit, match='no lease is active'):
        lease_runtime('smol-135', request, {})
    with pytest.raises(SystemExit, match='serves'):
        lease_runtime('smol-135', request, {**env, 'INFER_STACK_ENDPOINT_SMOL_135': 'other-name'})


def _leased_node(store, digest):
    from magnet.leasing import LeaseSettings

    node = EvaluationNode(
        name='evaluate',
        algo_params={
            'engine': 'inspect_ai', 'task': 't', 'models': [{'model': 'smol-135', 'revision': 'r'}],
            'measurement_identity': digest,
        },
        perf_params={'endpoint': 'smol-135', 'store_dpath': str(store)},
    )
    node.apply_lease_settings(LeaseSettings(enabled=True))
    node.configure({})
    return node


def test_lease_wraps_only_when_native_work_is_needed(tmp_path, monkeypatch):
    from magnet_evals.artifacts import publish_run
    from magnet_evals.contracts import (
        EvaluationRequest, EvaluationResult, ExecutionContext, MeasurementIdentity, ModelBinding,
        ResolvedEvaluation, ResultRecord,
    )
    from magnet_evals.store import ResultStore

    monkeypatch.delenv('INFER_STACK_LEASE_ID', raising=False)
    digest = 'a' * 64
    store = ResultStore(tmp_path / 'store')
    assert 'infer-stack run' in _leased_node(store.root, digest).command

    identity = MeasurementIdentity(algorithm='t', digest=digest, reusable=True)
    request = EvaluationRequest(engine='inspect_ai', task='t', models=(ModelBinding(role='primary', model='m'),))
    resolved = ResolvedEvaluation(request=request, adapter_version='a', engine_version=None,
                                  native_config={}, identity=identity)
    result = EvaluationResult(engine='inspect_ai', identity=identity, status='succeeded',
                              records=(ResultRecord(task='t', model_role='primary', metrics=()),))
    path = store.run_path(digest)
    publish_run(path, resolved=resolved, result=result, context=ExecutionContext(output_dir=path))
    # A valid stored run means the node only reuses: no model is leased.
    assert 'infer-stack run' not in _leased_node(store.root, digest).command


@pytest.mark.skipif(not REPO, reason='needs aiq-evals HELM plugin fixtures')
@needs_helm
def test_sigterm_cancels_the_engine_worker(tmp_path):
    import signal
    import subprocess
    import time

    pid_file = tmp_path / 'child.pid'
    request = {
        'engine': 'helm', 'task': 'aiq_p5_slow', 'task_revision': 'x', 'data_revision': 'x',
        'models': [{'role': 'primary', 'model': 'simple/model1', 'revision': 'local-v1'}],
        'task_options': {'max_eval_instances': 1},
        'engine_options': {'plugins': ['tests.native.helm_plugin_fixture']},
    }
    env = dict(os.environ, AIQ_P5_CHILD_PID_FILE=str(pid_file), PYTHONPATH=str(REPO))
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
