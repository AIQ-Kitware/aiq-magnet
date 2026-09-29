"""aiq-evals integration (aiq-evals docs/planning/aiq-magnet-integration-plan.md).

The end-to-end tests run a real engine through ``aiq_evals``: HELM's local
simple model, in this interpreter or ``$AIQ_EVALS_HELM_PYTHON``. They skip when
``aiq_evals`` or ``helm`` is unavailable. The projection tests are engine-free.
"""
import json
import os
import sys

import pytest
import ubelt as ub
import yaml

aiq_evals = pytest.importorskip('aiq_evals')

from magnet.backends.aiq_evals import EvaluationNode  # noqa: E402
from magnet.evaluation_new import NewEvaluationRecipe  # noqa: E402

HELM_PYTHON = os.environ.get('AIQ_EVALS_HELM_PYTHON', sys.executable)


def _helm_available() -> bool:
    import subprocess

    return subprocess.run([HELM_PYTHON, '-c', 'import helm, aiq_evals'], capture_output=True).returncode == 0


needs_helm = pytest.mark.skipif(not _helm_available(), reason='needs crfm-helm + aiq_evals in a worker')

SELECT = {'metric': 'exact_match', 'group': 'test', 'score': None}


def write_recipe(dpath, *, claim='assert metrics.evaluate.score >= 0', select=SELECT, max_eval=1):
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
                    'data_revision': 'helm-builtin',
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

    root = pathlib.Path(os.environ.get('AIQ_EVALS_REPO', pathlib.Path(aiq_evals.__file__).parents[1]))
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
    run = aiq_evals.load_run(row['metrics.evaluate.run_path'])
    ids = [(s.task, s.sample_id) for s in run.result.samples if s.native.get('kind') == 'sample']
    assert len({sid for _, sid in ids}) < len({t for t, _ in ids}) * 2
    assert {s.epoch for s in run.result.samples if s.epoch} == {1, 2}
    native = [m.value for m in aiq_evals.outputs.select_metrics(run, task='role_task', scorer='match', metric='accuracy')]
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
