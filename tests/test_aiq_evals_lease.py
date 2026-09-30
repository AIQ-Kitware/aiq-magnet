"""Real infer-stack leases for EvaluationNodes (integration plan M8).

`infer-stack run` with its ``null`` serving backend does the real lease
bookkeeping (sqlite ledger, coalescing, release) and exports the real lease
environment, but starts no model. The test serves the deterministic example
endpoint on a free port. A shim first on PATH adds ``--base_url`` for that port
to ``infer-stack run`` (MAGNET's lease prefix leaves the gateway address to
infer-stack) and otherwise runs the real CLI. Every request names a dead
``base_url``, so a successful run proves the leased endpoint reached the engine.

Needs ``infer-stack`` on PATH (MAGNET's ``leasing`` extra) and the engine
workers from ``aiq_evals_support``. The Inspect test also needs
``$AIQ_EVALS_INSPECT_OPENAI_PYTHON``: an Inspect worker with the ``openai``
package, which is outside the verified pin set.
"""
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest
from aiq_evals_support import (
    HAS_DOCKER,
    OLMO_PYTHON,
    evaluate,
    evaluation_node,
    evaluations,
    needs,
    needs_olmo,
    needs_repo,
    repo_on_worker_path,
    require_magnet_evals,
    write_recipe,
)

magnet_evals = require_magnet_evals()

DEAD_URL = 'http://127.0.0.1:9/v1'
INSPECT_OPENAI_PYTHON = os.environ.get('AIQ_EVALS_INSPECT_OPENAI_PYTHON')


INFER_STACK = shutil.which('infer-stack')
needs_infer_stack = needs(INFER_STACK is not None, 'needs infer-stack on PATH (aiq-magnet[leasing])')


@pytest.fixture
def lease_env(tmp_path, monkeypatch):
    """An isolated infer-stack (null backend, private ledger) and the example endpoint."""
    from magnet_evals.examples.chat_server import chat_server

    catalog = tmp_path / 'catalog.yaml'
    catalog.write_text(
        'models:\n  example-model:\n    source: hf://example/model\n'
        'endpoints:\n'
        '  gpt-4o-mini:\n    model: example-model\n    engine: vllm\n'
        '  gpt-4o:\n    model: example-model\n    engine: vllm\n'
    )
    monkeypatch.setenv('INFER_STACK_BACKEND', 'null')
    monkeypatch.setenv('INFER_STACK_CATALOG', str(catalog))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg-data'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg-config'))
    monkeypatch.setenv('OPENAI_API_KEY', 'example-lease-key')
    monkeypatch.delenv('INFER_STACK_LEASE_ID', raising=False)
    with chat_server() as port:
        shim = tmp_path / 'shim-bin' / 'infer-stack'
        shim.parent.mkdir()
        shim.write_text(
            '#!/bin/bash\n'
            f'if [ "$1" = run ]; then shift; exec {INFER_STACK} run --base_url http://127.0.0.1:{port}/v1 "$@"; fi\n'
            f'exec {INFER_STACK} "$@"\n'
        )
        shim.chmod(0o755)
        monkeypatch.setenv('PATH', f'{shim.parent}{os.pathsep}{os.environ["PATH"]}')
        yield tmp_path / 'xdg-data' / 'infer_stack' / 'leasing' / 'ledger.db'


def _leases(ledger: Path) -> list[tuple[str, list[str]]]:
    if not ledger.exists():
        return []
    with sqlite3.connect(ledger) as conn:
        leases = [row[0] for row in conn.execute('select id from leases order by created_at')]
        return [
            (lease, sorted(r[0] for r in conn.execute('select endpoint from claims where lease_id = ?', (lease,))))
            for lease in leases
        ]


def _lease_settings():
    from magnet.leasing import LeaseSettings

    return LeaseSettings(enabled=True, allowed_gpus=False)


def _no_secret(out):
    for path in Path(out).rglob('*.json'):
        assert 'example-lease-key' not in path.read_text(errors='ignore'), path


@needs_infer_stack
@needs_olmo
def test_olmo_agent_runs_inside_a_real_lease_and_reuse_takes_none(tmp_path, lease_env):
    algo = {
        'engine': 'olmo_eval', 'task': 'aiq_example_tool', 'task_revision': 'example-v1',
        'data_revision': 'example-v1',
        'models': [{'role': 'primary', 'model': 'gpt-4o-mini', 'provider': 'litellm',
                    'revision': 'example-endpoint-v1', 'provider_options': {'base_url': DEAD_URL}}],
        'engine_options': {
            'upstream_revision': '73ade80e24f796af55caeb8fd7b75a7f3fd607fd',
            'task_modules': ['magnet_evals.examples.olmo_tasks'],
            'required_secrets': ['OPENAI_API_KEY'],
            'harness_config': {'scaffold': 'openai_agents', 'tools': ['aiq_example_double'],
                               'scaffold_kwargs': {'enable_compaction': False}},
        },
    }
    selects = [{'metric': 'contains_42'}, {'metric': 'contains_42', 'task': 'aiq_example_tool'}]
    fpath = write_recipe(
        tmp_path, {'evaluate': evaluation_node(algo, worker=OLMO_PYTHON, endpoint='gpt-4o-mini')},
        claim='assert metrics.evaluate.score == 1.0',
        matrix={'evaluate.select': [json.dumps(s, sort_keys=True) for s in selects]},
    )
    out = tmp_path / 'out'
    _, card = evaluate(fpath, out, lease_settings=_lease_settings())
    assert card.result == 'VERIFIED', [c.evidence_row.get('metrics.evaluate.ineligible_reasons') for c in card.cell_results]
    records = [json.loads(p.read_text()) for p in evaluations(out)]
    assert sorted(r['action'] for r in records) == ['executed', 'reused']
    # One lease for the measurement: the second node's gate found the stored run.
    assert _leases(lease_env) == [(_leases(lease_env)[0][0], ['gpt-4o-mini'])]
    _no_secret(out)

    # Rescheduling reuses everything: no new lease.
    evaluate(fpath, out, lease_settings=_lease_settings())
    assert len(_leases(lease_env)) == 1

    # Only the endpoint URL changes: a new node (its parameters differ), the
    # same measurement (identity v3). Its gate reuses the run: no lease, and
    # the evidence is valid.
    algo['models'][0]['provider_options'] = {'base_url': 'http://127.0.0.1:8/v1'}
    moved = write_recipe(
        tmp_path, {'evaluate': evaluation_node(algo, worker=OLMO_PYTHON, endpoint='gpt-4o-mini')},
        claim='assert metrics.evaluate.score == 1.0', name='moved_endpoint',
        matrix={'evaluate.select': [json.dumps(selects[0], sort_keys=True)]},
    )
    _, moved_card = evaluate(moved, out, lease_settings=_lease_settings())
    assert moved_card.result == 'VERIFIED'
    records = [json.loads(p.read_text()) for p in evaluations(out)]
    assert len(records) == 3 and sorted(r['action'] for r in records) == ['executed', 'reused', 'reused']
    assert len({r['measurement_identity']['digest'] for r in records}) == 1
    assert len(_leases(lease_env)) == 1


@needs_infer_stack
@needs_repo
@needs(INSPECT_OPENAI_PYTHON is not None, 'needs $AIQ_EVALS_INSPECT_OPENAI_PYTHON (Inspect with openai)')
def test_inspect_primary_and_grader_share_one_multi_endpoint_lease(tmp_path, lease_env, monkeypatch):
    repo_on_worker_path(monkeypatch)
    algo = {
        'engine': 'inspect_ai', 'task': 'python:tests.native.inspect_fixture:role_task',
        'data_revision': 'fixture-v1',
        'models': [
            {'role': 'primary', 'model': 'gpt-4o-mini', 'provider': 'openai', 'revision': 'example-endpoint-v1',
             'provider_options': {'base_url': DEAD_URL, 'responses_api': False}},
            {'role': 'grader', 'model': 'gpt-4o', 'provider': 'openai', 'revision': 'example-endpoint-v1'},
        ],
        'engine_options': {'registration_modules': ['tests.native.inspect_fixture'],
                           'required_secrets': ['OPENAI_API_KEY']},
        'select': {'task': 'role_task', 'scorer': 'match', 'metric': 'accuracy', 'reducer': 'mean'},
    }
    node = evaluation_node(algo, worker=INSPECT_OPENAI_PYTHON, endpoint='gpt-4o-mini',
                           endpoints={'grader': 'gpt-4o'})
    fpath = write_recipe(tmp_path, {'evaluate': node}, claim='assert metrics.evaluate.eligible')
    out = tmp_path / 'out'
    _, card = evaluate(fpath, out, lease_settings=_lease_settings())
    assert card.result == 'VERIFIED', [c.evidence_row.get('metrics.evaluate.ineligible_reasons') for c in card.cell_results]
    (record,) = [json.loads(p.read_text()) for p in evaluations(out)]
    assert record['action'] == 'executed'
    run = magnet_evals.load_run(record['run_path'])
    assert run.attempt['execution_context']['model_endpoint_roles'] == ['grader', 'primary']
    # Both roles came from one lease holding both endpoints.
    ((_, endpoints),) = _leases(lease_env)
    assert endpoints == ['gpt-4o', 'gpt-4o-mini']
    _no_secret(out)


CONTAINER_VENV = os.environ.get('MAGNET_TEST_CONTAINER_VENV')


def _venv_base(venv: Path) -> Path:
    home = next(line.split('=', 1)[1].strip() for line in (venv / 'pyvenv.cfg').read_text().splitlines()
                if line.split('=', 1)[0].strip() == 'home')
    return Path(home).parent


@needs_infer_stack
@needs(HAS_DOCKER and CONTAINER_VENV is not None and INSPECT_OPENAI_PYTHON is not None,
       'needs MAGNET_TEST_DOCKER=1, $MAGNET_TEST_CONTAINER_VENV, and $AIQ_EVALS_INSPECT_OPENAI_PYTHON')
def test_a_leased_container_verifies_the_served_model(tmp_path, lease_env, monkeypatch):
    # The alias differs from the model it serves. run_node runs inside the
    # container and can verify the lease only if the lease's
    # INFER_STACK_ENDPOINT_<ALIAS> variable is forwarded into it.
    import magnet
    from magnet.containers import ContainerSettings

    catalog = Path(os.environ['INFER_STACK_CATALOG'])
    catalog.write_text(
        'models:\n  example-model:\n    source: hf://example/model\n'
        'endpoints:\n  example-lease:\n    model: example-model\n    engine: vllm\n'
        '    served_name: gpt-4o-mini\n'
    )
    assert INSPECT_OPENAI_PYTHON is not None and CONTAINER_VENV is not None
    worker_venv, container_venv = Path(INSPECT_OPENAI_PYTHON).parent.parent, Path(CONTAINER_VENV)
    mounts = sorted({
        str(_venv_base(container_venv)), str(_venv_base(worker_venv)), str(container_venv),
        str(Path(magnet.__file__).resolve().parents[1]), str(Path(magnet_evals.__file__).resolve().parents[1]),
        str(tmp_path),
    })
    settings = ContainerSettings.coerce(
        image=os.environ.get('MAGNET_TEST_CONTAINER_IMAGE', 'ubuntu:24.04'), mounts=mounts,
        env={'PATH': f'{container_venv}/bin:/usr/bin:/bin'},
        docker_args=f'-v {worker_venv}:/opt/aiq-inspect-worker:ro',
    )
    algo = {
        'engine': 'inspect_ai', 'task': 'python:magnet_evals.examples.inspect_tasks:tool_task',
        'data_revision': 'example-v1',
        'models': [{'role': 'primary', 'model': 'gpt-4o-mini', 'provider': 'openai',
                    'revision': 'example-endpoint-v1',
                    'provider_options': {'base_url': DEAD_URL, 'responses_api': False}}],
        'engine_options': {'registration_modules': ['magnet_evals.examples.inspect_tasks'],
                           'required_secrets': ['OPENAI_API_KEY']},
        'select': {'scorer': 'match', 'metric': 'accuracy'},
    }
    node = evaluation_node(algo, worker='/opt/aiq-inspect-worker/bin/python', endpoint='example-lease')
    fpath = write_recipe(tmp_path, {'evaluate': node}, claim='assert metrics.evaluate.score == 1.0')
    out = tmp_path / 'out'
    _, card = evaluate(fpath, out, lease_settings=_lease_settings(), container_settings=settings)
    assert card.result == 'VERIFIED', [c.evidence_row.get('metrics.evaluate.ineligible_reasons') for c in card.cell_results]
    (record,) = [json.loads(p.read_text()) for p in evaluations(out)]
    assert record['action'] == 'executed'
    assert _leases(lease_env)[0][1] == ['example-lease']
    _no_secret(out)

