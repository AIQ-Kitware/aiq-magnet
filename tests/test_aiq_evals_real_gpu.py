"""Opt-in real-GPU aiq-magnet-evals integration acceptance.

This is intentionally outside the routine integration gate.  It proves the
full production-shaped path instead of the deterministic null-serving fixture:

    MAGNET EvaluationNode
      -> infer-stack ``run`` lease
      -> Compose/vLLM on a real GPU
      -> Dockerized MAGNET node
      -> aiq-magnet-evals Inspect worker
      -> real OpenAI-compatible generation
      -> native metric/evidence returned to MAGNET

The request contains a deliberately dead ``base_url``.  A successful eligible
measurement therefore requires the infer-stack lease environment to replace
that operational endpoint before the worker talks to the model.

The model's *score* is diagnostic, not a pass criterion.  Model behavior is not
an infrastructure invariant: the acceptance gate checks that a real model was
leased, reached, evaluated, and materialized as eligible evidence.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from aiq_evals_support import (
    evaluate,
    evaluation_node,
    evaluations,
    needs,
    require_magnet_evals,
    write_recipe,
)


magnet_evals = require_magnet_evals()

REAL_GPU_ENABLED = os.environ.get('MAGNET_TEST_REAL_GPU', '').strip().lower() in {
    '1', 'true', 'yes',
}
needs_real_gpu = needs(
    REAL_GPU_ENABLED,
    'needs MAGNET_TEST_REAL_GPU=1; run dev/ci/aiq_evals_real_gpu.sh',
)

DEAD_URL = 'http://127.0.0.1:9/v1'


def _run_text(*argv: str) -> str:
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT)


def _status_value(status: str, key: str) -> str | None:
    """Extract one ``infer-stack status`` ``key: value`` line."""
    prefix = key.strip().lower() + ':'
    for line in status.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            return stripped.split(':', 1)[1].strip()
    return None


def _read_leases(ledger: Path) -> dict[str, list[str]]:
    """Return lease id -> claimed endpoints from infer-stack's sqlite ledger."""
    if not ledger.exists():
        return {}
    with sqlite3.connect(ledger) as conn:
        lease_ids = [
            row[0]
            for row in conn.execute('select id from leases order by created_at')
        ]
        return {
            lease_id: sorted(
                row[0]
                for row in conn.execute(
                    'select endpoint from claims where lease_id = ?',
                    (lease_id,),
                )
            )
            for lease_id in lease_ids
        }


def _python_base(venv: Path) -> Path:
    """Base interpreter prefix needed when a uv venv is bind-mounted."""
    python = venv / 'bin' / 'python'
    if not python.exists():
        raise AssertionError(f'python does not exist: {python}')
    return python.resolve().parents[1]


def _metric_diagnostics(card) -> dict[str, object]:
    """Collect selected evaluator fields for useful ``pytest -s`` output."""
    rows = [result.evidence_row for result in card.cell_results]
    if not rows:
        return {'rows': 0}
    row = rows[0]
    return {
        key: value
        for key, value in sorted(row.items())
        if key.startswith('metrics.evaluate.')
    }


@needs_real_gpu
def test_aiq_evals_real_gpu_compose_end_to_end(tmp_path):
    """Run one real Inspect measurement through an infer-stack GPU lease."""
    import magnet
    from magnet.containers import ContainerSettings
    from magnet.leasing import LeaseSettings

    infer_stack = shutil.which('infer-stack')
    assert infer_stack is not None, 'infer-stack is required on PATH'
    assert shutil.which('docker') is not None, 'docker is required on PATH'
    assert shutil.which('nvidia-smi') is not None, 'nvidia-smi is required on PATH'

    subprocess.run(['docker', 'info'], check=True, stdout=subprocess.DEVNULL)
    gpu_listing = _run_text('nvidia-smi', '-L')
    assert gpu_listing.strip(), 'nvidia-smi reported no GPUs'

    status_before = _run_text('infer-stack', 'status')
    print('\n=== infer-stack status before real-GPU evaluation ===')
    print(status_before)

    backend = _status_value(status_before, 'backend')
    assert backend == 'compose', (
        f'real-GPU acceptance requires infer-stack backend=compose, got {backend!r}'
    )

    data_dir_text = _status_value(status_before, 'data dir')
    assert data_dir_text, f'could not discover infer-stack data dir from:\n{status_before}'
    data_dir = Path(data_dir_text)
    ledger = data_dir / 'leasing' / 'ledger.db'

    endpoint = os.environ.get('MAGNET_REAL_GPU_ENDPOINT', 'qwen2.5-7b')
    model_revision = os.environ.get(
        'MAGNET_REAL_GPU_MODEL_REVISION',
        'Qwen/Qwen2.5-7B-Instruct',
    )

    catalog_text = _run_text('infer-stack', 'catalog', 'show')
    assert re.search(rf'(?m)^\s*{re.escape(endpoint)}:\s*$', catalog_text), (
        f'infer-stack catalog does not contain endpoint {endpoint!r}; '
        'seed the host catalog or set MAGNET_REAL_GPU_ENDPOINT'
    )

    worker_python_text = os.environ.get('AIQ_EVALS_INSPECT_OPENAI_PYTHON')
    container_venv_text = os.environ.get('MAGNET_TEST_CONTAINER_VENV')
    assert worker_python_text, 'AIQ_EVALS_INSPECT_OPENAI_PYTHON is required'
    assert container_venv_text, 'MAGNET_TEST_CONTAINER_VENV is required'

    worker_python = Path(worker_python_text)
    worker_venv = worker_python.parent.parent
    container_venv = Path(container_venv_text)
    assert worker_python.exists(), worker_python
    assert (container_venv / 'bin' / 'python').exists(), container_venv

    configured_output = os.environ.get('MAGNET_REAL_GPU_OUTPUT_DIR')
    root = Path(configured_output) if configured_output else tmp_path / 'real-gpu'
    root.mkdir(parents=True, exist_ok=True)
    out = root / 'out'

    mounts = sorted({
        str(_python_base(container_venv)),
        str(_python_base(worker_venv)),
        str(container_venv),
        str(Path(magnet.__file__).resolve().parents[1]),
        str(Path(magnet_evals.__file__).resolve().parents[1]),
        str(root),
    })
    container_settings = ContainerSettings.coerce(
        image=os.environ.get('MAGNET_TEST_CONTAINER_IMAGE', 'ubuntu:24.04'),
        mounts=mounts,
        env={'PATH': f'{container_venv}/bin:/usr/bin:/bin'},
        docker_args=f'-v {worker_venv}:/opt/aiq-inspect-worker:ro',
    )

    algo = {
        'engine': 'inspect_ai',
        'task': 'python:magnet_evals.examples.inspect_tasks:generation',
        'task_revision': None,
        'data_revision': f'real-gpu-{endpoint}-v1',
        'models': [{
            'role': 'primary',
            'model': endpoint,
            'provider': 'openai',
            'revision': model_revision,
            'provider_options': {
                # Deliberately unusable.  The real infer-stack lease must
                # replace the operational endpoint for this run to succeed.
                'base_url': DEAD_URL,
                'responses_api': False,
            },
        }],
        'task_options': {},
        'generation': {'temperature': 0.0},
        'engine_options': {
            'registration_modules': ['magnet_evals.examples.inspect_tasks'],
            'required_secrets': ['OPENAI_API_KEY'],
        },
        'select': {
            'task': 'generation',
            'scorer': 'includes',
            'metric': 'accuracy',
        },
    }

    node = evaluation_node(
        algo,
        worker='/opt/aiq-inspect-worker/bin/python',
        endpoint=endpoint,
    )
    recipe_fpath = write_recipe(
        root,
        {'evaluate': node},
        # The integration contract is eligible real evidence, not whether a
        # particular model happens to get this tiny benchmark item correct.
        claim='assert metrics.evaluate.eligible',
        name='aiq_evals_real_gpu',
    )

    before_leases = _read_leases(ledger)

    _, card = evaluate(
        recipe_fpath,
        out,
        lease_settings=LeaseSettings(enabled=True, allowed_gpus=True),
        container_settings=container_settings,
    )

    print('\n=== native evaluator diagnostics ===')
    print(json.dumps(_metric_diagnostics(card), indent=2, default=str))

    assert card.result == 'VERIFIED', [
        result.evidence_row.get('metrics.evaluate.ineligible_reasons')
        for result in card.cell_results
    ]

    records = [json.loads(path.read_text()) for path in evaluations(out)]
    assert len(records) == 1, records
    record = records[0]
    assert record['action'] == 'executed', record
    assert record['status'] == 'succeeded', record
    assert Path(record['attempt_path']).exists(), record['attempt_path']
    assert Path(record['run_path']).exists(), record['run_path']

    run = magnet_evals.load_run(record['run_path'])
    execution_context = run.attempt['execution_context']
    assert execution_context['model_endpoint_roles'] == ['primary']

    after_first = _read_leases(ledger)
    new_lease_ids = [lease_id for lease_id in after_first if lease_id not in before_leases]
    new_claims = {lease_id: after_first[lease_id] for lease_id in new_lease_ids}
    print('\n=== new infer-stack leases ===')
    print(json.dumps(new_claims, indent=2))

    matching = [
        lease_id
        for lease_id, endpoints in new_claims.items()
        if endpoint in endpoints
    ]
    assert matching, (
        f'no new infer-stack lease claimed {endpoint!r}; new leases={new_claims!r}'
    )

    # A second scheduling of the same measurement should reuse the materialized
    # evaluator result and must not take another real GPU lease.
    _, second_card = evaluate(
        recipe_fpath,
        out,
        lease_settings=LeaseSettings(enabled=True, allowed_gpus=True),
        container_settings=container_settings,
    )
    assert second_card.result == 'VERIFIED'
    after_second = _read_leases(ledger)
    assert after_second == after_first, (
        'rescheduling an already materialized measurement took another lease'
    )

    print('\n=== real-GPU aiq-magnet-evals acceptance: VERIFIED ===')
