"""Opt-in real-GPU aiq-magnet-evals integration acceptance.

This gate exercises three production-shaped evaluation paths against an actual
infer-stack Compose/vLLM endpoint on a physical GPU:

* Inspect one-shot generation;
* Inspect agentic tool execution; and
* OLMo Eval agentic tool execution through its OpenAI Agents scaffold.

Every request contains a deliberately dead ``base_url``.  Completion therefore
requires MAGNET's infer-stack lease to inject the real operational endpoint into
the Dockerized evaluation node.  Agentic cases additionally require normalized
trajectory evidence that a tool actually executed; merely configuring a tool or
starting an agent scaffold is not sufficient.

Model benchmark scores are diagnostic rather than infrastructure pass criteria.
The agentic contract is stronger: the evaluation must be eligible *and* contain
an executed tool turn in the normalized sample trajectory.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

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
OLMO_REVISION = '73ade80e24f796af55caeb8fd7b75a7f3fd607fd'


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


def _executed_tool_payloads(value: Any) -> list[Any]:
    """Find normalized tool-result payloads without depending on one engine shape."""
    found: list[Any] = []
    if isinstance(value, dict):
        if value.get('role') == 'tool':
            found.append(value)
        tool_results = value.get('tool_results')
        if tool_results:
            found.append(tool_results)
        for child in value.values():
            found.extend(_executed_tool_payloads(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_executed_tool_payloads(child))
    return found


@pytest.fixture(scope='module')
def real_gpu_context():
    """Validate host prerequisites once and expose the real controller state."""
    infer_stack = shutil.which('infer-stack')
    assert infer_stack is not None, 'infer-stack is required on PATH'
    assert shutil.which('docker') is not None, 'docker is required on PATH'
    assert shutil.which('nvidia-smi') is not None, 'nvidia-smi is required on PATH'

    subprocess.run(['docker', 'info'], check=True, stdout=subprocess.DEVNULL)
    gpu_listing = _run_text('nvidia-smi', '-L')
    assert gpu_listing.strip(), 'nvidia-smi reported no GPUs'

    status_before = _run_text('infer-stack', 'status')
    print('\n=== infer-stack status before real-GPU evaluations ===')
    print(status_before)

    backend = _status_value(status_before, 'backend')
    assert backend == 'compose', (
        f'real-GPU acceptance requires infer-stack backend=compose, got {backend!r}'
    )

    data_dir_text = _status_value(status_before, 'data dir')
    assert data_dir_text, f'could not discover infer-stack data dir from:\n{status_before}'
    ledger = Path(data_dir_text) / 'leasing' / 'ledger.db'

    endpoint = os.environ.get('MAGNET_REAL_GPU_ENDPOINT', 'qwen2.5-7b')
    agentic_endpoint = os.environ.get(
        'MAGNET_REAL_GPU_AGENTIC_ENDPOINT',
        f'{endpoint}-agentic-e2e',
    )
    model_revision = os.environ.get(
        'MAGNET_REAL_GPU_MODEL_REVISION',
        'Qwen/Qwen2.5-7B-Instruct',
    )

    catalog_text = _run_text('infer-stack', 'catalog', 'show')
    for required in (endpoint, agentic_endpoint):
        assert re.search(rf'(?m)^\s*{re.escape(required)}:\s*$', catalog_text), (
            f'infer-stack catalog does not contain endpoint {required!r}; '
            'run dev/ci/aiq_evals_real_gpu.sh so its temporary agentic alias is prepared'
        )

    configured_output = os.environ.get('MAGNET_REAL_GPU_OUTPUT_DIR')
    assert configured_output, 'MAGNET_REAL_GPU_OUTPUT_DIR is required'
    root = Path(configured_output)
    root.mkdir(parents=True, exist_ok=True)

    container_venv_text = os.environ.get('MAGNET_TEST_CONTAINER_VENV')
    inspect_python_text = os.environ.get('AIQ_EVALS_INSPECT_OPENAI_PYTHON')
    olmo_python_text = os.environ.get('AIQ_EVALS_OLMO_PYTHON')
    assert container_venv_text, 'MAGNET_TEST_CONTAINER_VENV is required'
    assert inspect_python_text, 'AIQ_EVALS_INSPECT_OPENAI_PYTHON is required'
    assert olmo_python_text, 'AIQ_EVALS_OLMO_PYTHON is required'

    return {
        'ledger': ledger,
        'endpoint': endpoint,
        'agentic_endpoint': agentic_endpoint,
        'model_revision': model_revision,
        'root': root,
        'container_venv': Path(container_venv_text),
        'inspect_python': Path(inspect_python_text),
        'olmo_python': Path(olmo_python_text),
    }


def _container_settings(ctx: dict[str, Any], worker_python: Path, label: str):
    """Build the same Dockerized node shape for either native evaluator worker."""
    import magnet
    from magnet.containers import ContainerSettings

    worker_venv = worker_python.parent.parent
    container_venv = ctx['container_venv']
    assert worker_python.exists(), worker_python
    assert (container_venv / 'bin' / 'python').exists(), container_venv

    mounts = {
        str(_python_base(container_venv)),
        str(_python_base(worker_venv)),
        str(container_venv),
        str(Path(magnet.__file__).resolve().parents[1]),
        str(Path(magnet_evals.__file__).resolve().parents[1]),
        str(ctx['root']),
    }

    # OLMo Eval is installed editable by ``uv sync``.  Its .venv therefore
    # points back into the checkout; mount that source tree at the same absolute
    # path in addition to mapping the venv to a stable /opt path.
    if label == 'olmo':
        mounts.add(str(worker_venv.parent))

    inside_venv = f'/opt/aiq-{label}-worker'
    return ContainerSettings.coerce(
        image=os.environ.get('MAGNET_TEST_CONTAINER_IMAGE', 'ubuntu:24.04'),
        mounts=sorted(mounts),
        env={'PATH': f'{container_venv}/bin:/usr/bin:/bin'},
        docker_args=f'-v {worker_venv}:{inside_venv}:ro',
    ), f'{inside_venv}/bin/python'


def _run_case(
    ctx: dict[str, Any],
    *,
    name: str,
    endpoint: str,
    algo: dict[str, Any],
    worker_python: Path,
    worker_label: str,
    require_tool_execution: bool,
):
    """Run one fresh leased measurement and assert its real-hardware artifacts."""
    from magnet.leasing import LeaseSettings

    root = ctx['root'] / name
    root.mkdir(parents=True, exist_ok=True)
    out = root / 'out'

    container_settings, inside_worker = _container_settings(
        ctx, worker_python, worker_label,
    )
    node = evaluation_node(
        algo,
        worker=inside_worker,
        endpoint=endpoint,
    )
    recipe_fpath = write_recipe(
        root,
        {'evaluate': node},
        # Score correctness belongs to model-quality evaluation.  This gate
        # requires a complete eligible native run; agentic cases separately
        # require evidence that a tool actually executed.
        claim='assert metrics.evaluate.eligible',
        name=f'aiq_evals_real_gpu_{name}',
    )

    before_leases = _read_leases(ctx['ledger'])
    _, card = evaluate(
        recipe_fpath,
        out,
        lease_settings=LeaseSettings(enabled=True, allowed_gpus=True),
        container_settings=container_settings,
    )

    print(f'\n=== {name}: native evaluator diagnostics ===')
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
    assert run.result.status == 'succeeded'
    assert run.result.samples, 'native evaluator returned no normalized samples'

    if require_tool_execution:
        tool_payloads = [
            payload
            for sample in run.result.samples
            for payload in _executed_tool_payloads(sample.trajectory)
        ]
        print(f'\n=== {name}: normalized agent trajectory ===')
        print(json.dumps(
            [sample.trajectory for sample in run.result.samples],
            indent=2,
            default=str,
        ))
        assert tool_payloads, (
            f'{name} completed but no executed tool turn was present in the '
            'normalized trajectory; this is not an agentic pass'
        )

    after_first = _read_leases(ctx['ledger'])
    new_lease_ids = [
        lease_id
        for lease_id in after_first
        if lease_id not in before_leases
    ]
    new_claims = {
        lease_id: after_first[lease_id]
        for lease_id in new_lease_ids
    }
    print(f'\n=== {name}: new infer-stack leases ===')
    print(json.dumps(new_claims, indent=2))

    matching = [
        lease_id
        for lease_id, endpoints in new_claims.items()
        if endpoint in endpoints
    ]
    assert matching, (
        f'no new infer-stack lease claimed {endpoint!r}; new leases={new_claims!r}'
    )

    return run


@needs_real_gpu
def test_inspect_real_gpu_generation(real_gpu_context):
    """One-shot Inspect benchmark path through real vLLM/GPU serving."""
    ctx = real_gpu_context
    endpoint = ctx['endpoint']
    algo = {
        'engine': 'inspect_ai',
        'task': 'python:magnet_evals.examples.inspect_tasks:generation',
        'task_revision': None,
        'data_revision': f'real-gpu-inspect-generation-{endpoint}-v2',
        'models': [{
            'role': 'primary',
            'model': endpoint,
            'provider': 'openai',
            'revision': ctx['model_revision'],
            'provider_options': {
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
    _run_case(
        ctx,
        name='inspect_generation',
        endpoint=endpoint,
        algo=algo,
        worker_python=ctx['inspect_python'],
        worker_label='inspect',
        require_tool_execution=False,
    )


@needs_real_gpu
def test_inspect_real_gpu_agent_executes_tool(real_gpu_context):
    """Inspect must execute an actual tool turn against the real GPU model."""
    ctx = real_gpu_context
    endpoint = ctx['agentic_endpoint']
    algo = {
        'engine': 'inspect_ai',
        'task': 'python:aiq_evals_real_gpu_tasks:forced_tool_task',
        'task_revision': 'forced-tool-v1',
        'data_revision': f'real-gpu-inspect-agent-{endpoint}-v1',
        'models': [{
            'role': 'primary',
            'model': endpoint,
            'provider': 'openai',
            'revision': ctx['model_revision'],
            'provider_options': {
                'base_url': DEAD_URL,
                'responses_api': False,
            },
        }],
        'task_options': {},
        'generation': {'temperature': 0.0},
        'engine_options': {
            'registration_modules': ['aiq_evals_real_gpu_tasks'],
            'required_secrets': ['OPENAI_API_KEY'],
        },
        'select': {
            'task': 'forced_tool_task',
            'scorer': 'match',
            'metric': 'accuracy',
        },
    }
    _run_case(
        ctx,
        name='inspect_agent',
        endpoint=endpoint,
        algo=algo,
        worker_python=ctx['inspect_python'],
        worker_label='inspect',
        require_tool_execution=True,
    )


@needs_real_gpu
def test_olmo_real_gpu_agent_executes_tool(real_gpu_context):
    """OLMo Eval/OpenAI Agents must execute a tool against the real GPU model."""
    ctx = real_gpu_context
    endpoint = ctx['agentic_endpoint']
    algo = {
        'engine': 'olmo_eval',
        'task': 'aiq_example_tool',
        'task_revision': 'example-v1',
        'data_revision': f'real-gpu-olmo-agent-{endpoint}-v1',
        'models': [{
            'role': 'primary',
            'model': endpoint,
            'provider': 'litellm',
            'revision': ctx['model_revision'],
            'provider_options': {'base_url': DEAD_URL},
        }],
        'engine_options': {
            'upstream_revision': OLMO_REVISION,
            'task_modules': ['magnet_evals.examples.olmo_tasks'],
            'required_secrets': ['OPENAI_API_KEY'],
            'harness_config': {
                'scaffold': 'openai_agents',
                'tools': ['aiq_example_double'],
                'max_turns': 4,
                'scaffold_kwargs': {'enable_compaction': False},
            },
        },
        'select': {
            'task': 'aiq_example_tool',
            'metric': 'contains_42',
        },
    }
    _run_case(
        ctx,
        name='olmo_agent',
        endpoint=endpoint,
        algo=algo,
        worker_python=ctx['olmo_python'],
        worker_label='olmo',
        require_tool_execution=True,
    )

