"""CPU/Docker acceptance: real Harbor workers and real null-backend leases.

The scripted endpoint tests transport and scheduling, not model capability.
Scientific serving facts describe the configured catalog, without asserting
that the null backend loaded model weights.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from aiq_evals_support import (
    HAS_DOCKER,
    REQUIRED,
    evaluate,
    evaluation_node,
    evaluations,
    recipe_rows,
    repo,
    repo_on_worker_path,
    write_recipe,
)

from magnet.backends.aiq_evals.pipeline import EvaluationNode, preflight_scope
from magnet.backends.aiq_evals.serving import (
    bind_provenance,
    catalog_provenance,
)
from magnet.leasing import LeaseSettings

WORKER = os.environ.get('AIQ_EVALS_HARBOR_PYTHON')
INFER = shutil.which('infer-stack')
AVAILABLE = HAS_DOCKER and bool(WORKER) and bool(INFER)
pytestmark = pytest.mark.skipif(not AVAILABLE and not REQUIRED,
                              reason='needs Docker, infer-stack, and AIQ_EVALS_HARBOR_PYTHON')


@pytest.fixture(autouse=True)
def prerequisites():
    assert AVAILABLE, 'required Docker/Harbor/infer-stack prerequisite is missing'


@pytest.fixture(scope='module')
def pinned_tasks(tmp_path_factory):
    root = repo()
    code = '''
import asyncio, subprocess
from tests.native.harbor_fixture import TASKS
from harbor.environments.docker.utils import ensure_docker_image_built, default_docker_platform
async def main():
    source = TASKS / 'division/environment'
    image = await ensure_docker_image_built(docker_name='division', docker_build_context=source,
        dockerfile_path=source / 'Dockerfile', build_args={}, platform=await default_docker_platform())
    print(subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip())
asyncio.run(main())
'''
    env = dict(os.environ, PYTHONPATH=str(root), HARBOR_TELEMETRY='0')
    output = subprocess.check_output([WORKER, '-c', code], env=env, text=True)
    image = output.strip().splitlines()[-1]
    task = tmp_path_factory.mktemp('magnet-harbor-tasks') / 'division'
    shutil.copytree(root / 'tests/native/harbor_tasks/division', task)
    config = task / 'task.toml'
    config.write_text(config.read_text().replace('[environment]\n',
                      '[environment]\ndocker_image = ' + json.dumps(image) + '\n'))
    return task.parent


@pytest.fixture
def lease(tmp_path, monkeypatch, request):
    repo_on_worker_path(monkeypatch)
    monkeypatch.setenv('HARBOR_TELEMETRY', '0')
    monkeypatch.setenv('INFER_STACK_BACKEND', 'null')
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('OPENAI_API_KEY', 'magnet-fixture-secret')
    monkeypatch.delenv('INFER_STACK_LEASE_ID', raising=False)
    monkeypatch.delenv('INFER_STACK_ENDPOINT_DESCRIPTOR', raising=False)
    catalog = tmp_path / 'catalog.yaml'
    catalog.write_text(yaml.safe_dump({
        'models': {'weights': {'source': 'hf://example/model', 'revision': 'a' * 40, 'quantization': 'Q4'}},
        'endpoints': {'fixture-coding': {'model': 'weights', 'engine': 'vllm',
            'runtime': {'image': 'example/vllm@sha256:' + 'b' * 64, 'max_model_len': 8192}}}}))
    monkeypatch.setenv('INFER_STACK_CATALOG', str(catalog))
    sys.path.insert(0, str(repo()))
    from tests.native.scripted_endpoint import scripted_endpoint

    patch = repo() / 'tests/native/harbor_tasks/division/solution/model.patch'
    commands = [
        'cd /app && cat calc.py',
        'cd /app && python -c ' + shlex.quote(
            "from pathlib import Path; Path('calc.py').write_text("
            "'def divide(numerator, denominator):\\n    if denominator == 0:\\n"
            "        raise ValueError(\"zero denominator\")\\n    return numerator / denominator\\n')"),
        'cd /app && python -m pytest -q test_calc.py',
    ]
    with scripted_endpoint(commands, error=bool(getattr(request, 'param', False))) as (url, calls):
        shim = tmp_path / 'bin/infer-stack'
        shim.parent.mkdir()
        entries = tmp_path / 'lease-entries'
        shim.write_text('#!/bin/bash\n'
            'if [ "$1" = run ]; then\n'
            f' echo entered >> {shlex.quote(str(entries))}\n'
            ' shift\n'
            f' exec {shlex.quote(INFER)} run --base_url {shlex.quote(url)} "$@"\n'
            'fi\n'
            f'exec {shlex.quote(INFER)} "$@"\n')
        shim.chmod(0o755)
        monkeypatch.setenv('PATH', str(shim.parent) + os.pathsep + os.environ['PATH'])
        yield {'catalog': catalog, 'calls': calls, 'entries': entries, 'shim': shim,
               'patch': patch, 'url': url, 'root': tmp_path}


def algo(tasks):
    return {'engine': 'harbor', 'task': 'path:' + str(tasks), 'data_revision': 'synthetic-v1',
        'models': [{'role': 'primary', 'model': 'fixture-coding', 'provider': 'openai',
                    'revision': 'fixture-endpoint', 'provider_options': {'base_url': 'http://127.0.0.1:9/v1'}}],
        'task_options': {'agent': 'python:tests.native.harbor_probe_agent:ProbeAgent',
                         'agent_kwargs': {'targets': {'allowed': 'http://placeholder.invalid/v1/models'}}},
        'select': {'metric': 'mean', 'score': 'reward'}}


def released(lease, count):
    path = lease['root'] / 'data/infer_stack/leasing/ledger.db'
    with sqlite3.connect(path) as db:
        states = [row[0] for row in db.execute('select state from leases')]
    assert states == ['released'] * count


def records(out):
    return [json.loads(path.read_text()) for path in evaluations(out)]


def container_settings(lease, tasks):
    import magnet
    from magnet.containers import ContainerSettings

    worker = Path(WORKER).parent.parent
    orchestrator = Path(sys.executable).parent.parent
    def base(venv):
        home = next(line.split('=', 1)[1].strip() for line in (venv / 'pyvenv.cfg').read_text().splitlines()
                    if line.startswith('home'))
        return str(Path(home).parent)
    # The worker path is deliberately absent from the scheduling host.
    mount = '/opt/aiq-harbor-worker'
    assert not Path(mount).exists()
    return ContainerSettings.coerce(image=os.environ.get('MAGNET_TEST_CONTAINER_IMAGE', 'ubuntu:24.04'),
        mounts=sorted({base(worker), base(orchestrator), str(orchestrator), str(worker),
            str(Path(magnet.__file__).resolve().parents[1]), str(repo()), str(tasks.parent),
            str(lease['root']), '/var/run/docker.sock', '/usr/bin/docker',
            '/usr/libexec/docker/cli-plugins'}),
        env={'PATH': f'{orchestrator}/bin:/usr/bin:/bin'},
        docker_args=f'--group-add {Path("/var/run/docker.sock").stat().st_gid} -v {worker}:{mount}:ro'), mount + '/bin/python'


@pytest.mark.parametrize('containerized', [False, True])
def test_harbor_scheduler_lease_reuse_and_evidence(tmp_path, lease, pinned_tasks, containerized):
    options = {'lease_settings': LeaseSettings(enabled=True, allowed_gpus=False)}
    worker = WORKER
    if containerized:
        settings, worker = container_settings(lease, pinned_tasks)
        options['container_settings'] = settings
    selects = [{'metric': 'mean', 'score': 'reward'},
               {'metric': 'mean', 'score': 'reward', 'task': 'path:' + str(pinned_tasks)}]
    path = write_recipe(tmp_path, {'evaluate': evaluation_node(algo(pinned_tasks), worker=worker,
                        endpoint='fixture-coding')}, claim='assert metrics.evaluate.score == 1.0',
                        matrix={'evaluate.select': [json.dumps(s) for s in selects]})
    out = tmp_path / 'out'
    recipe, card = evaluate(path, out, **options)
    assert card.result == 'VERIFIED', [c.evidence_row for c in card.cell_results]
    found = records(out)
    assert sorted(r['action'] for r in found) == ['executed', 'reused']
    assert all(r['measurement_identity']['reusable'] for r in found)
    assert all(r['request']['models'][0]['cache_token'] == r['endpoint_provenance']['primary']['digest']
               for r in found)
    assert len(lease['calls']) == 4
    assert lease['entries'].read_text().splitlines() == ['entered']
    released(lease, 1)
    rows = recipe_rows(recipe)
    assert len(rows) == 2 and all(r['row']['metrics.evaluate.score'] == 1 for r in rows)
    for record in found:
        import magnet_evals

        run = magnet_evals.load_run(record['run_path'])
        sample = run.result.samples[0]
        assert (run.path / sample.native['patch_artifact']).read_bytes() == lease['patch'].read_bytes()
        assert record['preflight_identity'] == run.resolved.identity.digest
    evaluate(path, out, **options)
    assert len(lease['calls']) == 4 and lease['entries'].read_text().splitlines() == ['entered']
    for file in out.rglob('*.json'):
        assert 'magnet-fixture-secret' not in file.read_text(errors='ignore')
    if capture := os.environ.get('AIQ_MAGNET_HARBOR_CAPTURE_DIR'):
        destination = Path(capture) / ('container' if containerized else 'host')
        shutil.copytree(run.path, destination / 'run')
        (destination / 'observation.json').write_text(json.dumps({
            'schema': 'magnet-harbor-native-observation/1',
            'backend': 'null', 'containerized_worker': containerized,
            'native_inferences': len(lease['calls']), 'leases_acquired': 1,
            'leases_released': 1, 'actions': sorted(r['action'] for r in found),
            'rows': len(rows), 'endpoint_provenance': found[0]['endpoint_provenance'],
            'api_key_retained': False, 'preflight_matches_execution': True,
        }, indent=2) + '\n')


def test_quantization_invalidates_same_alias_measurement(tmp_path, lease, pinned_tasks):
    path = write_recipe(tmp_path, {'evaluate': evaluation_node(algo(pinned_tasks), worker=WORKER,
                        endpoint='fixture-coding')}, claim='assert metrics.evaluate.score == 1.0')
    out = tmp_path / 'out'
    options = {'lease_settings': LeaseSettings(enabled=True, allowed_gpus=False)}
    assert evaluate(path, out, **options)[1].result == 'VERIFIED'
    before = records(out)[0]
    value = yaml.safe_load(lease['catalog'].read_text())
    value['models']['weights']['quantization'] = 'Q5'
    lease['catalog'].write_text(yaml.safe_dump(value))
    assert evaluate(path, out, **options)[1].result == 'VERIFIED'
    found = records(out)
    assert len(found) == 2 and all(r['action'] == 'executed' for r in found)
    assert len({r['measurement_identity']['digest'] for r in found}) == 2
    assert len({r['request']['models'][0]['cache_token'] for r in found}) == 2
    assert len({r['request']['models'][0]['model'] for r in found}) == 1
    assert before['request']['models'][0]['cache_token'] in {
        r['request']['models'][0]['cache_token'] for r in found}
    assert len(lease['calls']) == 8
    released(lease, 2)


def compiled_node(tmp_path, tasks):
    node = EvaluationNode(algo_params=algo(tasks), perf_params={'worker_python': WORKER,
        'endpoint': 'fixture-coding', 'store_dpath': str(tmp_path / 'store')})
    node.apply_lease_settings(LeaseSettings(enabled=True, allowed_gpus=False))
    node.configure({'out_dpath': str(tmp_path / 'node'),
                    'evaluation_fname': str(tmp_path / 'node/evaluation.json')})
    with preflight_scope(True):
        return node.command


@pytest.mark.parametrize('change_at', ['catalog', 'lease'])
def test_stale_serving_facts_abort_before_inference(tmp_path, lease, pinned_tasks, change_at):
    command = compiled_node(tmp_path, pinned_tasks)
    changed = yaml.safe_load(lease['catalog'].read_text())
    changed['models']['weights']['quantization'] = 'Q5'
    if change_at == 'catalog':
        lease['catalog'].write_text(yaml.safe_dump(changed))
    else:
        changed_path = tmp_path / 'q5.yaml'
        changed_path.write_text(yaml.safe_dump(changed))
        shim = lease['shim']
        shim.write_text(shim.read_text().replace(' shift\n',
            ' shift\n' + f' cp {shlex.quote(str(changed_path))} {shlex.quote(str(lease["catalog"]))}\n'))
    process = subprocess.run(['bash', '-c', command], capture_output=True, text=True)
    assert process.returncode == 3, process.stdout + process.stderr
    assert not lease['calls'] and not (tmp_path / 'node/evaluation.json').exists()
    summary = json.loads((tmp_path / 'node/attempt_summary.json').read_text())
    assert 'serving provenance' in summary['error']
    assert not list((tmp_path / 'store').rglob('RUN_COMPLETE'))
    if change_at == 'lease':
        released(lease, 1)
    else:
        assert not lease['entries'].exists()


def test_uncertain_serving_inputs_cannot_become_reusable(tmp_path, lease, pinned_tasks):
    value = yaml.safe_load(lease['catalog'].read_text())
    value['endpoints']['fixture-coding']['runtime']['image'] = 'example/vllm:latest'
    lease['catalog'].write_text(yaml.safe_dump(value))
    expected = catalog_provenance({'primary': 'fixture-coding'})
    request = bind_provenance(algo(pinned_tasks), expected)
    assert request['models'][0]['revision'] == 'fixture-endpoint'
    assert request['models'][0]['cache_token'] == expected['primary']['digest']
    # This is the same isolated worker resolver used by preflight.
    from magnet.backends.aiq_evals.pipeline import run_preflight

    node = EvaluationNode(perf_params={'worker_python': WORKER})
    resolution = run_preflight(node.preflight_command({**algo(pinned_tasks), 'endpoint_provenance': expected}))
    assert not resolution['measurement_identity']['reusable']
    assert any('serving image' in reason for reason in resolution['measurement_identity']['unknown_reasons'])
    assert not lease['calls'] and not lease['entries'].exists()


def test_conflicting_explicit_token_requires_opt_out(lease, pinned_tasks):
    expected = catalog_provenance({'primary': 'fixture-coding'})
    request = algo(pinned_tasks)
    request['models'][0]['cache_token'] = 'user-assertion'
    with pytest.raises(ValueError, match='conflicts'):
        bind_provenance(request, expected)
    assert bind_provenance(request, {}) == request


def test_concurrent_selectors_share_one_native_execution(tmp_path, lease, pinned_tasks):
    commands = [compiled_node(tmp_path / str(i), pinned_tasks) for i in range(2)]
    # Both gates acquire from the same store, but write separate node artifacts.
    commands = [command.replace(str(tmp_path / str(i) / 'store'), str(tmp_path / 'shared-store'))
                for i, command in enumerate(commands)]
    processes = [subprocess.Popen(['bash', '-c', command], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True) for command in commands]
    for process in processes:
        stdout, stderr = process.communicate(timeout=240)
        assert process.returncode == 0, stdout + stderr
    found = [json.loads((tmp_path / str(i) / 'node/evaluation.json').read_text()) for i in range(2)]
    assert sorted(r['action'] for r in found) == ['executed', 'reused']
    assert len({r['run_path'] for r in found}) == 1 and len(lease['calls']) == 4
    assert lease['entries'].read_text().splitlines() == ['entered']
    released(lease, 1)


@pytest.mark.parametrize('lease', [True], indirect=True)
def test_infrastructure_failure_produces_no_false_score(tmp_path, lease, pinned_tasks):
    path = write_recipe(tmp_path, {'evaluate': evaluation_node(algo(pinned_tasks), worker=WORKER,
                        endpoint='fixture-coding')}, claim='assert metrics.evaluate.score > 0.5')
    out = tmp_path / 'out'
    _, card = evaluate(path, out, lease_settings=LeaseSettings(enabled=True, allowed_gpus=False))
    assert card.result != 'FALSIFIED' and not card.cell_results
    assert card.requested_work['attempt_status'] == {'failed': 1}
    assert not evaluations(out)
    assert lease['calls'] and not list(out.rglob('RUN_COMPLETE'))
    released(lease, 1)


def test_native_pro_replay_failure_is_ineligible(tmp_path):
    from magnet_evals import ExecutionContext, load_run
    from magnet_evals.artifacts import publish_run
    from magnet_evals.benchmarks.swe_bench_pro import normalize_pro_jobs

    from magnet.backends.aiq_evals.projection import evidence_view, flat_metrics

    original = load_run(repo() / 'tests/fixtures/swebench-pro-native/oracle')
    source = tmp_path / 'native'
    shutil.copytree(original.path / 'native', source)
    # A retained native reward does not prove that required tests completed.
    next((source / 'harbor/replay').glob('*/verifier/output.json')).unlink()
    result = normalize_pro_jobs(source / 'harbor', identity=original.resolved.identity,
                                fallback_task=original.resolved.request.task)
    context = ExecutionContext(output_dir=tmp_path / 'incomplete')
    bundle = publish_run(context.output_dir, resolved=original.resolved, result=result,
                         context=context, native_dir=source)
    view = evidence_view(bundle, {'metric': 'mean', 'score': 'reward'})
    assert result.status == 'incomplete' and not bundle.complete
    assert not view.eligible and 'score' not in flat_metrics(view.to_dict())
    assert result.samples[0].native['native_rewards'] == {'reward': 1.0}
    assert not result.samples[0].native['counted_as_model_failure']
