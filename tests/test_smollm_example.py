"""
The SmolLM example: what it renders, and what it computes.

Two halves, both hermetic. The first pins the *shape* of the rendered
campaign -- which nodes lease, which are containerized, and in which order the
two wrappers nest -- because that is what the example exists to demonstrate.
The second runs the three node executables end to end against a stub OpenAI
server, so the example is known to work without a GPU, a container or a model.
"""

import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import ubelt as ub
import yaml
from kwdagger.pipeline import coerce_pipeline

from magnet import containers, leasing
from magnet.containers import ContainerSettings
from smollm_example.cli.compare_answers import compare
from magnet.leasing import LeaseSettings

import shlex as _shlex
import sys as _sys
#: On the host route a bare ``python`` renders as this interpreter (magnet.containers.host_interpreter).
HOST_PY = _shlex.quote(_sys.executable)

#: `examples/`, so a subprocess can `python -m smollm_example.cli.*`. pytest
#: puts it on *this* process's path via `pythonpath` in pyproject.toml; a child
#: needs it in the environment, which is what `run.sh` does for a real run.
EXAMPLES_DPATH = str(ub.Path(__file__).parent.parent / 'examples')


def _child_env(**overrides):
    import os

    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [EXAMPLES_DPATH, *([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])]
    )
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


CARD_FPATH = (
    ub.Path(__file__).parent.parent
    / 'examples/smollm_example/smollm_kwdagger.yaml'
)
IMAGE = 'aiq-eval-node:latest'


@pytest.fixture
def card():
    return yaml.safe_load(CARD_FPATH.read_text())


def _configured_nodes(card, *, image=IMAGE, leasing_on=True):
    pipeline = coerce_pipeline(card['kwdagger']['pipeline'])
    containers.apply_settings(
        pipeline, ContainerSettings.coerce(image=image, mounts='/repo')
    )
    LeaseSettings(enabled=leasing_on).apply(pipeline)
    return pipeline.node_dict


# --- what the example demonstrates -----------------------------------------


def test_only_the_node_that_needs_a_model_leases(card):
    """The argument for per-node leasing, stated as a DAG.

    One lease around the evaluation would hold both SmolLM models while the
    dataset is written and while the comparison runs, neither of which can use
    one.
    """
    nodes = _configured_nodes(card)
    nodes['ask'].configure({'endpoint': 'smol-135'})
    nodes['items'].configure({})
    nodes['compare'].configure({})

    assert 'infer-stack run' in nodes['ask'].command
    assert 'infer-stack run' not in nodes['items'].command
    assert 'infer-stack run' not in nodes['compare'].command


def test_the_lease_is_outside_the_container(card):
    """Acquiring needs the host's daemon and ledger; consuming happens inside.

    Being inside is also what lets the container inherit OPENAI_BASE_URL and
    OPENAI_API_KEY from the lease with no extra plumbing.
    """
    node = _configured_nodes(card)['ask']
    node.configure({'endpoint': 'smol-135'})
    command = node.command
    assert command.index('infer-stack run') < command.index('docker run')
    assert '-e OPENAI_BASE_URL' in command
    assert '-e OPENAI_API_KEY' in command


def test_every_node_can_use_the_image(card):
    """A node that cannot be containerized takes --container_image and drops
    it, which is a green run that containerized nothing."""
    for name, node in _configured_nodes(card).items():
        assert isinstance(node, containers.ContainerProcessNode), name


def test_the_card_carries_no_wiring(card):
    """The card points at a Python pipeline and says nothing about I/O.

    Each node's inputs, outputs and parameters are derived from its CLI's own
    kwconf declaration, so the card restates none of them. What is left in the
    file is the claim, the evidence scope and the sweep.
    """
    assert card['kwdagger']['pipeline'] == (
        'smollm_example.pipeline.smollm_pipeline()')
    text = CARD_FPATH.read_text()
    for restated in ('out_paths', 'in_paths', 'algo_params', 'executable'):
        assert restated not in text, restated


def test_the_nodes_take_their_io_from_their_cli(card):
    """One source of authority: the tags on the CLI's kwconf values."""
    nodes = _configured_nodes(card)
    assert nodes['items'].out_paths == {'out_fpath': 'items.json'}
    assert sorted(nodes['ask'].in_paths) == ['items_fpath']
    assert nodes['ask'].out_paths == {'out_fpath': 'answers.json'}
    assert sorted(nodes['compare'].in_paths) == ['answer_fpaths']
    # And the generic loader is inherited rather than hand-rolled.
    for node in nodes.values():
        assert node._load_result_ref is None


def test_it_runs_on_released_kwdagger(card):
    """The reason the DAG is Python and not a declarative `nodes:` block.

    `endpoint_params` has no node-spec key before kwdagger 0.4.1, so a
    declarative card cannot say it. As a class attribute it needs nothing
    unreleased -- which is what lets this example ship now and be ported later.
    """
    assert isinstance(card['kwdagger']['pipeline'], str)
    assert _configured_nodes(card)['ask'].endpoint_params == ('endpoint',)


def test_the_endpoint_axis_is_what_gets_leased(card):
    """`endpoint` is an ordinary algo_param, so the matrix sweeps it; naming
    it in `endpoint_params` is what also makes its value the alias."""
    assert card['kwdagger']['matrix']['ask.endpoint'] == [
        'smol-135', 'smol-360']
    for alias in card['kwdagger']['matrix']['ask.endpoint']:
        node = _configured_nodes(card)['ask']
        node.configure({'endpoint': alias})
        assert f'--endpoint {alias}' in node.command


def test_the_serial_example_does_not_queue_for_external_gpus(card):
    """Serial cells cannot contend with each other; stale leases should fail fast."""
    node = _configured_nodes(card)['ask']
    node.configure({'endpoint': 'smol-135'})
    assert node.lease_queue is False
    assert node.lease_ttl == '1h'
    assert '--queue' not in node.command
    assert '--ttl 1h' in node.command


def test_it_runs_on_the_host_when_nothing_is_configured(card):
    """The same card during development: no image, no lease, no wrappers."""
    nodes = _configured_nodes(card, image='', leasing_on=False)
    nodes['ask'].configure({'endpoint': 'smol-135'})
    command = nodes['ask'].command
    assert 'docker run' not in command
    assert 'infer-stack run' not in command
    assert command.startswith(HOST_PY + ' -m smollm_example')


def _write_executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


def test_run_sh_checks_commands_without_invoking_them(tmp_path):
    """Presence checks use ``command -v``; only infer-stack status is run."""
    bindir = ub.Path(tmp_path) / 'bin'
    bindir.mkdir()
    calls = ub.Path(tmp_path) / 'calls.txt'

    _write_executable(
        bindir / 'magnet',
        f'#!/bin/sh\necho "magnet:$*" >> {calls}\nexit 97\n',
    )
    infer_stack_script = (
        '#!/bin/sh\n'
        f'printf "infer-stack:%s\\n" "$*" >> {calls}\n'
        'if [ "$1" = status ]; then\n'
        '    printf "backend: compose\\n"\n'
        '    exit 0\n'
        'fi\n'
        'exit 98\n'
    )
    _write_executable(bindir / 'infer-stack', infer_stack_script)
    _write_executable(
        bindir / 'python',
        '#!/bin/sh\n'
        f'printf "python:catalog=%s args=%s\\n" "$INFER_STACK_CATALOG" "$*" >> {calls}\n'
        'exit 0\n',
    )

    env = _child_env()
    env['PATH'] = f'{bindir}:/usr/bin:/bin'
    run_sh = CARD_FPATH.parent / 'run.sh'
    proc = subprocess.run(
        ['bash', run_sh, '--mock', '--no-container', '--dry_run=1'],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    invoked = calls.read_text().splitlines()
    assert not any(line.startswith('magnet:') for line in invoked)
    assert 'infer-stack:status' in invoked
    python_call = next(line for line in invoked if line.startswith('python:'))
    assert 'catalog-mock.yaml' in python_call
    assert '--container_image=' not in python_call
    assert '--per_node_leasing ' in python_call + ' '
    assert '--per_node_leasing=1' not in python_call


def test_run_sh_defaults_to_gpu_and_container(tmp_path):
    """The zero-flag path selects real weights and containerized node commands."""
    bindir = ub.Path(tmp_path) / 'bin'
    bindir.mkdir()
    calls = ub.Path(tmp_path) / 'calls.txt'

    _write_executable(
        bindir / 'magnet',
        f'#!/bin/sh\necho "magnet:$*" >> {calls}\nexit 97\n',
    )
    _write_executable(
        bindir / 'infer-stack',
        '#!/bin/sh\n'
        f'printf "infer-stack:%s\\n" "$*" >> {calls}\n'
        'if [ "$1" = status ]; then printf "backend: compose\\n"; exit 0; fi\n'
        'exit 98\n',
    )
    _write_executable(
        bindir / 'nvidia-smi',
        '#!/bin/sh\n'
        f'printf "nvidia-smi:%s\\n" "$*" >> {calls}\n'
        'if [ "$1" = -L ]; then printf "GPU 0: Fake GPU (UUID: GPU-fake)\\n"; exit 0; fi\n'
        'exit 2\n',
    )
    _write_executable(
        bindir / 'docker',
        f'#!/bin/sh\nprintf "docker:%s\\n" "$*" >> {calls}\nexit 0\n',
    )
    _write_executable(
        bindir / 'python',
        '#!/bin/sh\n'
        f'printf "python:catalog=%s args=%s\\n" "$INFER_STACK_CATALOG" "$*" >> {calls}\n'
        'exit 0\n',
    )

    env = _child_env()
    env['PATH'] = f'{bindir}:/usr/bin:/bin'
    run_sh = CARD_FPATH.parent / 'run.sh'
    proc = subprocess.run(
        ['bash', run_sh, '--dry_run=1'],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    invoked = calls.read_text().splitlines()
    assert 'nvidia-smi:-L' in invoked
    docker_build = next(
        line for line in invoked if line.startswith('docker:build ')
    )
    assert docker_build.split()[-1] == str(CARD_FPATH.parent)
    python_call = next(line for line in invoked if line.startswith('python:'))
    assert 'catalog.yaml' in python_call
    assert 'catalog-mock.yaml' not in python_call
    assert '--container_image=magnet-smollm-example:latest' in python_call
    assert '--container_mounts=' in python_call
    assert (
        '--container_env={"PYTHONPATH": "/opt/examples"}' in python_call
    )


def test_node_image_is_decoupled_from_the_repository_checkout():
    """Catalog/card edits must not invalidate MAGNET installation in the image."""
    dockerfile = (CARD_FPATH.parent / 'Dockerfile').read_text()
    dockerignore = (CARD_FPATH.parent / 'Dockerfile.dockerignore').read_text()

    assert 'ARG MAGNET_INSTALL=' in dockerfile
    assert 'cf9cc968d7a88470657e7938addfb3a1a6d0f986' in dockerfile
    assert 'RUN pip install "$MAGNET_INSTALL"' in dockerfile
    assert 'COPY . ' not in dockerfile
    assert 'COPY __init__.py ' in dockerfile
    assert 'COPY cli ' in dockerfile
    assert '!cli/**' in dockerignore
    assert dockerignore.lstrip().startswith('#')
    assert '\n*\n' in dockerignore


def test_run_sh_accepts_an_untracked_real_catalog(tmp_path):
    """Custom model experiments select another catalog without editing the fixture."""
    bindir = ub.Path(tmp_path) / 'bin'
    bindir.mkdir()
    calls = ub.Path(tmp_path) / 'calls.txt'
    custom_catalog = ub.Path(tmp_path) / 'catalog.local.yaml'
    custom_catalog.write_text('models: {}\nendpoints: {}\n')

    _write_executable(bindir / 'magnet', '#!/bin/sh\nexit 97\n')
    _write_executable(
        bindir / 'infer-stack',
        '#!/bin/sh\n'
        f'printf "infer-stack:%s\\n" "$*" >> {calls}\n'
        'if [ "$1" = status ]; then printf "backend: compose\\n"; exit 0; fi\n'
        'exit 98\n',
    )
    _write_executable(
        bindir / 'nvidia-smi',
        '#!/bin/sh\n'
        'if [ "$1" = -L ]; then printf "GPU 0: Fake GPU (UUID: GPU-fake)\\n"; exit 0; fi\n'
        'exit 2\n',
    )
    _write_executable(
        bindir / 'python',
        '#!/bin/sh\n'
        f'printf "python:catalog=%s args=%s\\n" "$INFER_STACK_CATALOG" "$*" >> {calls}\n'
        'exit 0\n',
    )

    env = _child_env(SMOLLM_CATALOG=str(custom_catalog))
    env['PATH'] = f'{bindir}:/usr/bin:/bin'
    run_sh = CARD_FPATH.parent / 'run.sh'
    proc = subprocess.run(
        ['bash', run_sh, '--no-container', '--dry_run=1'],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    python_call = next(
        line for line in calls.read_text().splitlines()
        if line.startswith('python:')
    )
    assert f'catalog={custom_catalog}' in python_call


def test_local_catalogs_are_ignored():
    gitignore = (CARD_FPATH.parents[1] / '.gitignore').read_text()
    assert 'examples/smollm_example/catalog.local*.yaml' in gitignore


def test_developer_smoke_test_release_is_explicit_and_summarized():
    text = (CARD_FPATH.parent / 'test.sh').read_text()
    assert 'infer-stack leases --json' in text
    assert 'require_clean_lease_pool "after $name"' in text
    assert 'infer-stack release --all --yes --evict' in text
    assert '--release)' in text
    assert 'SmolLM developer smoke variants' in text
    for name in ('real-container', 'mock-container', 'real-host', 'mock-host'):
        assert name in text


def test_run_sh_help_needs_no_runtime_prerequisites(tmp_path):
    """Help is wrapper documentation, so it must not probe the machine."""
    env = _child_env()
    env['PATH'] = '/usr/bin:/bin'
    run_sh = CARD_FPATH.parent / 'run.sh'
    proc = subprocess.run(
        ['bash', run_sh, '--help'],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'Usage: ./run.sh' in proc.stdout
    assert '--mock' in proc.stdout
    assert '--no-container' in proc.stdout
    assert '--params=' in proc.stdout
    assert 'ask.endpoint' in proc.stdout
    assert 'SMOLLM_RUNS' in proc.stdout
    assert 'SMOLLM_CATALOG' in proc.stdout
    assert 'requires MAGNET' not in proc.stderr


def test_run_sh_default_requires_a_gpu(tmp_path):
    """Real-model mode fails before backend/build work when no GPU is usable."""
    bindir = ub.Path(tmp_path) / 'bin'
    bindir.mkdir()
    calls = ub.Path(tmp_path) / 'calls.txt'

    _write_executable(bindir / 'magnet', '#!/bin/sh\nexit 97\n')
    _write_executable(
        bindir / 'infer-stack',
        f'#!/bin/sh\necho "infer-stack:$*" >> {calls}\nprintf "backend: compose\\n"\n',
    )
    _write_executable(
        bindir / 'nvidia-smi',
        f'#!/bin/sh\necho "nvidia-smi:$*" >> {calls}\nexit 1\n',
    )
    _write_executable(
        bindir / 'python',
        f'#!/bin/sh\necho "python:$*" >> {calls}\nexit 0\n',
    )

    env = _child_env()
    env['PATH'] = f'{bindir}:/usr/bin:/bin'
    run_sh = CARD_FPATH.parent / 'run.sh'
    proc = subprocess.run(
        ['bash', run_sh, '--no-container', '--dry_run=1'],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 1
    assert 'No usable NVIDIA GPU was detected' in proc.stderr
    assert './run.sh --mock' in proc.stderr
    invoked = calls.read_text().splitlines()
    assert 'nvidia-smi:-L' in invoked
    assert not any(line.startswith('infer-stack:') for line in invoked)
    assert not any(line.startswith('python:') for line in invoked)


def test_run_sh_reports_all_missing_commands(tmp_path):
    """A missing environment fails once with one useful prerequisite error."""
    bindir = ub.Path(tmp_path) / 'bin'
    bindir.mkdir()
    env = _child_env()
    env['PATH'] = f'{bindir}:/usr/bin:/bin'
    run_sh = CARD_FPATH.parent / 'run.sh'
    proc = subprocess.run(
        ['bash', run_sh, '--dry_run=1'],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 127
    assert 'requires MAGNET and infer-stack on PATH' in proc.stderr
    assert 'Missing: magnet infer-stack' in proc.stderr
    assert 'usage:' not in proc.stderr.lower()


# --- what the example computes ---------------------------------------------


class _StubOpenAI(BaseHTTPRequestHandler):
    """An OpenAI chat-completions server that answers the arithmetic."""

    def do_POST(self):
        body = json.loads(
            self.rfile.read(int(self.headers['Content-Length'])))
        prompt = body['messages'][0]['content']
        left, right = (int(v) for v in re.findall(r'\d+', prompt)[:2])
        # The 360M stub is wrong on purpose, so agreement is not trivially 1.
        total = left + right
        if body['model'] == 'smol-360':
            total += 1
        payload = {'choices': [{'message': {'content': str(total)}}]}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_server():
    server = HTTPServer(('127.0.0.1', 0), _StubOpenAI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}/v1'
    server.shutdown()


def _run(module, dpath, **kwargs):
    args = [f'--{k}={v}' for k, v in kwargs.items()]
    env = _child_env(OPENAI_BASE_URL=dpath['url'])
    proc = subprocess.run(
        [sys.executable, '-m', f'smollm_example.cli.{module}',
         *args],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc


def test_the_three_nodes_run_end_to_end(tmp_path, stub_server):
    """No container, no lease, no model -- just the executables and an
    OpenAI-shaped server, which is all the node code actually assumes."""
    tmp = ub.Path(tmp_path)
    ctx = {'url': stub_server}

    items_fpath = tmp / 'items.json'
    _run('make_items', ctx, n_items=4, seed=0, out_fpath=items_fpath)
    assert json.loads(items_fpath.read_text())['result']['metrics'][
        'n_items'] == 4

    answer_fpaths = []
    for alias in ('smol-135', 'smol-360'):
        out_fpath = tmp / f'{alias}.json'
        _run('ask_model', ctx, endpoint=alias, items_fpath=items_fpath,
             out_fpath=out_fpath)
        metrics = json.loads(out_fpath.read_text())['result']['metrics']
        assert metrics['endpoint'] == alias
        assert metrics['answered_rate'] == 1.0
        answer_fpaths.append(out_fpath)

    manifest = tmp / 'manifest.txt'
    manifest.write_text('\n'.join(map(str, answer_fpaths)))
    out_fpath = tmp / 'comparison.json'
    _run('compare_answers', ctx, answer_fpaths=manifest, out_fpath=out_fpath)

    metrics = json.loads(out_fpath.read_text())['result']['metrics']
    # The claim's subject: both endpoints answered everything.
    assert metrics['coverage'] == 1.0
    # The stubs disagree by construction, so this must not be 1.0 -- a test
    # that passed either way would not be testing the comparison.
    assert metrics['agreement'] == 0.0
    assert metrics['endpoints'] == 'smol-135,smol-360'


def test_a_missing_lease_says_so_rather_than_failing_per_request(tmp_path):
    """Without OPENAI_BASE_URL there is no server; one clear error beats N
    connection refusals."""
    env = _child_env(OPENAI_BASE_URL=None)
    items_fpath = ub.Path(tmp_path) / 'items.json'
    subprocess.run(
        [sys.executable, '-m', 'smollm_example.cli.make_items',
         f'--out_fpath={items_fpath}', '--n_items=1'],
        check=True, capture_output=True, env=env,
    )
    proc = subprocess.run(
        [sys.executable, '-m', 'smollm_example.cli.ask_model',
         '--endpoint=smol-135', f'--items_fpath={items_fpath}',
         f'--out_fpath={ub.Path(tmp_path) / "answers.json"}'],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode != 0
    assert 'not running inside a lease' in proc.stderr


def test_the_comparison_refuses_an_empty_gather():
    """A gather that produced nothing is a failed campaign, not 0% agreement."""
    with pytest.raises(ValueError, match='no answers'):
        compare([])
