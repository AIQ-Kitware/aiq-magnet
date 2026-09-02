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
from magnet.examples.smollm_example.cli.compare_answers import compare
from magnet.leasing import LeaseSettings

CARD_FPATH = (
    ub.Path(__file__).parent.parent
    / 'magnet/examples/smollm_example/smollm_kwdagger.yaml'
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
    leasing.apply_settings(pipeline, LeaseSettings(enabled=leasing_on))
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
        'magnet.examples.smollm_example.pipeline.smollm_pipeline()')
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


def test_it_runs_on_the_host_when_nothing_is_configured(card):
    """The same card during development: no image, no lease, no wrappers."""
    nodes = _configured_nodes(card, image='', leasing_on=False)
    nodes['ask'].configure({'endpoint': 'smol-135'})
    command = nodes['ask'].command
    assert 'docker run' not in command
    assert 'infer-stack run' not in command
    assert command.startswith('python -m magnet.examples.smollm_example')


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
    env = {**dict(__import__('os').environ), 'OPENAI_BASE_URL': dpath['url']}
    proc = subprocess.run(
        [sys.executable, '-m', f'magnet.examples.smollm_example.cli.{module}',
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
    import os

    env = {k: v for k, v in os.environ.items() if k != 'OPENAI_BASE_URL'}
    items_fpath = ub.Path(tmp_path) / 'items.json'
    subprocess.run(
        [sys.executable, '-m', 'magnet.examples.smollm_example.cli.make_items',
         f'--out_fpath={items_fpath}', '--n_items=1'],
        check=True, capture_output=True, env=env,
    )
    proc = subprocess.run(
        [sys.executable, '-m', 'magnet.examples.smollm_example.cli.ask_model',
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
