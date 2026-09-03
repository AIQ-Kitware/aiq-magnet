"""
A verdict says what produced it.

A run against a simulator writes the same shape of result as a run against a
served model. The caller knows which it was; MAGNET cannot infer it from the
shared alias/API, so the verdict is where that external fact has to live.
"""
import json
import textwrap

import pytest
import ubelt as ub
import yaml

from magnet.evaluation_new import (
    NewEvaluationCLI,
    NewEvaluationRecipe,
    coerce_provenance,
)

SCRIPT = """
import json, sys, pathlib
args = dict(a.lstrip('-').split('=', 1) for a in sys.argv[1:])
out = pathlib.Path(args['results_fpath'])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'result': {'metrics': {'score': float(args['seed']) / 10}}}))
"""


@pytest.fixture
def card(tmp_path):
    dpath = ub.Path(tmp_path)
    script = dpath / 'emit.py'
    script.write_text(textwrap.dedent(SCRIPT))
    data = {
        'name': 'probe', 'title': 'probe', 'description': 'probe',
        'version': '1.0', 'organizations': ['Kitware'],
        'submitter': {'name': 't', 'email': 't@example.com'},
        'links': [], 'tags': ['test'],
        'claim': {'python': 'assert metrics.emit.score < 100'},
        'kwdagger': {
            'result_node': 'emit',
            'pipeline': {'nodes': {'emit': {
                'executable': f'python {script}',
                'algo_params': {'seed': 1},
                'out_paths': {'results_fpath': 'results.json'},
                'primary_out_key': 'results_fpath',
            }}},
            'matrix': {'emit.seed': [1]},
        },
    }
    fpath = dpath / 'card.yaml'
    fpath.write_text(yaml.safe_dump(data, sort_keys=False))
    return fpath


def test_provenance_is_written_into_the_verdict(card, tmp_path):
    out = ub.Path(tmp_path) / 'out'
    prov = {'endpoint': {'kind': 'simulator', 'catalog': 'rehearsal'}}
    recipe = NewEvaluationRecipe(card, out)
    result_card = recipe.evaluate(backend='serial', provenance=prov)
    assert result_card.provenance == prov
    verdict = json.loads((out / recipe._run_hash / 'verdict.json').read_text())
    assert verdict['provenance'] == prov
    assert verdict['result'] == 'VERIFIED'


def test_no_provenance_means_no_key(card, tmp_path):
    out = ub.Path(tmp_path) / 'out'
    recipe = NewEvaluationRecipe(card, out)
    recipe.evaluate(backend='serial')
    verdict = json.loads((out / recipe._run_hash / 'verdict.json').read_text())
    assert 'provenance' not in verdict


def test_cli_accepts_provenance_as_yaml_text_or_file(card, tmp_path):
    out = ub.Path(tmp_path) / 'out'
    NewEvaluationCLI.main(argv=[
        str(card), '--output_path', str(out), '--backend', 'serial',
        '--provenance', '{endpoint_kind: replay, mock: false}'])
    verdicts = list(out.glob('*/verdict.json'))
    assert len(verdicts) == 1
    assert json.loads(verdicts[0].read_text())['provenance'] == {
        'endpoint_kind': 'replay', 'mock': False}

    prov_file = ub.Path(tmp_path) / 'prov.yaml'
    prov_file.write_text('endpoint_kind: real\n')
    out2 = ub.Path(tmp_path) / 'out2'
    NewEvaluationCLI.main(argv=[
        str(card), '--output_path', str(out2), '--backend', 'serial',
        '--provenance', str(prov_file)])
    verdict = json.loads(next(out2.glob('*/verdict.json')).read_text())
    assert verdict['provenance'] == {'endpoint_kind': 'real'}


def test_provenance_must_be_a_mapping():
    with pytest.raises(ValueError, match='mapping'):
        coerce_provenance('[1, 2]')
    assert coerce_provenance(None) is None
