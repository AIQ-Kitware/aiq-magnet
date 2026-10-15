"""Record identity survives reuse, failure and interruption without fake verdicts."""

import json
import sys
from pathlib import Path

import pytest
import yaml

from magnet.evaluation_new import NewEvaluationRecipe
from magnet.records import atomic_json, source_snapshot


@pytest.fixture
def card(tmp_path):
    program = tmp_path / 'emit.py'
    program.write_text(
        'import json,pathlib,sys\na=dict(x.lstrip("-").split("=",1) for x in sys.argv[1:])\n'
        'p=pathlib.Path(a["out"]);p.parent.mkdir(parents=True,exist_ok=True)\n'
        'p.write_text(json.dumps({"result":{"metrics":{"score":0.5}}}))\n'
    )
    path = tmp_path / 'card.yaml'
    path.write_text(
        yaml.safe_dump(
            {
                'name': 'probe',
                'title': 'probe',
                'description': 'probe',
                'version': '1',
                'organizations': ['Kitware'],
                'submitter': {'name': 't', 'email': 't@example.com'},
                'links': [],
                'tags': ['test'],
                'claim': {'python': 'assert metrics.emit.score > 0'},
                'kwdagger': {
                    'result_node': 'emit',
                    'pipeline': {
                        'nodes': {
                            'emit': {
                                'executable': f'{sys.executable} {program}',
                                'out_paths': {'out': 'result.json'},
                                'primary_out_key': 'out',
                            }
                        }
                    },
                    'matrix': {},
                },
            }
        )
    )
    return path


def test_identity_is_published_before_scheduling_and_reports_are_native(
    card, tmp_path, monkeypatch
):
    from magnet._kwdagger import KWDaggerProcessor

    descriptor = tmp_path / 'identity.json'
    original = KWDaggerProcessor.schedule

    def observe(self, **options):
        data = json.loads(descriptor.read_text())
        assert data['record_id'] == recipe._run_hash
        record = Path(data['record_directory'])
        assert (
            json.loads((record / 'execution.json').read_text())['state']
            == 'scheduling'
        )
        assert (record / 'card.yaml').is_file()
        return original(self, **options)

    monkeypatch.setattr(KWDaggerProcessor, 'schedule', observe)
    recipe = NewEvaluationRecipe(card, tmp_path / 'out')
    result = recipe.evaluate(backend='serial', record_path_file=descriptor)
    record = Path(json.loads(descriptor.read_text())['record_directory'])
    assert result.result == 'VERIFIED'
    assert (record / 'report.md').is_file() and (
        record / 'report.txt'
    ).is_file()
    execution = json.loads((record / 'execution.json').read_text())
    assert (
        execution['state'] == 'evaluated'
        and execution['options']['backend'] == 'serial'
    )
    requests = json.loads((record / 'requested_runs.json').read_text())
    assert requests[0]['output_sha256_at_readout']
    assert requests[0]['config_sha256_at_readout']


@pytest.mark.parametrize(
    'exception,state',
    [(KeyboardInterrupt, 'interrupted'), (RuntimeError, 'error')],
)
def test_interrupted_or_failed_schedule_keeps_identity_without_claim_verdict(
    card, tmp_path, monkeypatch, exception, state
):
    from magnet._kwdagger import KWDaggerProcessor

    def fail(*args, **kwargs):
        raise exception('test interruption')

    monkeypatch.setattr(KWDaggerProcessor, 'schedule', fail)
    descriptor = tmp_path / 'identity.json'
    recipe = NewEvaluationRecipe(card, tmp_path / 'out')
    with pytest.raises(exception):
        recipe.evaluate(backend='serial', record_path_file=descriptor)
    record = Path(json.loads(descriptor.read_text())['record_directory'])
    assert json.loads((record / 'execution.json').read_text())['state'] == state
    assert not (record / 'verdict.json').exists()


def test_repeat_invocations_keep_separate_identity_and_reuse_node_ids(
    card, tmp_path
):
    out = tmp_path / 'out'
    first = NewEvaluationRecipe(card, out)
    first.evaluate(backend='serial')
    output = next((out / '_kwdagger').glob('*/*/result.json'))
    old = output.stat().st_mtime_ns
    second = NewEvaluationRecipe(card, out)
    second.evaluate(backend='serial', skip_existing=True)
    assert first._run_hash != second._run_hash
    assert output.stat().st_mtime_ns == old
    for recipe in [first, second]:
        assert (out / recipe._run_hash / 'execution.json').is_file()


def test_execution_records_facts_without_copying_runtime_settings(
    card, tmp_path
):
    from magnet.containers import ContainerSettings

    recipe = NewEvaluationRecipe(card, tmp_path / 'out')
    recipe.evaluate(
        backend='serial',
        dry_run=True,
        container_settings=ContainerSettings(
            env={'API_KEY': 'not-execution-metadata'}
        ),
    )
    execution = json.loads(
        (tmp_path / 'out' / recipe._run_hash / 'execution.json').read_text()
    )
    assert execution['options'] == {
        'backend': 'serial',
        'dry_run': True,
        'readout_only': False,
    }
    assert 'not-execution-metadata' not in json.dumps(execution)


def test_atomic_json_preserves_existing_record_on_serialization_error(tmp_path):
    target = tmp_path / 'nested' / 'execution.json'
    atomic_json(target, {'state': 'scheduling'})
    previous = target.read_bytes()
    with pytest.raises(TypeError):
        atomic_json(target, {'state': 'partial', 'unsupported': object()})
    assert target.read_bytes() == previous
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize(
    'relative',
    ['card.yaml', 'requested_runs.json', 'results/claim/verdict.json'],
)
def test_launcher_descriptor_cannot_replace_record_files(
    card, tmp_path, relative
):
    recipe = NewEvaluationRecipe(card, tmp_path / 'out')
    record = recipe.output_path / recipe._run_hash
    with pytest.raises(ValueError, match='outside the result directory'):
        recipe.evaluate(backend='serial', record_path_file=record / relative)
    assert not (record / relative).exists()


def test_readout_only_reuses_existing_evidence_without_running_task(
    card, tmp_path
):
    out = tmp_path / 'out'
    original = NewEvaluationRecipe(card, out)
    original.evaluate(backend='serial')
    artifact = next((out / '_kwdagger').glob('*/*/result.json'))
    mtime = artifact.stat().st_mtime_ns
    (tmp_path / 'emit.py').write_text('raise RuntimeError("must not run")')
    readout = NewEvaluationRecipe(card, out)
    result = readout.evaluate(backend='serial', readout_only=True)
    assert result.result == 'VERIFIED'
    assert artifact.stat().st_mtime_ns == mtime
    execution = json.loads(
        (out / readout._run_hash / 'execution.json').read_text()
    )
    assert execution['state'] == 'readout'


def test_readout_only_does_not_fill_missing_outputs(card, tmp_path):
    out = tmp_path / 'out'
    recipe = NewEvaluationRecipe(card, out)
    result = recipe.evaluate(backend='serial', readout_only=True)
    assert result.result == 'INCONCLUSIVE'
    assert not list((out / '_kwdagger').glob('*/*/result.json'))
    assert (out / recipe._run_hash / 'verdict.json').is_file()


def test_failed_presentation_does_not_change_or_hide_serialized_verdict(
    card, tmp_path, monkeypatch
):
    import magnet.results

    def fail(*args, **kwargs):
        raise RuntimeError('presentation failure')

    monkeypatch.setattr(magnet.results, 'render_result', fail)
    recipe = NewEvaluationRecipe(card, tmp_path / 'out')
    assert recipe.evaluate(backend='serial').result == 'VERIFIED'
    record = tmp_path / 'out' / recipe._run_hash
    assert (
        json.loads((record / 'verdict.json').read_text())['result']
        == 'VERIFIED'
    )
    assert (
        json.loads((record / 'execution.json').read_text())[
            'presentation_error'
        ]
        == 'RuntimeError'
    )


def test_unavailable_source_provenance_is_explicit_not_an_execution_gate(
    tmp_path,
):
    snapshot = source_snapshot(tmp_path / 'unavailable.py')
    assert snapshot['sha256'] is None
    assert snapshot['source_error'] == 'FileNotFoundError'
