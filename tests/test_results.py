"""Detached native result reading, presentation and portable evidence exports."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from magnet.results import (
    discover_result_bundles, export_result_bundle, load_result_bundle,
    render_result, verify_result_bundle,
)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'source' / 'record'
    root.mkdir(parents=True)
    artifact = root.parent / '_kwdagger' / 'summary' / 'summary_id_123' / 'summary.json'
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({'mean': 0.125, 'rows': 7}))
    (artifact.parent / 'job_config.json').write_text('{"budget": 10}')
    card = {'title': 'A detached result', 'description': 'The experimental scope.',
            'claim': {'python': "raise RuntimeError('report must not execute this')"},
            'kwdagger': {'pipeline': 'unavailable_pipeline:factory', 'result_node': 'summary'},
            'symbols': {'threshold': {'value': 0}}}
    (root / 'card.yaml').write_text(yaml.safe_dump(card))
    verdict = {'result': 'VERIFIED', 'claims': ['summary_id_123_abc'],
               'evidence': {'scope': 'requested', 'available': 1, 'discovered': 3},
               'requested_work': {'processes': 3, 'outputs_available': 1}}
    (root / 'verdict.json').write_text(json.dumps(verdict))
    cellpath = root / 'results' / verdict['claims'][0] / 'verdict.json'
    cellpath.parent.mkdir(parents=True)
    cellpath.write_text(json.dumps({'status': 'VERIFIED', 'output': 'Assertion holds',
        'cell': 'summary_id_123', 'artifact': str(artifact),
        'symbols': {'threshold': 0, 'metrics.summary.mean': 0.125},
        'consumed': ['metrics.summary.mean'],
        'evidence': {'metrics.summary.mean': 0.125,
                     'metrics.summary.folds': [{'name': 'a', 'delta': 0.1, 'ci': [0.01, 0.2]},
                                                {'name': 'b', 'delta': 0.15, 'ci': [0.02, 0.3]}],
                     'metrics.parent.rows': ['inherited data should not dominate presentation']}}))
    (root / 'requested_runs.json').write_text(json.dumps([
        {'process_id': 'summary_id_123', 'node': 'summary', 'expected_output': str(artifact)},
        {'process_id': 'pending_id', 'node': 'pending', 'expected_output': str(artifact.parent / 'absent.json')},
    ]))
    (root / 'log').write_text('not needed for reading results')
    # An unrelated deployment descriptor must never enter a core/evidence export.
    (root.parent / '_kwdagger' / 'pool.json').write_text('{"API_KEY": "do-not-export"}')
    return root


def test_detached_report_needs_neither_pipeline_nor_claim_execution(bundle, tmp_path):
    output = export_result_bundle(bundle, tmp_path / 'core')
    shutil.rmtree(bundle.parent)
    stored = load_result_bundle(output)
    assert stored.verdict['result'] == 'VERIFIED'
    assert not (output / 'log').exists()
    report = render_result(stored)
    assert '0.125' in report and 'The experimental scope.' in report
    assert '| name | delta |' in report
    assert 'inherited data should not dominate' not in report
    assert not list(output.rglob('pool.json'))
    assert verify_result_bundle(output)['valid']


def test_exported_artifact_locator_survives_moving_and_original_deletion(bundle, tmp_path):
    native = load_result_bundle(bundle)
    original = next(iter(native.cells.values()))['artifact']
    output = export_result_bundle(bundle, tmp_path / 'export', supporting='terminal')
    assert (output / 'card.yaml').read_bytes() == (bundle / 'card.yaml').read_bytes()
    moved = tmp_path / 'moved'
    output.rename(moved)
    shutil.rmtree(bundle.parent)
    stored = load_result_bundle(moved)
    assert json.loads(stored.artifact_path(original).read_text()) == {'mean': 0.125, 'rows': 7}
    assert verify_result_bundle(moved)['valid']
    # Re-export an already portable bundle, preserving the original source locator.
    second = export_result_bundle(moved, tmp_path / 'again', supporting='terminal')
    assert load_result_bundle(second).artifact_path(original).is_file()


def test_optional_missing_supporting_files_preserve_partial_result(bundle, tmp_path):
    output = export_result_bundle(bundle, tmp_path / 'out', supporting='requested')
    verified = verify_result_bundle(output)
    assert verified['valid']
    assert verified['missing_supporting'][0]['process_id'] == 'pending_id'
    assert load_result_bundle(output).verdict['result'] == 'VERIFIED'


def test_node_filter_exports_only_selected_declared_outputs(bundle, tmp_path):
    output = export_result_bundle(bundle, tmp_path / 'out', supporting='requested', nodes=('summary',))
    manifest = json.loads((output / 'bundle_manifest.json').read_text())
    assert len(manifest['artifacts']) == 1
    assert not manifest['missing_supporting']
    assert any(k.endswith('job_config.json') for k in manifest['files'])


@pytest.mark.parametrize('change', ['edit', 'remove'])
def test_integrity_detects_modified_or_missing_export_file(bundle, tmp_path, change):
    output = export_result_bundle(bundle, tmp_path / 'out', supporting='terminal')
    artifact = next((output / 'artifacts').glob('*/summary.json'))
    if change == 'edit':
        artifact.write_text('{"mean": 99}')
    else:
        artifact.unlink()
    checked = verify_result_bundle(output)
    assert not checked['valid']
    assert checked['errors']


def test_missing_claim_is_reported_separately_from_aggregate_verdict(bundle):
    next((bundle / 'results').glob('*/verdict.json')).unlink()
    stored = load_result_bundle(bundle)
    assert stored.verdict['result'] == 'VERIFIED'
    assert stored.missing_claims == ['summary_id_123_abc']
    assert 'Missing referenced claim' in render_result(stored)
    assert not verify_result_bundle(bundle)['valid']


def test_empty_evidence_and_interrupted_records_can_be_read(bundle):
    (bundle / 'verdict.json').write_text('{"result": "INCONCLUSIVE", "claims": []}')
    assert 'INCONCLUSIVE' in render_result(load_result_bundle(bundle))
    (bundle / 'verdict.json').unlink()
    (bundle / 'execution.json').write_text('{"state": "interrupted"}')
    report = render_result(load_result_bundle(bundle))
    assert 'NOT_RECORDED' in report and 'interrupted' in report


def test_legacy_symbols_and_declared_metrics_are_presented(bundle):
    card = yaml.safe_load((bundle / 'card.yaml').read_text())
    card.pop('kwdagger')
    (bundle / 'card.yaml').write_text(yaml.safe_dump(card))
    verdict = json.loads((bundle / 'verdict.json').read_text())
    verdict['metrics'] = {'Average score': 0.25}
    (bundle / 'verdict.json').write_text(json.dumps(verdict))
    path = next((bundle / 'results').glob('*/verdict.json'))
    cell = json.loads(path.read_text())
    cell.pop('evidence')
    path.write_text(json.dumps(cell))
    report = render_result(load_result_bundle(bundle), format='text')
    assert 'Average score' in report and '0.25' in report and '0.125' in report
    assert not report.startswith('#')


def test_result_commands_return_normal_exit_statuses(bundle, tmp_path, capsys):
    from magnet.cli.main import MagnetCLI
    assert MagnetCLI.main(argv=['results', 'show', str(bundle), '--format=json']) == 0
    output = capsys.readouterr().out
    assert json.loads(output)['verdict']['result'] == 'VERIFIED'
    target = tmp_path / 'out'
    assert MagnetCLI.main(argv=['results', 'export', str(bundle), str(target)]) == 0
    assert MagnetCLI.main(argv=['results', 'verify', str(target)]) == 0
    (target / 'report.md').write_text('changed')
    assert MagnetCLI.main(argv=['results', 'verify', str(target)]) == 1


@pytest.mark.parametrize('relative', ['../outside', '/absolute'])
def test_manifest_paths_cannot_escape_bundle(bundle, tmp_path, relative):
    output = export_result_bundle(bundle, tmp_path / 'out')
    manifest = json.loads((output / 'bundle_manifest.json').read_text())
    manifest['files'][relative] = {'sha256': 'x', 'size': 0}
    (output / 'bundle_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='invalid bundle-relative'):
        verify_result_bundle(output)


def test_export_does_not_replace_existing_or_nest_in_source(bundle, tmp_path):
    with pytest.raises(FileExistsError):
        export_result_bundle(bundle, bundle)
    with pytest.raises(ValueError, match='outside'):
        export_result_bundle(bundle, bundle / 'export')
    with pytest.raises(ValueError, match='invalid claim ID'):
        verdict = json.loads((bundle / 'verdict.json').read_text())
        verdict['claims'] = ['../../escape']
        (bundle / 'verdict.json').write_text(json.dumps(verdict))
        load_result_bundle(bundle)


def test_discovery_prunes_cache_and_latest_alias(bundle):
    (bundle.parent / 'latest').symlink_to(bundle.name, target_is_directory=True)
    # Even a cache with a file named card.yaml is not a MAGNET record.
    (bundle.parent / '_kwdagger' / 'card.yaml').write_text('invalid: [')
    records = discover_result_bundles(bundle.parent)
    assert len(records) == 1
    assert records[0]['path'] == str(bundle)
    assert records[0]['result'] == 'VERIFIED'


def test_explicit_attachments_are_hashed_without_replacing_native_records(bundle, tmp_path):
    note = tmp_path / 'note.txt'
    note.write_text('supplemental analysis, not a native verdict')
    output = export_result_bundle(bundle, tmp_path / 'out', attachments={'verdict.json': note})
    assert (output / 'attachments/verdict.json').read_text() == note.read_text()
    assert load_result_bundle(output).verdict['result'] == 'VERIFIED'
    assert verify_result_bundle(output)['valid']
    (output / 'attachments/verdict.json').write_text('changed')
    assert not verify_result_bundle(output)['valid']
