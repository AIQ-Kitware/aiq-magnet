import pytest
pytest.importorskip('helm', reason="needs the helm extra: pip install 'aiq-magnet[helm]'")  # noqa: E402
"""
Acquisition as a pipeline node: a whole HELM suite, cache-first.
"""
import json
from pathlib import Path

import pytest

from magnet.backends.helm.cli.materialize_helm_suite import (
    MaterializeHelmSuiteConfig, materialize_suite, main,
)
from magnet.demo.helm_demodata import ensure_helm_remote_store_fixture


@pytest.fixture
def bucket():
    return Path(ensure_helm_remote_store_fixture())


def _runs_in(bucket, benchmark='lite', version='v1.13.0'):
    return sorted(p.name for p in (bucket / benchmark / 'benchmark_output' / 'runs' / version).iterdir())


def test_precomputed_root_is_linked_not_fetched(tmp_path, bucket):
    out = tmp_path / 'node'
    config = MaterializeHelmSuiteConfig(
        benchmark='lite', version='v1.13.0', runs='regex:med_qa.*',
        precomputed_roots=str(bucket), download='never',
        suite_dpath=str(out / 'benchmark_output/runs/v1.13.0'))
    manifest = materialize_suite(config)
    want = [r for r in _runs_in(bucket) if r.startswith('med_qa')]
    assert sorted(manifest['members']) == want
    assert all(m['how'] == 'symlink' for m in manifest['members'].values())
    for run_id in want:
        link = out / 'benchmark_output/runs/v1.13.0' / run_id
        assert link.is_symlink() and link.resolve().is_dir()
    assert (out / 'DONE').read_text().strip() == 'DONE'
    assert json.loads((out / 'materialize_manifest.json').read_text())['version'] == 'v1.13.0'


def test_missing_runs_are_fetched_when_allowed(tmp_path, bucket):
    out = tmp_path / 'node'
    config = MaterializeHelmSuiteConfig(
        benchmark='lite', version='v1.13.0', runs='regex:med_qa.*',
        precomputed_roots=str(tmp_path / 'empty-root'), download='auto',
        bucket=str(bucket),  # a local directory stands in for the remote
        suite_dpath=str(out / 'benchmark_output/runs/v1.13.0'))
    manifest = materialize_suite(config)
    assert manifest['members'] and all(m['how'] == 'download' for m in manifest['members'].values())
    for run_id in manifest['members']:
        assert (out / 'benchmark_output/runs/v1.13.0' / run_id).is_dir()
    assert (out / 'DONE').exists()


def test_download_never_refuses_an_absent_run(tmp_path, bucket):
    out = tmp_path / 'node'
    config = MaterializeHelmSuiteConfig(
        benchmark='lite', version='v1.13.0', runs='regex:med_qa.*',
        precomputed_roots=str(tmp_path / 'empty-root'), download='never',
        bucket=str(bucket),
        suite_dpath=str(out / 'benchmark_output/runs/v1.13.0'))
    with pytest.raises(RuntimeError, match='no runs|precomputed root'):
        materialize_suite(config)
    assert not (out / 'DONE').exists()


def test_cli_entry_point(tmp_path, bucket):
    out = tmp_path / 'node'
    rc = main(argv=[
        '--benchmark', 'lite', '--version', 'v1.13.0', '--runs', 'regex:med_qa.*',
        '--precomputed_roots', str(bucket), '--download', 'never',
        '--suite_dpath', str(out / 'benchmark_output/runs/v1.13.0'),
        '--done_fpath', str(out / 'DONE')])
    assert rc == 0 and (out / 'DONE').exists()
