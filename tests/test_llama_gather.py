"""
The llama recipe reads a declared set of runs, not a directory.

Removing the glob is the point of the materialize node: what a verdict averaged
over is fixed when the matrix is compiled, so a run that appears in a shared
HELM cache afterwards cannot silently join the average.
"""
import pytest
import ubelt as ub

from magnet.examples.llama_consistency.llama_predict import (
    read_gathered_run_dpaths)


def _materialized(root, name, runs):
    """Build a directory shaped like one materialize node's output."""
    dpath = (ub.Path(root) / name).ensuredir()
    for run in runs:
        (dpath / 'benchmark_output' / 'runs' / 'a-suite' / run).ensuredir()
    return dpath


def test_a_manifest_resolves_to_the_runs_it_names(tmp_path):
    first = _materialized(tmp_path, 'one', ['mmlu:subject=anatomy,model=m'])
    second = _materialized(tmp_path, 'two', ['mmlu:subject=algebra,model=m'])
    manifest = ub.Path(tmp_path) / 'gathered.txt'
    manifest.write_text(f'{first}\n{second}\n')

    found = read_gathered_run_dpaths(manifest)
    assert [p.name for p in found] == [
        'mmlu:subject=anatomy,model=m', 'mmlu:subject=algebra,model=m',
    ]


def test_an_unlisted_run_is_not_reachable(tmp_path):
    """The directory holding the manifest is not what gets read."""
    listed = _materialized(tmp_path, 'listed', ['mmlu:subject=anatomy,model=m'])
    _materialized(tmp_path, 'unlisted', ['mmlu:subject=ethics,model=m'])
    manifest = ub.Path(tmp_path) / 'gathered.txt'
    manifest.write_text(f'{listed}\n')

    found = read_gathered_run_dpaths(manifest)
    assert [p.name for p in found] == ['mmlu:subject=anatomy,model=m']


def test_blank_manifest_lines_are_ignored(tmp_path):
    only = _materialized(tmp_path, 'only', ['mmlu:subject=anatomy,model=m'])
    manifest = ub.Path(tmp_path) / 'gathered.txt'
    manifest.write_text(f'\n{only}\n\n')
    assert len(read_gathered_run_dpaths(manifest)) == 1


def test_a_manifest_naming_an_unmaterialized_run_is_an_error(tmp_path):
    """Silently averaging over fewer runs than declared would be worse."""
    manifest = ub.Path(tmp_path) / 'gathered.txt'
    manifest.write_text(f'{tmp_path / "never-ran"}\n')
    with pytest.raises(FileNotFoundError, match='benchmark_output/runs'):
        read_gathered_run_dpaths(manifest)
