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


def test_a_sidecar_path_is_normalized_for_the_materializer(tmp_path):
    """KWDagger owns the sentinel path and passes it whole.

    `materialize_helm_run` treats it as a filename and joins it onto
    `out_dpath`, so a relative path carrying directories -- which is what
    KWDagger hands over whenever `--output_path` was relative -- would nest a
    second copy of itself inside the node directory.
    """
    import os
    from magnet.examples.llama_consistency.materialize_run import _sidecar_path

    # A bare filename is what the join expects; leave it alone.
    assert _sidecar_path('DONE') == 'DONE'

    # Anything with directories is made absolute, which wins the join.
    cwd = ub.Path(os.getcwd())
    assert _sidecar_path('runs/node_id/DONE') == str(cwd / 'runs/node_id/DONE')

    absolute = str(ub.Path(tmp_path) / 'runs' / 'node_id' / 'DONE')
    assert _sidecar_path(absolute) == absolute


def test_a_relative_output_path_does_not_nest_the_node_directory(
        tmp_path, monkeypatch):
    """The failure this guards: `<out>/<out>/DONE` instead of `<out>/DONE`."""
    from magnet.demo.helm_demodata import ensure_helm_llama_fixture_outputs
    from magnet.examples.llama_consistency.materialize_run import (
        MaterializeLlamaRunCLI)

    fixture = ensure_helm_llama_fixture_outputs()
    monkeypatch.chdir(tmp_path)
    out_dpath = 'runs/materialize_run_id_probe'

    MaterializeLlamaRunCLI.main(
        argv=False,
        model='meta/llama-2-13b',
        subject='anatomy',
        precomputed_root=str(fixture),
        mode='reuse_only',
        out_dpath=out_dpath,
        done_fname=f'{out_dpath}/DONE',
    )

    assert (ub.Path(tmp_path) / out_dpath / 'DONE').exists()
    assert not (ub.Path(tmp_path) / out_dpath / 'runs').exists()
