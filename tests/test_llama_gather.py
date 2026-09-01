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


def test_compute_parameters_reach_the_materializer(monkeypatch):
    """The wrapper must not narrow the tool it wraps.

    It exists to make `model` and `subject` matrix axes. Dropping everything it
    did not name left `compute_if_missing` reachable but unusable: no way to set
    `max_eval_instances`, which helm-run refuses to start without.
    """
    from magnet.examples.llama_consistency import materialize_run

    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(
        materialize_run.MaterializeHelmRunConfig, 'main', staticmethod(spy))
    materialize_run.MaterializeLlamaRunCLI.main(
        argv=False,
        model='huggingface/smollm2-135m',
        subject='abstract_algebra',
        precomputed_root='/nowhere',
        out_dpath='/tmp/unused-by-the-spy',
        max_eval_instances=10,
        enable_huggingface_models='HuggingFaceTB/SmolLM2-135M',
        num_threads=4,
    )

    assert seen['max_eval_instances'] == 10
    assert seen['enable_huggingface_models'] == 'HuggingFaceTB/SmolLM2-135M'
    assert seen['num_threads'] == 4
    # `family` groups the gather; it says nothing about which run to find.
    assert 'family' not in seen
    assert seen['run_entry'] == (
        'mmlu:subject=abstract_algebra,method=multiple_choice_joint,'
        'model=huggingface/smollm2-135m'
    )


def test_a_gather_grouped_by_family_does_not_pool_families(tmp_path):
    """Two families in one sweep must not land in each other's average.

    `group_by: []` would hand every comparison every run in the matrix, which
    is invisible while there is one family and wrong the moment there are two.
    """
    import ubelt as ub
    from kwdagger.schedule import ScheduleEvaluationConfig, build_schedule

    params = ub.codeblock(
        '''
        pipeline:
          nodes:
            mat:
              executable: "echo mat"
              algo_params: {model: null, family: null}
              out_paths: {out_dpath: ".", done_fname: DONE}
              primary_out_key: done_fname
            pred:
              executable: "echo pred"
              in_paths: [runs]
              algo_params: {family: null}
              out_paths: {out_fpath: out.json}
              primary_out_key: out_fpath
          edges:
            - src: mat.out_dpath
              dst: pred.runs
              gather:
                group_by: [{src: family, dst: family}]
                require: all_success
        matrix:
          mat.model: [l7, l13, s135, s360]
          pred.family: [llama, smollm2]
          include:
            - {mat.model: l7,   mat.family: llama}
            - {mat.model: l13,  mat.family: llama}
            - {mat.model: s135, mat.family: smollm2}
            - {mat.model: s360, mat.family: smollm2}
        ''')
    config = ScheduleEvaluationConfig(
        params=params, root_dpath=str(tmp_path), run=False, backend='serial',
        print_commands=0, print_queue=0)
    dag, _ = build_schedule(config)

    # Four runs, two comparisons, and each comparison sees only its own two.
    summary = dag.compile_summary
    assert summary['collection_groups'] == 2
    assert summary['largest_collection'] == 2
    assert summary['collection_memberships'] == 4
