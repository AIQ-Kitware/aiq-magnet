import pytest
pytest.importorskip('helm', reason="needs the helm extra: pip install 'aiq-magnet[helm]'")  # noqa: E402
def test_small_helm_output_fixture():
    """The shared HELM loader fixture is local and contains real JSON shapes."""
    from magnet.demo.helm_demodata import ensure_helm_fixture_outputs

    dpath = ensure_helm_fixture_outputs(max_eval_instances=1)
    suite = dpath / 'benchmark_output' / 'runs' / 'my-suite'
    run_dpaths = sorted(p for p in suite.iterdir() if p.is_dir())
    assert len(run_dpaths) == 4
    for run_dpath in run_dpaths:
        assert (run_dpath / 'run_spec.json').is_file()
        assert (run_dpath / 'stats.json').is_file()
        assert (run_dpath / 'scenario_state.json').is_file()
