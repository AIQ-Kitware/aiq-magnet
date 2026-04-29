import json
import subprocess

import kwutil
import pytest
import ubelt as ub

from magnet.backends.helm.helm_outputs import HelmRun
from magnet.backends.helm.helm_run_diff import HelmRunDiff


def _coerce_demo_run() -> HelmRun:
    """Best-effort demo run fetch; skip if environment cannot materialize it."""
    try:
        return HelmRun.demo()
    except subprocess.CalledProcessError as ex:
        pytest.skip(f'Unable to materialize HelmRun.demo(): {ex!r}')


def test_helm_run_diff_heavy_demo_workflow():
    """Heavyweight regression test for end-to-end HelmRunDiff behavior."""
    run_a = _coerce_demo_run()
    dpath = ub.Path.appdir('magnet/tests/helm/helm_run_diff_heavy').delete().ensuredir()

    # Case 1: identical copy
    same_path = dpath / (run_a.path.name + '_same')
    run_a.path.copy(same_path)
    run_b = HelmRun(same_path)
    rd = HelmRunDiff(run_a, run_b, a_name='orig', b_name='same')
    info = rd.summary_dict(level=20)
    assert info['run_spec_dict_ok'] is True
    assert info['scenario_ok'] in {True, None}
    assert info['value_agreement']['overall']['mismatched'] == 0
    assert info['value_agreement']['overall']['agree_ratio'] == 1.0
    assert info['dataset_overlap']['base_iou'] == 1.0
    assert info['diagnosis']['label'] in {'reproduced', 'core_match_bookkeeping_drift'}
    json.dumps(info, allow_nan=False)

    # Case 2: perturb one run-level stat mean
    stats_path = dpath / (run_a.path.name + '_statsmod')
    run_a.path.copy(stats_path)
    stat_fpath = stats_path / 'stats.json'
    stats = kwutil.Json.loads(stat_fpath.read_text())
    old_mean = float(stats[0].get('mean', 0.0))
    stats[0]['mean'] = old_mean + 1.23
    stat_fpath.write_text(kwutil.Json.dumps(stats))

    rd2 = HelmRunDiff(run_a, HelmRun(stats_path), a_name='orig', b_name='stats+1.23')
    info2 = rd2.summary_dict(level=20)
    assert info2['value_agreement']['overall']['mismatched'] >= 1
    assert info2['diagnosis']['label'] in {
        'core_metric_drift',
        'core_match_bookkeeping_drift',
        'reproduced',
    }
    json.dumps(info2, allow_nan=False)

    # Case 3: perturb one per-instance stat mean (if file exists)
    inst_path = dpath / (run_a.path.name + '_perinstmod')
    run_a.path.copy(inst_path)
    pi_fpath = inst_path / 'per_instance_stats.json'
    if pi_fpath.exists():
        perinst = kwutil.Json.loads(pi_fpath.read_text())
        ei, sj = 0, None
        for j, stat in enumerate(perinst[ei]['stats']):
            if int(stat.get('count', 0) or 0) and ('mean' in stat):
                sj = j
                break
        assert sj is not None
        old = float(perinst[ei]['stats'][sj]['mean'])
        perinst[ei]['stats'][sj]['mean'] = old + 9.0
        pi_fpath.write_text(kwutil.Json.dumps(perinst))

        rd_i = HelmRunDiff(run_a, HelmRun(inst_path), a_name='orig', b_name='perinst+9')
        inst_info = rd_i.instance_summary_dict(top_n=5)
        assert inst_info['means']['mismatched'] >= 1
        json.dumps(inst_info, allow_nan=False)

    # Case 4: run-spec deployment change
    spec_path = dpath / (run_a.path.name + '_runspec_mod')
    run_a.path.copy(spec_path)
    spec_fpath = spec_path / 'run_spec.json'
    run_spec = kwutil.Json.loads(spec_fpath.read_text())
    run_spec.setdefault('adapter_spec', {})
    run_spec['adapter_spec']['model_deployment'] = 'someotherdeploy/gpt2'
    spec_fpath.write_text(kwutil.Json.dumps(run_spec))

    rd4 = HelmRunDiff(run_a, HelmRun(spec_path), a_name='orig', b_name='runspec_mod')
    info4 = rd4.summary_dict(level=20)
    assert info4['run_spec_dict_ok'] is False
    assert info4['run_spec_semantic']['deployment_changed'] is True
    assert info4['diagnosis']['label'] in {'deployment_drift', 'execution_spec_drift'}
    json.dumps(info4, allow_nan=False)
