"""
!uv pip install kaleido plotly
"""

import pandas as pd
import ubelt as ub
import kwutil
from magnet.backends.helm.helm_outputs import HelmRun
from magnet.backends.helm.helm_outputs import HelmOutputs
from magnet.backends.helm.rundiff import sankey
from magnet.backends.helm.helm_run_analysis import HelmRunAnalysis
from magnet.backends.helm.helm_run_diff import HelmRunDiff

"""
!python ~/code/aiq-magnet/dev/poc/inspect_historic_helm_runs.py /data/crfm-helm-public --out_fpath run_specs.yaml --out_detail_fpath run_details.yaml
"""
helm_rows = kwutil.Yaml.load('run_details.yaml')

if 0:
    # Debug HelmRunAnalysis
    run_dir = '/data/crfm-helm-public/classic/benchmark_output/runs/v0.3.0/wikifact:k=5,subject=symptoms_and_signs,model=lmsys_vicuna-7b-v1.3'
    helm_run = HelmRun.coerce(run_dir)
    self = HelmRunAnalysis(helm_run)
    self.summary(level=10)
    run_dir = '/data/crfm-helm-public/classic/benchmark_output/runs/v0.2.4/boolq:model=eleutherai_pythia-2.8b-v0,data_augmentation=canonical/'
    helm_run = HelmRun.coerce(run_dir)
    self = HelmRunAnalysis(helm_run)
    self.summary(level=10)
    run_dir = '/data/crfm-helm-public/capabilities/benchmark_output/runs/v1.12.0/ifeval:model=openai_gpt-oss-20b/'
    helm_run = HelmRun.coerce(run_dir)
    self = HelmRunAnalysis(helm_run)
    self.summary(level=10)

    for helm_row in helm_rows:
        run_dir = ub.Path(helm_row["run_dir"])
        helm_run = HelmRun.coerce(run_dir)
        self = HelmRunAnalysis(helm_run)

finished_jobs = list(
    ub.Path('/home/local/KHQ/jon.crall/code/aiq-magnet/results/helm').glob('*/DONE')
)
kwdagger_rows = []
for fpath in finished_jobs:
    config = kwutil.Json.coerce(fpath.parent / 'job_config.json')
    run_spec_name = config['helm.run_entry']
    dpath = fpath.parent
    runs = HelmOutputs.coerce(dpath / 'benchmark_output').suites()[0].runs()
    assert len(runs) == 1
    run = runs[0]
    kwdagger_rows.append(
        {
            'dpath': dpath,
            'run_spec_name': run_spec_name,
            'run': run,
        }
    )
kwdagger_lut = {r["run_spec_name"]: r for r in kwdagger_rows}
print(f"len(helm_rows)={len(helm_rows)}")
print(f"len(kwdagger_rows)={len(kwdagger_rows)}")


rundiff_lut = {}
for helm_row in ub.ProgIter(helm_rows, desc="compare runs"):
    run_dir = ub.Path(helm_row["run_dir"])
    suite_name = run_dir.parent.name
    benchmark_name = run_dir.parent.parent.parent.parent.name
    assert run_dir.parent.parent.parent.name == "benchmark_output"
    assert run_dir.parent.parent.name == "runs"
    helm_row["suite_name"] = suite_name
    helm_row["benchmark_name"] = benchmark_name
    run_dir = ub.Path(helm_row["run_dir"])
    run_spec_name = helm_row["run_spec_name"]

    kwrow = kwdagger_lut.get(run_spec_name)
    helm_row["reproduced_step1"] = (kwrow is not None)

    if kwrow is None:
        helm_row["agreement_bucket_base_task"] = "not attempted"
        continue

    raise Exception

    helm_run = HelmRun.coerce(run_dir)
    kwdg_run = kwrow["run"]

    a = HelmRunAnalysis(helm_run)
    b = HelmRunAnalysis(kwdg_run)

    rd = HelmRunDiff(run_a=helm_run, run_b=kwdg_run, a_name="HELM", b_name="KWDG")
    self = rd  # NOQA
    print(rd.summary_text(level='page'))
    print(rd.instance_summary_text(level='page'))

    if 0:
        print(rd.drilldown_core_metric_instances())
        rd.lookup_instance(('instance_id', 'id14045'), which='a')
        rd.lookup_instance(('instance_id', 'id14045'), which='b')

    raise Exception

    helm_row.update(rd.summary_base_task())
    helm_row.update(rd.summary_core())
    rundiff_lut[run_spec_name] = rd   # save for later drilldown

    # helm_stats = helm_run.json.stats()
    # kwdg_stats = kwdg_run.json.stats()

    # # out = compare.compare_run_pair(helm_stats, kwdg_stats, rel_tol=1e-4, abs_tol=1e-8)
    # helm_row.update(out)

df = pd.DataFrame(helm_rows)
print(df.value_counts(["benchmark_name", "reproduced_step1"]))


def attempt_status(row: dict[str, object]) -> str:
    return (
        'attempted' if row.get('reproduced_step1', False) else 'not_attempted'
    )


def agreement_label(row: dict[str, object]) -> str:
    # Used in the sankey plan; keep it stable.
    return row.get('agreement_bucket_base_task', 'unknown')

plan = sankey.Plan(
    sankey.Root("Initial Set"),
    sankey.Group("benchmark", by="benchmark_name"),
    sankey.Bucket("attempt", by=attempt_status),
    sankey.Bucket("agreement", by=agreement_label),
)
print(plan.to_text())

G = plan.build_sankey(helm_rows, label_fmt="{name}: {value}")
print(G.summarize(max_edges=150))

fig = G.to_plotly(title="HELM Reproduction Funnel")
fpath = 'helm_repro_sankey.jpg'
fig.write_image(fpath)
print(f'Wrote helm_repro_sankey: {fpath}')

if 1:
    ub.cmd(f'wormhole send {fpath}', verbose=3)
    """
    !wormhole send helm_repro_sankey.jpg
    """
