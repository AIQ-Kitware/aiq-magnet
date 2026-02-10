"""
!uv pip install kaleido plotly
"""

import pandas as pd
import ubelt as ub

from magnet.helm_outputs import HelmRun
from magnet.backends.helm.rundiff import io, compare, sankey
from magnet.backends.helm.rundiff.sankey import Plan, Root, Group, Bucket
from magnet.backends.helm.rundiff.compare import attempt_status, agreement_label

"""
!python ~/code/aiq-magnet/dev/poc/inspect_historic_helm_runs.py /data/crfm-helm-public --out_fpath run_specs.yaml --out_detail_fpath run_details.yaml
"""
# helm_rows = kwutil.Yaml.load('run_details.yaml')

# finished_jobs = list(
#     ub.Path('/home/local/KHQ/jon.crall/code/aiq-magnet/results/helm').glob('*/DONE')
# )
# kwdagger_rows = []
# for fpath in finished_jobs:
#     config = kwutil.Json.coerce(fpath.parent / 'job_config.json')
#     run_spec_name = config['helm.run_entry']
#     dpath = fpath.parent
#     runs = HelmOutputs.coerce(dpath / 'benchmark_output').suites()[0].runs()
#     assert len(runs) == 1
#     run = runs[0]
#     kwdagger_rows.append(
#         {
#             'dpath': dpath,
#             'run_spec_name': run_spec_name,
#             'run': run,
#         }
#     )


# fig = go.Figure(
#     go.Sankey(
#         node=dict(label=nodes, pad=15, thickness=18),
#         link=dict(source=source, target=target, value=value),
#     )
# )

# fig.update_layout(
#     title_text='HELM Reproduction Funnel',
#     font_size=14,
# )

# fig.write_image('helm_repro_sankey.png', scale=2)  # higher scale = sharper
# fig.write_image('helm_repro_sankey.jpg', scale=2)
# # For papers, SVG/PDF is often best:
# fig.write_image('helm_repro_sankey.svg')
# fig.write_image('helm_repro_sankey.pdf')

# # import wormhole
# ub.cmd('wormhole send helm_repro_sankey.jpg', verbose=3)
# """
# !wormhole send helm_repro_sankey.jpg
# """

# ---- Inputs (edit these) ----
RUN_DETAILS_YAML = "run_details.yaml"
KWDAGGER_RESULTS_ROOT = "/home/local/KHQ/jon.crall/code/aiq-magnet/results/helm"

helm_rows = io.load_public_helm_rows(RUN_DETAILS_YAML)
kwdagger_rows = io.discover_kwdagger_runs(KWDAGGER_RESULTS_ROOT)
kwdagger_lut = {r["run_spec_name"]: r for r in kwdagger_rows}

print(f"len(helm_rows)={len(helm_rows)}")
print(f"len(kwdagger_rows)={len(kwdagger_rows)}")

for helm_row in ub.ProgIter(helm_rows, desc="compare runs"):
    io.annotate_public_row_paths(helm_row)
    run_dir = ub.Path(helm_row["run_dir"])
    run_spec_name = helm_row["run_spec_name"]

    kwrow = kwdagger_lut.get(run_spec_name)
    helm_row["reproduced_step1"] = (kwrow is not None)

    if kwrow is None:
        helm_row["agreement_bucket_base_task"] = "not attempted"
        continue

    helm_run = HelmRun.coerce(run_dir)
    kwdg_run = kwrow["run"]

    helm_stats = helm_run.json.stats()
    kwdg_stats = kwdg_run.json.stats()

    out = compare.compare_run_pair(helm_stats, kwdg_stats, rel_tol=1e-4, abs_tol=1e-8)
    helm_row.update(out)

df = pd.DataFrame(helm_rows)
print(df.value_counts(["benchmark_name", "reproduced_step1"]))

plan = Plan(
    Root(root),
    Group("benchmark", by="benchmark_name"),
    Bucket("attempt", by=attempt_status),
    Bucket("agreement", by=agreement_label),
)
print(plan.to_text())

G = plan.build_sankey(helm_rows, label_fmt="{name}: {value}")
print(plan.graph_to_text(G, max_edges=150))

fig = sankey.plotly_sankey(G, title="HELM Reproduction Funnel")
sankey.write_plotly_figure(fig, out_prefix="helm_repro_sankey", scale=2, formats=("png","jpg","svg","pdf"))
print("Wrote helm_repro_sankey.[png/jpg/svg/pdf]")
