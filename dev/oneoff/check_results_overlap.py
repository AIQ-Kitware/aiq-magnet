"""
!uv pip install kaleido plotly
"""

import pandas as pd
import ubelt as ub
import kwutil
from magnet.backends.helm.helm_outputs import HelmRun
from magnet.backends.helm.helm_outputs import HelmOutputs
from magnet.backends.helm.helm_run_analysis import HelmRunAnalysis
from magnet.backends.helm.helm_run_diff import HelmRunDiff
from magnet.utils import sankey

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
        run_dir = ub.Path(helm_row['run_dir'])
        helm_run = HelmRun.coerce(run_dir)
        self = HelmRunAnalysis(helm_run)

finished_jobs = list(
    ub.Path('/home/local/KHQ/jon.crall/code/aiq-magnet/results/helm').glob(
        '*/DONE'
    )
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
kwdagger_lut = {r['run_spec_name']: r for r in kwdagger_rows}
print(f'len(helm_rows)={len(helm_rows)}')
print(f'len(kwdagger_rows)={len(kwdagger_rows)}')


def sankey_stats(rd: HelmRunDiff) -> dict:
    """
    Return a small, stable set of fields intended for building Sankey tables.
    """
    s = self.summary_dict(level=1)
    va = (s.get("value_agreement") or {})
    by_class = (va.get("by_class") or {})
    overall = (va.get("overall") or {})

    core = (by_class.get("core") or {})
    book = (by_class.get("bookkeeping") or {})

    core_ratio = core.get("agree_ratio", None)
    book_ratio = book.get("agree_ratio", None)
    overall_ratio = overall.get("agree_ratio", None)

    spec_ok = bool(s.get("run_spec_dict_ok", False))
    scen_ok = s.get("scenario_ok", None)  # True/False/None

    spec_status = "spec match" if spec_ok else "spec mismatch"
    if scen_ok is None:
        scenario_status = "scenario unknown"
    else:
        scenario_status = "scenario match" if scen_ok else "scenario mismatch"

    # stats_name_status is cheap and very useful as “schema drift” indicator
    cov = (s.get("stats_coverage_by_name") or {})
    stats_name_status = "stats names match" if (cov.get("only_a", 0) == 0 and cov.get("only_b", 0) == 0) else "stats names mismatch"

    def _bucket_ratio(x: float | None, *, good=0.995, ok=0.95) -> str:
        if x is None:
            return "unknown"
        if x >= good:
            return "high"
        if x >= ok:
            return "medium"
        return "low"

    core_b = _bucket_ratio(core_ratio)
    book_b = _bucket_ratio(book_ratio)

    if core_b == "high" and (book_b in {"high", "unknown"}):
        agreement_quality = "match"
    elif core_b == "high" and book_b in {"medium", "low"}:
        agreement_quality = "core match, bookkeeping differs"
    elif core_b == "medium":
        agreement_quality = "core partial"
    elif core_b == "low":
        agreement_quality = "core mismatch"
    else:
        agreement_quality = "unknown"

    return {
        # orthogonal statuses
        "spec_status": spec_status,
        "scenario_status": scenario_status,
        "stats_name_status": stats_name_status,
        "agreement_quality": agreement_quality,

        # numeric signals
        "run_agree_ratio_core": core_ratio,
        "run_agree_ratio_bookkeeping": book_ratio,
        "run_agree_ratio_overall": overall_ratio,
        "comparable_core": core.get("comparable", None),
        "mismatched_core": core.get("mismatched", None),
        "comparable_bookkeeping": book.get("comparable", None),
        "mismatched_bookkeeping": book.get("mismatched", None),
        "comparable_overall": overall.get("comparable", None),
        "mismatched_overall": overall.get("mismatched", None),
    }
    return out

sankey_rows = []

rundiff_lut = {}
for helm_row in ub.ProgIter(helm_rows, desc='compare runs'):
    run_dir = ub.Path(helm_row['run_dir'])
    suite_name = run_dir.parent.name
    benchmark_name = run_dir.parent.parent.parent.parent.name
    assert run_dir.parent.parent.parent.name == 'benchmark_output'
    assert run_dir.parent.parent.name == 'runs'
    helm_row['suite_name'] = suite_name
    helm_row['benchmark_name'] = benchmark_name
    run_dir = ub.Path(helm_row['run_dir'])
    run_spec_name = helm_row['run_spec_name']

    kwrow = kwdagger_lut.get(run_spec_name)
    helm_row['reproduced_step1'] = kwrow is not None

    # Base row (always emitted, even if not attempted)
    out = {
        "run_spec_name": run_spec_name,
        "run_dir": str(run_dir),
        "suite_name": suite_name,
        "benchmark_name": benchmark_name,
        "model_name": helm_row['model'],

        # default pipeline fields
        "reproduced_step1": False,
        "attempt_status": None,
        "attempt_error": None,

        # default agreement fields
        "agreement_bucket": None,
        "agreement_bucket_instances": None,
        "run_agree_ratio_core": None,
        "run_agree_ratio_bookkeeping": None,
        "run_agree_ratio_overall": None,
        "inst_agree_ratio_unperturbed": None,
        "inst_agree_ratio_perturbed": None,
        "n_instance_mismatched": None,

        # signatures (fill opportunistically)
        "sig_run_spec": None,
        "sig_scenario": None,
        "sig_stats_name": None,
    }
    out["reproduced_step1"] = (kwrow is not None)
    sankey_rows.append(out)

    if kwrow is None:
        out["attempt_status"] = "not attempted"
        out["agreement_bucket"] = "not attempted"

        helm_row['agreement_bucket_base_task'] = 'not attempted'
        continue

    # raise Exception

    # Attempt exists: try to compare
    out["attempt_status"] = "compared"
    try:
        helm_run = HelmRun.coerce(run_dir)
        kwdg_run = kwrow["run"]

        a = HelmRunAnalysis(helm_run, name="HELM")
        b = HelmRunAnalysis(kwdg_run, name="KWDG")

        # Light signatures: cheap to compute and very useful for debugging.
        sa = a.summary_dict(level=0)
        sb = b.summary_dict(level=0)
        out["sig_run_spec"] = f"{sa['signatures'].get('run_spec_sig')}|{sb['signatures'].get('run_spec_sig')}"
        out["sig_scenario"] = f"{sa['signatures'].get('scenario_sig')}|{sb['signatures'].get('scenario_sig')}"
        out["sig_stats_name"] = f"{sa['signatures'].get('stats_name_sig')}|{sb['signatures'].get('stats_name_sig')}"

        rd = HelmRunDiff(run_a=helm_run, run_b=kwdg_run, a_name="HELM", b_name="KWDG")
        rundiff_lut[run_spec_name] = rd  # save for later drilldown
        out.update(sankey_stats(rd))

    except Exception as ex:
        raise
        out["attempt_status"] = "error"
        out["attempt_error"] = repr(ex)
        out["agreement_bucket"] = "error"

    # # raise Exception

    # helm_run = HelmRun.coerce(run_dir)
    # kwdg_run = kwrow['run']

    # a = HelmRunAnalysis(helm_run)
    # b = HelmRunAnalysis(kwdg_run)

    # if 0:
    #     a.summary(level=10)
    #     b.summary(level=10)

    # rd = HelmRunDiff(
    #     run_a=helm_run, run_b=kwdg_run, a_name='HELM', b_name='KWDG'
    # )
    # self = rd  # NOQA
    # rd.summary(level=1)
    # rd.summarize_instances()

    # if 0:
    #     table1 = rd.a.joined_instance_stat_table()
    #     table2 = rd.a.joined_instance_stat_table()

    #     instance_id = 'id1237'
    #     keys1 = table1.variant_keys_for_instance(instance_id)
    #     keys2 = table2.variant_keys_for_instance(instance_id)
    #     print(f'keys1 = {ub.urepr(keys1, nl=1)}')
    #     print(f'keys2 = {ub.urepr(keys2, nl=1)}')
    #     assert set(keys1) == set(keys2)
    #     for k1 in keys1:
    #         table1.rows_by_variant[k1]
    #         table2.rows_by_variant[k1]

    #     print(rd.drilldown_core_metric_instances())
    #     rd.lookup_instance(('instance_id', 'id14045'), which='a')
    #     rd.lookup_instance(('instance_id', 'id14045'), which='b')

    # raise Exception

    # helm_row.update(rd.summary_base_task())
    # helm_row.update(rd.summary_core())

    # helm_stats = helm_run.json.stats()
    # kwdg_stats = kwdg_run.json.stats()

    # # out = compare.compare_run_pair(helm_stats, kwdg_stats, rel_tol=1e-4, abs_tol=1e-8)
    # helm_row.update(out)


for rd in rundiff_lut.values():
    rd.summary(level=0)
    rd.summary()
    a = rd.a
    b = rd.b
    rd = HelmRunDiff(run_a=a, run_b=b, a_name="HELM", b_name="KWDG")
    spec_a = rd.a.run_spec()
    spec_b = rd.b.run_spec()
    print(f'spec_a = {ub.urepr(spec_a, nl=3)}')
    print(f'spec_b = {ub.urepr(spec_b, nl=3)}')

    if 0:
        rd.summarize_instances()

df = pd.DataFrame(sankey_rows)

df = df[df['attempt_status'] == 'compared']
print(df.value_counts(['benchmark_name', 'reproduced_step1']))
print(df.value_counts(['benchmark_name', 'reproduced_step1', 'spec_status', 'agreement_quality']))
print(df.value_counts(['attempt_status', 'agreement_bucket']).sort_index())


def attempt_status(row: dict[str, object]) -> str:
    return (
        'attempted' if row.get('reproduced_step1', False) else 'not_attempted'
    )


def attempt_label(row: dict[str, object]) -> str:
    # We already computed this in the table builder
    return str(row.get('attempt_status', 'unknown'))


def agreement_label(row: dict[str, object]) -> str:
    # Used in the sankey plan; keep it stable.
    return row.get('agreement_bucket_base_task', 'unknown')


plan = sankey.Plan(
    sankey.Root('Attempted Set'),
    sankey.Group('benchmark', by='benchmark_name'),
    sankey.Bucket('spec', by='spec_status'),
    sankey.Bucket('agreement', by='agreement_quality'),
)

print(plan.to_text())

G = plan.build_sankey(helm_rows, label_fmt='{name}: {value}')
print(G.summarize(max_edges=150))

fig = G.to_plotly(title='HELM Reproduction Funnel')
fpath = 'helm_repro_sankey.jpg'
fig.write_image(fpath)
print(f'Wrote helm_repro_sankey: {fpath}')

if 1:
    print(ub.codeblock(
        f'''
        # On Host
        rm -rf {fpath}
        # Run the wormhole command
        '''))
    ub.cmd(f'wormhole send {fpath}', verbose=3)
    """
    !wormhole send helm_repro_sankey.jpg
    """

# --- Per-benchmark drilldown sankeys (deeper, but still run-level) ---
# Here we drill by model within each benchmark (only if model_name exists).
# If model_name is missing, you can swap this to group by run_spec_name prefix, etc.
bench_groups = ub.group_items(sankey_rows, key=lambda r: r.get('benchmark_name', 'unknown'))

out_dpath = ub.Path('benchmark_sankeys').ensuredir()
for bench, rows in bench_groups.items():
    if bench in {None, 'unknown'}:
        continue

    # Skip tiny groups if you want:
    # if len(rows) < 5: continue

    plan_bench = sankey.Plan(
        sankey.Root(f'{bench}'),
        sankey.Group('model', by='model_name'),
        sankey.Bucket('spec', by='spec_status'),
        sankey.Bucket('agreement', by='agreement_quality'),
    )

    Gb = plan_bench.build_sankey(rows, label_fmt='{name}: {value}')
    figb = Gb.to_plotly(title=f'HELM Repro Funnel: {bench}')

    # sanitize bench for filename
    bench_slug = ub.Path(str(bench)).name.replace('/', '_')
    fpath_b = out_dpath / f'{bench_slug}.jpg'
    figb.write_image(str(fpath_b))
    print(f'Wrote benchmark sankey: {fpath_b}')

if 1:
    print(ub.codeblock(
        f'''
        # On Host
        rm -rf {out_dpath}
        # Run the wormhole command
        '''))
    ub.cmd(f'wormhole send {out_dpath}', verbose=3)
