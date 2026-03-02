"""
Notebook style code the developer is working with interactively by copy/pasting
blocks into IPython.

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


for dupname, dupx in ub.find_duplicates([r['run_spec_name'] for r in helm_rows]).items():
    for idx in dupx:
        helm_rows[idx]
    ...



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
    ub.Path('~/code/aiq-magnet/results/helm').expand().glob(
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


def make_bucket_fn(
    edges_desc,
    *,
    nan_label="unknown (no comparable core)",
    endpoints_as_strict_buckets=False,
):
    """
    edges_desc: descending bin edges, e.g. [1.0, 0.90, 0.75, 0.50, 0.0]

    endpoints_as_strict_buckets:
      - False (default): bins are intervals like "0.9–<1.0" plus a top open bin ">=1.0"
      - True: create exact buckets for the endpoints (e.g. "1.0" and "0.0")
    """
    edges = list(edges_desc)
    assert len(edges) >= 2, "need at least two edges"
    assert all(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)), "edges must be descending"

    def _fmt_edge(v: float) -> str:
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        return s if s else "0"

    top = edges[0]
    bot = edges[-1]

    def bucket(x):
        import math
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return nan_label

        # Strict endpoint buckets
        if endpoints_as_strict_buckets:
            if x == top:
                return _fmt_edge(top)
            if x == bot:
                return _fmt_edge(bot)

        # Top open bucket (captures >top, and also ==top when not strict)
        if x >= top:
            return f">={_fmt_edge(top)}"

        # Interior interval buckets: [lo, hi)
        for i in range(1, len(edges)):
            hi = edges[i - 1]
            lo = edges[i]
            if x >= lo:
                return f"{_fmt_edge(lo)}–<{_fmt_edge(hi)}"

        # Below bottom edge
        return f"<{_fmt_edge(bot)}"

    return bucket


def sankey_stats(rd: HelmRunDiff) -> dict:
    """
    Return a small, stable set of fields intended for building Sankey tables.
    """
    s = rd.summary_dict(level=1)
    va = s.get('value_agreement') or {}
    by_class = va.get('by_class') or {}
    overall = va.get('overall') or {}

    core = by_class.get('core') or {}
    book = by_class.get('bookkeeping') or {}

    core_ratio = core.get('agree_ratio', None)
    book_ratio = book.get('agree_ratio', None)
    overall_ratio = overall.get('agree_ratio', None)

    spec_ok = bool(s.get('run_spec_dict_ok', False))
    scen_ok = s.get('scenario_ok', None)  # True/False/None

    spec_status = 'spec match' if spec_ok else 'spec mismatch'
    if scen_ok is None:
        scenario_status = 'scenario unknown'
    else:
        scenario_status = 'scenario match' if scen_ok else 'scenario mismatch'

    # stats_name_status is cheap and very useful as “schema drift” indicator
    cov = s.get('stats_coverage_by_name') or {}
    stats_name_status = (
        'stats names match'
        if (cov.get('only_a', 0) == 0 and cov.get('only_b', 0) == 0)
        else 'stats names mismatch'
    )

    def _bucket_ratio(x: float | None, *, good=0.995, ok=0.95) -> str:
        if x is None:
            return 'unknown'
        if x >= good:
            return 'high'
        if x >= ok:
            return 'medium'
        return 'low'

    core_b = _bucket_ratio(core_ratio)
    book_b = _bucket_ratio(book_ratio)

    if core_b == 'high' and (book_b in {'high', 'unknown'}):
        agreement_quality = 'match'
    elif core_b == 'high' and book_b in {'medium', 'low'}:
        agreement_quality = 'core match, bookkeeping differs'
    elif core_b == 'medium':
        agreement_quality = 'core partial'
    elif core_b == 'low':
        agreement_quality = 'core mismatch'
    else:
        agreement_quality = 'unknown'

    return {
        # orthogonal statuses
        'spec_status': spec_status,
        'scenario_status': scenario_status,
        'stats_name_status': stats_name_status,
        'agreement_quality': agreement_quality,
        # numeric signals
        'run_agree_ratio_core': core_ratio,
        'run_agree_ratio_bookkeeping': book_ratio,
        'run_agree_ratio_overall': overall_ratio,
        'comparable_core': core.get('comparable', None),
        'mismatched_core': core.get('mismatched', None),
        'comparable_bookkeeping': book.get('comparable', None),
        'mismatched_bookkeeping': book.get('mismatched', None),
        'comparable_overall': overall.get('comparable', None),
        'mismatched_overall': overall.get('mismatched', None),
    }
    return row


def modified_wormhole_send(fpath):
    import subprocess
    import selectors

    class NonBlockingPopenIO:
        def __init__(self, p: subprocess.Popen, max_bytes=65536):
            if p.stdout is None or p.stderr is None:
                raise ValueError("Start Popen with stdout=PIPE and stderr=PIPE")
            self.p = p
            self.max_bytes = max_bytes
            self.sel = selectors.DefaultSelector()
            self.sel.register(p.stdout, selectors.EVENT_READ, data="stdout")
            self.sel.register(p.stderr, selectors.EVENT_READ, data="stderr")

        def drain(self, timeout=0.0):
            """
            Read whatever is available right now (bounded), without blocking.
            Returns (stdout_bytes, stderr_bytes).
            """
            out = b""
            err = b""
            for key, _ in self.sel.select(timeout):
                stream = key.fileobj
                name = key.data
                # read1() avoids blocking for "more"; fallback to read()
                reader = getattr(stream, "read1", stream.read)
                data = reader(self.max_bytes)
                if not data:
                    continue
                if name == "stdout":
                    out += data
                else:
                    err += data
            return out, err

    cmd = ['wormhole', 'send', '--no-qr', str(fpath)]

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    import time
    time.sleep(1)

    io = NonBlockingPopenIO(p)

    # call this whenever you want (e.g., in your main loop)
    out, err = io.drain(timeout=0.0)
    if out:
        out_text = out.decode("utf-8", errors="replace")
        print("STDOUT:", out_text, end="")
    if err:
        err_text = err.decode("utf-8", errors="replace")
        print("STDERR:", err_text, end="")
    else:
        err_text = ''

    code = [p for p in err_text.split('\n') if p][-1].split(' ')[-1]

    print(ub.codeblock(
        f"""
        # On the host run:
        rm -rf {fpath}
        # Run the wormhole command
        wormhole recieve {code} --accept-file
        eog {fpath}
        """
    ))
    p.communicate()


sankey_rows = []
rundiffs = []

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
    row = {
        'run_spec_name': run_spec_name,
        'run_dir': str(run_dir),
        'suite_name': suite_name,
        'benchmark_name': benchmark_name,
        'model_name': helm_row['model'],
        # default pipeline fields
        'reproduced_step1': False,
        'attempt_status': None,
        'attempt_error': None,
        # default agreement fields
        'agreement_bucket': None,
        'agreement_bucket_instances': None,
        'run_agree_ratio_core': None,
        'run_agree_ratio_bookkeeping': None,
        'run_agree_ratio_overall': None,
        'inst_agree_ratio_unperturbed': None,
        'inst_agree_ratio_perturbed': None,
        'n_instance_mismatched': None,
        # signatures (fill opportunistically)
        'sig_run_spec': None,
        'sig_scenario': None,
        'sig_stats_name': None,
    }
    row['reproduced_step1'] = kwrow is not None
    sankey_rows.append(row)

    if kwrow is None:
        row['attempt_status'] = 'not attempted'
        row['agreement_bucket'] = 'not attempted'

        helm_row['agreement_bucket_base_task'] = 'not attempted'
        continue

    # raise Exception

    # Attempt exists: try to compare
    row['attempt_status'] = 'compared'
    try:
        helm_run = HelmRun.coerce(run_dir)
        kwdg_run = kwrow['run']

        a = HelmRunAnalysis(helm_run, name='HELM')
        b = HelmRunAnalysis(kwdg_run, name='KWDG')

        # Light signatures: cheap to compute and very useful for debugging.
        sa = a.summary_dict(level=0)
        sb = b.summary_dict(level=0)
        row['sig_run_spec'] = (
            f'{sa["signatures"].get("run_spec_sig")}|{sb["signatures"].get("run_spec_sig")}'
        )
        row['sig_scenario'] = (
            f'{sa["signatures"].get("scenario_sig")}|{sb["signatures"].get("scenario_sig")}'
        )
        row['sig_stats_name'] = (
            f'{sa["signatures"].get("stats_name_sig")}|{sb["signatures"].get("stats_name_sig")}'
        )

        rd = HelmRunDiff(
            run_a=helm_run, run_b=kwdg_run, a_name='HELM', b_name='KWDG'
        )
        rundiffs.append(rd)  # save for later drilldown
        row.update(sankey_stats(rd))
        # Keep the row attached to the rundiff for easy interactive use
        rd.row = row

    except Exception as ex:
        raise
        row['attempt_status'] = 'error'
        row['attempt_error'] = repr(ex)
        row['agreement_bucket'] = 'error'

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

    # # row = compare.compare_run_pair(helm_stats, kwdg_stats, rel_tol=1e-4, abs_tol=1e-8)
    # helm_row.update(row)


DEVELOPER_DETAILED_DIFF_ANALYSIS = True
if DEVELOPER_DETAILED_DIFF_ANALYSIS:
    import json
    with open('rundiff.jsonl', mode='a', encoding='utf8') as file:
        for rd in ub.ProgIter(rundiffs, desc='drill down', verbose=3):
            # rd.summary(level=0)
            # rd.summary()
            a = rd.a
            b = rd.b
            rd = HelmRunDiff(run_a=a, run_b=b, a_name='HELM', b_name='KWDG')

            summary = rd.summary_dict(level=100)
            # json.dumps(summary, ensure_ascii=False)

            # list(kwutil.Json.find_unserializable(summary))
            # summary = kwutil.Json.ensure_serializable(summary)
            file.write(json.dumps(summary, ensure_ascii=False) + "\n")
            file.flush()  # optional; good if you want progress written even if interrupted

            core_agreement = summary['value_agreement']['by_class']['core']

            if 0:
                idx = a.stat_index(drop_zero_count=True, require_mean=True)
                core_a = pd.DataFrame(
                    {k: m for k, m in idx.items() if m.metric_class == 'core'}.values()
                )
                idx = b.stat_index(drop_zero_count=True, require_mean=True)
                core_b = pd.DataFrame(
                    {k: m for k, m in idx.items() if m.metric_class == 'core'}.values()
                )
                spec_a = rd.a.run_spec()
                spec_b = rd.b.run_spec()
                print(f'spec_a = {ub.urepr(spec_a, nl=3)}')
                print(f'spec_b = {ub.urepr(spec_b, nl=3)}')
                rd.summarize_instances()

df = pd.DataFrame(sankey_rows)
df_comp = df[df['attempt_status'] == 'compared']
print(df.value_counts(['benchmark_name', 'reproduced_step1']))
print(
    df.value_counts(
        [
            'benchmark_name',
            'reproduced_step1',
            'spec_status',
            'agreement_quality',
        ]
    )
)
print(df.value_counts(['attempt_status', 'agreement_bucket']).sort_index())


def diagnose_status(row):
    if row['attempt_status'] == 'not attempted':
        return None
    mismatch = set()
    print(f'row={row}')
    if row['spec_status'].split(' ')[-1] == 'mismatch':
        mismatch |= {'run_spec'}
    if row['scenario_status'].split(' ')[-1] == 'mismatch':
        mismatch |= {'scenario_spec'}
    # if row['scenario_status'].split(' ')[-1] == 'mismatch':
    #     mismatch |= {'stats_name'}
    if not mismatch:
        return 'specs match'
    else:
        return 'mismatch: ' + ', '.join(sorted(mismatch))

df['spec_diagnostic'] = [diagnose_status(row) for _, row in df.iterrows()]
df['spec_diagnostic'].value_counts()


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


CORE_IOU_BINS = [
    1.0,
    # 0.90,
    0.75,
    0.50,
    0.25,
    0.00
]
df['core_iou'] = (df['comparable_core'] - df['mismatched_core']) / df[
    'comparable_core'
]
core_iou_bucket = make_bucket_fn(CORE_IOU_BINS, endpoints_as_strict_buckets=True, nan_label='no comparable metrics')
df['core_iou_bucket'] = df['core_iou'].map(core_iou_bucket)
print(df['core_iou_bucket'].value_counts())

df['spec_status'].value_counts()
df['scenario_status'].value_counts()
df['stats_name_status'].value_counts()

# Sankey plan: same skeleton, but add a core IoU bucket stage

# root = sankey.Root('All Attempts')
# # When we get the data if one of the names isn't available, we handle it by
# # dynamically adding it, but here we can use the result object to specify
# # cases.
# splits = root.split(by='attempt_status')
# compared = splits.add_case(value='compared')
# failcase = splits.add_case(value='not attempted')
# failcase.set_label('Failed')

# bench_group = compared.group(by='benchmark_name', name='benchmark')
# bench_group.group(by='core_iou_bucket')


# # Note it should alway be possible to put "benchmarks" before attempt status like:
# root = sankey.Root('All Attempts')
# bench_group = root.group(by='benchmark_name', name='benchmark')
# splits = bench_group.split(by='attempt_status')
# splits['not attempted'].set_label('Failed')
# compared = splits.cases['compared']  # behaves like a defaultdict

# compared.group(by='spec_diagnostic')
# compared.group(by='core_iou_bucket')


# ---
# Does this make sense?


from magnet.utils import sankey_builder
root = sankey_builder.Root()
bench_groups = root.group(by='benchmark_name')

rungroup = bench_groups.group(by='attempt_status')
compared_node = rungroup['compared']
compared_node.label = 'Run'
unrun_node = rungroup['not attempted']
unrun_node.label = 'Failed'

compared_node \
    .group(by='core_iou_bucket')
    # .group(by='spec_diagnostic') \

# plan = sankey.Plan(
#     sankey.Root(f'Attempted Runs n={len(df)}'),
#     sankey.Split('Runs', 'attempt_status', branches={
#         'compared': sankey.Plan(
#             sankey.Group('benchmark', by='benchmark_name'),
#             sankey.Bucket('core_iou', by='core_iou_bucket'),
#         ),
#         'not attempted': sankey.Node('Failed')  # I want any that meet this condition to be sent to a node called failed.
#     })
# )

print(root.to_text())

G = root.build_sankey(df.to_dict('records'), label_fmt='{value}')
print(G.summarize(max_edges=150))

G.nodes['CONST']['label'] = f'Attempted Runs n={len(df)}'
# fig = G.to_plotly(title='HELM Reproduction Funnel')


import plotly.graph_objects as go
node_labels, source, target, value = G._to_sankey_data()
sankey = go.Sankey(
    # arrangement='freeform',
    node=dict(label=node_labels, pad=15, thickness=18),
    link=dict(source=source, target=target, value=value),
)
fig = go.Figure(sankey)
title = 'Title'
fig.update_layout(title_text=title, font_size=14)


fpath = 'helm_repro_sankey.jpg'
fig.write_image(fpath, scale=4.0)
import kwplot
kwplot.cropwhite_ondisk(fpath)
print(f'Wrote helm_repro_sankey: {fpath}')
modified_wormhole_send(fpath)

# --- Per-benchmark drilldown sankeys (deeper, but still run-level) ---
# Here we drill by model within each benchmark (only if model_name exists).
# If model_name is missing, you can swap this to group by run_spec_name prefix, etc.
bench_groups = ub.group_items(
    sankey_rows, key=lambda r: r.get('benchmark_name', 'unknown')
)

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

if 0:
    print(
        ub.codeblock(
            f"""
        # On Host
        rm -rf {out_dpath}
        # Run the wormhole command
        """
        )
    )
    ub.cmd(f'wormhole send {out_dpath}', verbose=3)
