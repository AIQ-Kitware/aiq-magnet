# Llama consistency example

This directory contains the kwdagger version of the Llama consistency example.
The complete pipeline is declared inline in `llama_kwdagger.yaml`:

```text
materialize_run -> llama_evaluate
```

`materialize_run.py` materializes one precomputed HELM-Lite MMLU run per
`(model, subject)`. `llama_evaluate.py` receives those runs as a gathered
collection, averages each model's exact_match across the subjects, and writes
the score gap. The card declares `llama_evaluate` as its `result_node`, so
KWDagger aggregate loads its available results and exposes them to the
transitional Python claim as `llama_evaluate.<field>` -- or
`metrics.llama_evaluate.<field>` qualified.

`llama_predict.py` is still here because the legacy `cards/llama_pipeline.yaml`
runs it. The kwdagger recipe no longer does.

## Why this example has two nodes

Because the second node needs the first, and the edge is the interesting part.

`materialize_run` turns each MMLU run the comparison needs into an artifact of
its own -- reused from the downloaded HELM cache by symlink, or computed when
the pipeline has the deployment configuration for it. `llama_evaluate` receives
those artifacts as a **gathered collection** and scores them.

Nothing scans a directory. The set of runs a verdict rests on is declared by the
matrix, resolved when the pipeline is compiled, and recorded in the result, so a
run that lands in a shared HELM cache later cannot silently join an average.
That is the difference between "the data was somewhere on disk" and "these runs
are the inputs to this number".

```text
materialize_run[model, subject]        one artifact per HELM run
     |
     |  gather (group_by: []) -> newline-delimited manifest
     v
llama_evaluate[base_model, comp_model] averages subjects, reports the gap
```

`group_by: []` hands every evaluate cell the whole declared corpus, and the cell
picks its two models out of it. Grouping by model instead would not work: a cell
needs runs for *two* models, which is a self-join rather than a group.

## Provide a HELM-Lite cache

`materialize_run` reuses runs from a local HELM cache; it does not fetch them.
The recipe runs `mode: reuse_only`, so a run it cannot find is an error rather
than a silently smaller average. Something has to put the runs on disk first.

**If you already have a HELM corpus**, skip the download and point the recipe at
it -- any directory containing nested `benchmark_output` directories works:

```bash
magnet evaluate_new \
    magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --params="matrix: {materialize_run.precomputed_root: /path/to/crfm-helm-public}" \
    --output_path ./results_kwdagger --backend serial
```

**Otherwise**, download the MMLU Llama runs from the two public HELM-Lite
releases the example uses:

```bash
magnet download helm \
    --download_dir ./data/crfm-helm-public \
    --benchmark=lite \
    --version=v1.0.0 \
    --runs='regex:mmlu.*model=.*llama.*'

magnet download helm \
    --download_dir ./data/crfm-helm-public \
    --benchmark=lite \
    --version=v1.2.0 \
    --runs='regex:mmlu.*model=.*llama.*'
```

That is where the card looks by default:

```text
./data/crfm-helm-public
```

The downloader is incremental, so rerunning these commands only fills in data
that is missing or changed.

The third option is `mode: compute_if_missing`, which runs HELM to produce a run
it cannot reuse. That needs a working HELM deployment and real model access, so
it is not how you try the example.

## Exercise the HELM materializer directly

The recipe drives `materialize_run` for you, but the underlying single-run
materializer is worth seeing on its own. After downloading the cache above:

```bash
python -m magnet.backends.helm.cli.materialize_helm_run \
    --run_entry='mmlu:subject=philosophy,model=meta/llama-2-13b' \
    --suite=llama-materialize-smoke \
    --out_dpath=./results/materialized/llama-2-13b-philosophy \
    --precomputed_root=./data/crfm-helm-public \
    --mode=reuse_only \
    --materialize=symlink
```

The output directory contains the selected HELM run under
`benchmark_output/runs/`, an `adapter_manifest.json`, and a `DONE` sentinel.
Use `--mode=compute_if_missing` when a pipeline has the model deployment
configuration needed to compute a cache miss; `--mode=force_recompute` bypasses
reuse entirely.

The recipe's node is a thin wrapper over this. It takes `--model` and
`--subject` separately rather than one composed `run_entry`, so the card's
matrix sweeps the two axes it actually means:

```bash
python -m magnet.examples.llama_consistency.materialize_run \
    --model=meta/llama-2-13b --subject=philosophy \
    --precomputed_root=./data/crfm-helm-public \
    --out_dpath=./results/materialized/llama-2-13b-philosophy
```

## The declared subject list

MMLU-Lite has 57 subjects. The recipe declares a handful, because averaging over
a *declared* subset is the price of not globbing -- and it is what makes the
number reproducible. Widen `materialize_run.subject` to widen the average; the
job count grows with `models x subjects`, and each materialized run is cached
and shared across all 36 comparison cells.

## Run the recipe with the new evaluator

`evaluate_new` forwards the KWDagger schedule controls it exposes using the
same names and semantics as `kwdagger schedule`. MAGNET does not resolve
backends, synthesize queue settings, or use environment variables as an
internal parameter transport.

For a local foreground run:

```bash
magnet evaluate_new \
    magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --output_path ./results_kwdagger \
    --backend serial
```

For tmux execution, `--tmux_workers` is the native KWDagger worker control:

```bash
magnet evaluate_new \
    magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --output_path ./results_kwdagger \
    --backend tmux \
    --tmux_workers 4
```

The cache/reuse controls are also KWDagger's own options. `--skip_existing=1`
avoids submitting nodes whose expected products already exist. `--cache=1`
(the default) lets submitted node commands guard themselves against existing
outputs. To request recomputation using KWDagger's native semantics, disable
both mechanisms:

```bash
magnet evaluate_new \
    magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --output_path ./results_kwdagger \
    --backend serial \
    --skip_existing=0 \
    --cache=0
```

`--max_configs` is useful for a matrix smoke test without changing the recipe:

```bash
magnet evaluate_new \
    magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --output_path ./results_kwdagger \
    --backend serial \
    --max_configs=1
```

Node-level selection remains part of the KWDagger pipeline configuration
(e.g. `node.__enabled__` in a matrix/config row); `evaluate_new` does not add a
second interpretation of it.

These controls limit the work requested by this invocation. Evidence selection
is a separate recipe setting. `evidence.scope: all` evaluates every compatible
row accumulated under the shared output root. `evidence.scope: requested`
evaluates only result-node computations requested by this invocation, including
requested computations that KWDagger satisfies from an existing cached output.
The Llama recipe uses `requested`, so `--max_configs=1` produces a one-cell
MAGNET result snapshot even when older Llama results already exist in the
KWDagger store.

### Load the results in the visualization dashboard

The output from `evaluate_new` is directly compatible with the existing MAGNET
visualization dashboard. After running the recipe, create a dashboard upload ZIP
whose root contains the MAGNET run directories. The shared `_kwdagger` result
store is not needed by the dashboard and should be left out of the upload:

```bash
(
    cd results_kwdagger
    zip -ry ../results_kwdagger-dashboard.zip . \
        -x '_kwdagger' '_kwdagger/*'
)
```

If the dashboard is not already checked out, start it with:

```bash
git clone https://github.com/AIQ-Kitware/eval-card-viz.git ../eval-card-viz
cd ../eval-card-viz
uv run visualization.py ./evaluations
```

Open the dashboard in the browser, click **Upload Local Run (.zip)** in the top
right, and select `results_kwdagger-dashboard.zip` from the MAGNET checkout.
There is no additional conversion or copy into the dashboard's `evaluations/`
tree for this local-upload path.

Each run in the ZIP contains the legacy dashboard contract:
`card.yaml`, `log`, `results/*/verdict.json`, and `verdict.json`. The per-run
`symbols` mapping is populated with the qualified KWDagger values consumed by
the claim, so the existing dashboard can show the model pair, scores, threshold,
and gap without any dashboard-side changes.

Use `--params` to override the recipe's kwdagger matrix/configuration without
editing the recipe. For example:

```bash
magnet evaluate_new \
    magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --output_path ./results_kwdagger \
    --backend serial \
    --params='matrix: {llama_predict.base_model: [meta/llama-2-13b]}'
```

The matrix in the checked-in recipe has six base models and six comparison
models, so one full scheduling request asks for 36 comparisons. KWDagger
artifacts accumulate under `./results_kwdagger/_kwdagger`. After scheduling,
MAGNET uses KWDagger aggregate to discover all currently available
`llama_compare` results, then the recipe's `evidence.scope: requested` keeps the
rows corresponding to this invocation's matrix. Switching the scope to `all`
lets a sequence of partial campaigns build and reevaluate a growing evidence
set over time.

Each MAGNET invocation gets its own run directory. `requested_runs.json`
records the operational state of the processes requested by that invocation,
while `verdict.json` is computed only from available aggregate rows. A failed or
not-yet-started job is therefore visible as execution provenance without being
interpreted as evidence that the claim is false.

The current claim/verdict layer is transitional. `evaluate_new` lets KWDagger
aggregate values and non-sweep recipe symbols feed that existing claim
machinery, but it
does not run legacy `pipeline:` computation or legacy symbol sweeps. Those
remain available through `magnet evaluate_legacy` and its
`magnet evaluate` compatibility alias until the old evaluator can be retired.

## Pipeline shape

The recipe uses the standard declarative kwdagger YAML form:

```yaml
kwdagger:
  result_node: llama_evaluate
  pipeline:
    nodes:
      materialize_run:
        executable: "python -m magnet.examples.llama_consistency.materialize_run"
        out_paths: {out_dpath: ".", done_fname: DONE}
        primary_out_key: done_fname
        # A materializer yields an artifact, not a measurement. Aggregate loads
        # a result node's predecessors too, so this says there is nothing to
        # read rather than let the generic loader parse DONE as JSON.
        load_result: "...materialize_run.load_kwdagger_result"
      llama_evaluate:
        executable: "python -m magnet.examples.llama_consistency.llama_evaluate"
        in_paths: [run_dpaths]
        # ...
    edges:
      - src: materialize_run.out_dpath
        dst: llama_evaluate.run_dpaths
        gather:
          group_by: []
          order_by: [model, subject]
          require: all_success
  matrix:
    # ...
```

There is no separate Python pipeline definition. The recipe owns the DAG
declaration; the Python files implement the node executables.

`llama_evaluate` writes KWDagger's generic result envelope -- metrics under
`result.metrics` -- so it needs no `load_result` of its own. The one on
`materialize_run` exists for the opposite reason: to declare that the node has
no result to load. A Python node says that by not defining the method; a
declarative node inherits the generic one whether it wants it or not.
