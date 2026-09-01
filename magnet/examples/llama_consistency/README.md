# Llama consistency example

This directory contains the kwdagger version of the Llama consistency example.
The complete pipeline is declared inline in `llama_kwdagger.yaml`:

```text
materialize_run -> llama_predict -> llama_compare
```

`materialize_run.py` materializes one HELM-Lite MMLU run per `(model,
subject)`. `llama_predict.py` receives those runs as a gathered collection and
averages each model's exact_match across the subjects. `llama_compare.py`
reduces the two scores to their gap. The card declares `llama_compare` as its
`result_node`, so KWDagger aggregate loads its available results and exposes
them to the transitional Python claim as `llama_compare.<field>` -- or
`metrics.llama_compare.<field>` qualified.

## Why this example has three nodes

The first edge is the one that matters, and the second is what compatibility
costs.

`materialize_run` turns each MMLU run the comparison needs into an artifact of
its own. `llama_predict` receives those artifacts as a **gathered collection**
and scores them. Nothing scans a directory: the set of runs a verdict rests on
is declared by the matrix, resolved when the pipeline is compiled, and recorded
in the result, so a run that lands in a shared HELM cache later cannot silently
join an average. That is the difference between "the data was somewhere on disk"
and "these runs are the inputs to this number".

```text
materialize_run[model, subject]        one artifact per HELM run
     |
     |  gather (group_by: []) -> newline-delimited manifest
     v
llama_predict[base_model, comp_model]  averages the gathered subjects
     |
     |  results.json
     v
llama_compare[base_model, comp_model]  reduces the two scores to their gap
```

`group_by: []` hands every predict cell the whole declared corpus, and the cell
picks its two models out of it. Grouping by model instead would not work: a cell
needs runs for *two* models, which is a self-join rather than a group.

The score and the gap stay in separate nodes because `llama_predict` is the
same executable the legacy `cards/llama_pipeline.yaml` runs. It accepts either a
gather manifest or a corpus directory, so both cards share one implementation of
the HELM scoring instead of growing a second copy -- and it has to emit scores
for the legacy card, while this card's claim wants the gap. That is a better
reason for the split than the one this example used to give.

## Provide a HELM-Lite cache, or let the pipeline compute

`materialize_run` runs `mode: compute_if_missing`: it reuses a run from a local
HELM cache when it finds one, and runs HELM to produce it when it does not.
Computing needs a working HELM deployment and real model access, so in practice
you give it a cache and it reuses everything.

**If you already have a HELM corpus**, point the recipe at it -- any directory
containing nested `benchmark_output` directories works:

```bash
magnet evaluate_new \
    magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --params="matrix: {materialize_run.precomputed_root: /path/to/crfm-helm-public}" \
    --output_path ./results_kwdagger --backend serial
```

**Otherwise**, download the MMLU Llama runs from the two public HELM-Lite
releases the example uses, into the location the card looks in by default:

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

The downloader is incremental, so rerunning these commands only fills in data
that is missing or changed.

Set `mode: reuse_only` to make a cache miss an error instead of a HELM run --
which is what the test suite does, so a miss there is a bug rather than a cue to
go compute something.

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

To see what a run would do without doing it, `--dry_run` schedules with
`run=0`. The matrix compiles and `requested_runs.json` reports the whole
campaign, but nothing is submitted -- useful for checking the job count before
committing to a materialize sweep:

```bash
magnet evaluate_new \
    magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --output_path ./results_kwdagger --dry_run
```

Nothing is judged: the result is `NOT_EVALUATED` and no verdict is written,
whatever the shared store already holds. A dry run answers what would be
submitted, and that is all it answers.

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
  result_node: llama_compare
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
      llama_predict:
        executable: "python -m magnet.examples.llama_consistency.llama_predict"
        in_paths: [run_dpaths]
        # ...
      llama_compare:
        executable: "python -m magnet.examples.llama_consistency.llama_compare"
        in_paths: [scores_fpath]
        # ...
    edges:
      - src: materialize_run.out_dpath
        dst: llama_predict.run_dpaths
        gather:
          group_by: []
          order_by: [model, subject]
          require: all_success
      - llama_predict.results_fpath -> llama_compare.scores_fpath
  matrix:
    # ...
```

There is no separate Python pipeline definition. The recipe owns the DAG
declaration; the Python files implement the node executables and, for these
flat JSON artifacts, small KWDagger result loaders. New node formats can use
KWDagger's generic result envelope instead.

The loader on `materialize_run` exists for the opposite reason to the other
two: to declare that the node has *no* result to load. A Python node says that
by not defining the method; a declarative node inherits the generic one whether
it wants it or not.
