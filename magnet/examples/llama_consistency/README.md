# Llama consistency example

This directory contains the kwdagger version of the Llama consistency example.
The complete pipeline is declared inline in `llama_kwdagger.yaml`:

```text
llama_predict -> llama_compare
```

`llama_predict.py` reads precomputed HELM-Lite MMLU results and writes the two
model scores for one matrix cell. `llama_compare.py` reads that artifact and
writes the score gap. The card declares `llama_compare` as its `result_node`, so
its fields are exposed to the transitional Python claim as
`metrics.llama_compare.<field>`.

## Why this example has two nodes

The Llama calculation itself does not require a two-stage pipeline.
`llama_predict` could compute the gap and emit the final result directly, and a
one-node pipeline would be simpler for this particular problem.

The extra `llama_compare` stage is kept for now because it makes the small
example exercise a declarative kwdagger edge, artifact handoff, and explicit
`result_node`. It should eventually be replaced by an example whose computation
actually requires two stages rather than preserving a second node only for
pedagogy.

## Prepare the HELM-Lite cache

The example uses Llama results from two public HELM-Lite releases. Download only
the MMLU Llama runs:

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

The card expects the resulting benchmark output at:

```text
./data/crfm-helm-public/lite/benchmark_output
```

The downloader is incremental, so rerunning these commands only fills in data
that is missing or changed.

## Exercise the HELM materializer

MAGNET also provides a single-run materializer for kwdagger pipelines that need
to reuse a cached HELM run or compute one when it is missing. After downloading
the cache above, this command exercises the reuse path without launching HELM:

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
configuration needed to compute a cache miss. `--mode=force_recompute` bypasses
cache reuse.

This Llama card reads the complete downloaded HELM-Lite corpus rather than one
materialized run because each cell compares scores averaged across the MMLU
subjects. The materializer is the lower-level primitive for held-out pipelines
where each missing `(model, dataset)` run is represented by a kwdagger node.

## Run the card with the new evaluator

The kwdagger-native evaluator has a deliberately small CLI. For a local
foreground run:

```bash
magnet evaluate_new \
    magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --output_path ./results_kwdagger \
    --backend serial
```

`--backend` is passed directly to kwdagger. The new evaluator does not use
`MAGNET_QUEUE_BACKEND` or `MAGNET_TMUX_WORKERS` as an internal parameter
transport. For tmux execution, an explicit worker limit can be given with
`--workers`:

```bash
magnet evaluate_new \
    magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --output_path ./results_kwdagger \
    --backend tmux \
    --workers 4
```

Use `--params` to override the card's kwdagger matrix/configuration without
editing the card. For example:

```bash
magnet evaluate_new \
    magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --output_path ./results_kwdagger \
    --backend serial \
    --params='matrix: {llama_predict.base_model: [meta/llama-2-13b]}'
```

The matrix in the checked-in card has six base models and six comparison models,
so `llama_compare` normally produces 36 result cells. KWDagger artifacts are
shared under `./results_kwdagger/_kwdagger`; the MAGNET run directory records
the card provenance and links to those artifacts.

The current claim/verdict layer is transitional. `evaluate_new` lets result-node
values and non-sweep card symbols feed that existing claim machinery, but it
does not run legacy `pipeline:` computation or legacy symbol sweeps. Those
remain available through `magnet evaluate_legacy` and its
`magnet evaluate` compatibility alias until the old evaluator can be retired.

## Pipeline shape

The card uses the standard declarative kwdagger YAML form:

```yaml
kwdagger:
  result_node: llama_compare
  pipeline:
    nodes:
      llama_predict:
        executable: "python -m magnet.examples.llama_consistency.llama_predict"
        # ...
      llama_compare:
        executable: "python -m magnet.examples.llama_consistency.llama_compare"
        # ...
    edges:
      - llama_predict.results_fpath -> llama_compare.scores_fpath
  matrix:
    # ...
```

There is no separate Python pipeline definition. The card owns the DAG
declaration; the Python files implement only the node executables.
