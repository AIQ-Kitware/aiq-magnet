# Llama consistency example

This directory contains the kwdagger version of the Llama consistency example.
The complete two-node pipeline is declared inline in
`llama_kwdagger.yaml`:

```text
llama_predict -> llama_compare
```

`llama_predict.py` reads precomputed HELM-Lite MMLU results and writes the two
model scores for one matrix cell. `llama_compare.py` reads that artifact and
writes the score gap. The card declares `llama_compare` as its `result_node`, so
its result fields are available to the legacy claim as
`metrics.llama_compare.<field>`.

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

MAGNET also provides a single-run materializer used by kwdagger pipelines that
need to reuse a cached HELM run or compute it when missing. After downloading
the cache above, this command checks the reuse path without running HELM:

```bash
python -m magnet.backends.helm.cli.materialize_helm_run \
    --run_entry='mmlu:subject=philosophy,model=meta/llama-2-13b' \
    --suite=llama-materialize-smoke \
    --out_dpath=./results/materialized/llama-2-13b-philosophy \
    --precomputed_root=./data/crfm-helm-public \
    --mode=reuse_only \
    --materialize=symlink
```

The output directory contains the HELM run under `benchmark_output/runs/`, an
`adapter_manifest.json`, and a `DONE` sentinel. Change `--mode` to
`compute_if_missing` when a pipeline has the model deployment configuration
needed to compute a cache miss. `force_recompute` bypasses cache reuse.

This Llama card itself reads the complete downloaded HELM-Lite corpus rather
than one materialized run because each cell compares scores averaged across the
MMLU subjects. The materializer is the lower-level primitive used by held-out
pipelines where each missing `(model, dataset)` run is a kwdagger node.

## Run the card

For a local foreground run, use the serial queue backend:

```bash
MAGNET_QUEUE_BACKEND=serial \
magnet evaluate magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --results_path ./results_kwdagger
```

The matrix has six base models and six comparison models, so the
`llama_compare` result node produces 36 cells. KWDagger artifacts are shared
under `./results_kwdagger/_kwdagger`; the MAGNET run directory records the card
provenance and links to those artifacts.

The current legacy claim path evaluates each result through
`metrics.llama_compare`. With the public HELM-Lite data used by the regression
test, at least one model pair exceeds the configured `0.1` threshold.

## Pipeline shape

The card uses the standard declarative kwdagger YAML format:

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

There is no separate Python pipeline definition. The card is the authoritative
DAG declaration; the Python files implement only the node executables.
