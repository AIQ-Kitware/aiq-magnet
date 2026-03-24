# Audit HELM Reproduction

This folder contains a reusable, shell-first workflow for reproducing selected
historic HELM runs with the local `kwdagger` pipeline, then comparing the
reproduced outputs against the public HELM bundle.

The design goals are:

- work out of the box in the current environment,
- remain portable to a fresh operator via environment variables,
- keep heavyweight run artifacts outside the repo by default,
- start with a small smoke-test batch,
- scale later to larger reproduction attempts without redesign.

## Defaults

These defaults are chosen to work in the current environment:

- `AIQ_MAGNET_ROOT=/home/joncrall/code/aiq-magnet`
- `AIQ_PYTHON=python`
- `HELM_PRECOMPUTED_ROOT=/data/crfm-helm-public`
- `AUDIT_RESULTS_ROOT=/data/crfm-helm-audit`

Assumptions:

- `magnet` is importable from `AIQ_PYTHON`
- `kwdagger` is on `PATH`
- `helm-run` is on `PATH`

Optional environment variables:

- `CUDA_VISIBLE_DEVICES`
- `AUDIT_DEFAULT_MAX_EVAL_INSTANCES`
- `AUDIT_DEFAULT_TMUX_WORKERS`

## Folder Layout

- `configs/`
  Checked-in manifests and templates.
- `scripts/`
  Shell entrypoints for operators.
- `python/`
  Python helpers used by the shell scripts.
- `reports/`
  Lightweight comparison reports.
- `examples/`
  Example command sequences.

## Quick Start

```bash
cd $HOME/code/aiq-magnet
```

1. Validate the environment:

```bash
dev/experiments/audit-helm-reproduction/scripts/check_env.sh
```

2. Materialize a smoke-test manifest:

```bash
dev/experiments/audit-helm-reproduction/scripts/make_smoke_manifest.sh
```

By default this writes:

```text
dev/experiments/audit-helm-reproduction/configs/generated/smoke_manifest.generated.yaml
```

3. Launch the smoke-test batch on the GPU machine:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
dev/experiments/audit-helm-reproduction/scripts/run_smoke.sh
```

4. Compare the completed batch to the historic HELM bundle:

```bash
dev/experiments/audit-helm-reproduction/scripts/compare_batch.sh
```

Reports are written by default to:

```text
dev/experiments/audit-helm-reproduction/reports/<experiment_name>/
```

Heavy run outputs are written by default to:

```text
/data/crfm-helm-audit/<experiment_name>/
```

## Manifest Schema

The workflow uses a small YAML manifest as the unit of experiment definition.

Fields:

- `schema_version`
- `experiment_name`
- `description`
- `run_entries`
- `max_eval_instances`
- `suite`
- `mode`
- `materialize`
- `backend`
- `devices`
- `tmux_workers`
- `local_path`
- `precomputed_root`
- `require_per_instance_stats`
- `model_deployments_fpath`
- `enable_huggingface_models`
- `enable_local_huggingface_models`

See:

- `configs/smoke_manifest.yaml`
- `configs/manifest_template.yaml`

## Smoke-Test Batch

The checked-in smoke batch is intentionally small and uses only models that
already have built-in Hugging Face deployments in HELM:

- `eleutherai/pythia-6.9b`
- `lmsys/vicuna-7b-v1.3`

Tasks:

- `mmlu:subject=us_foreign_policy,...`
- `boolq:...`
- `narrative_qa:...`

Total:

- 6 runs

Defaults:

- `max_eval_instances=100`
- low worker count
- no custom deployment override YAML

## Scaling Up

For larger experiments:

1. copy `configs/manifest_template.yaml`,
2. expand `run_entries`,
3. optionally introduce `model_deployments_fpath` for Together-only families,
4. run:

```bash
dev/experiments/audit-helm-reproduction/scripts/run_from_manifest.sh /path/to/manifest.yaml
dev/experiments/audit-helm-reproduction/scripts/compare_batch.sh /path/to/manifest.yaml
```

## Notes

- This workflow is intended for execution on an external GPU machine.
- The current development environment does not need a GPU to generate manifests
  or comparison commands.
- Comparison relies on `HelmRunDiff` and will emit JSONL, JSON, text, and
  optional sankey artifacts when the required plotting dependencies are
  available.
