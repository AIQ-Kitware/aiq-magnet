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

To materialize the first apples-to-apples control manifest instead:

```bash
dev/experiments/audit-helm-reproduction/scripts/make_apples_manifest.sh
```

By default this writes:

```text
dev/experiments/audit-helm-reproduction/configs/generated/smoke_manifest.generated.yaml
```

And the apples-to-apples variant writes:

```text
dev/experiments/audit-helm-reproduction/configs/generated/apples_manifest.generated.yaml
```

3. Launch the smoke-test batch on the GPU machine:

```bash
dev/experiments/audit-helm-reproduction/scripts/run_smoke.sh
```

To change which GPUs `kwdagger` schedules onto, set the manifest `devices`
field or regenerate the manifest with `--devices`, for example:

```bash
dev/experiments/audit-helm-reproduction/scripts/make_smoke_manifest.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/smoke_manifest.generated.yaml \
  --devices 2,3
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

The runner also derives a distinct `kwdagger` queue name from the experiment
name, which helps avoid interactive tmux collision prompts when multiple audit
batches have been launched on the same machine.

## Reproducibility Checklist

For any experiment you want to cite later, preserve all of the following:

- the exact manifest YAML used to launch the run
- the exact results root under:
  - `/data/crfm-helm-audit/<experiment_name>/`
- the generated comparison reports under:
  - `dev/experiments/audit-helm-reproduction/reports/<experiment_name>/`
- the current git commit of `aiq-magnet`
- the Python executable used as `AIQ_PYTHON`
- the value of:
  - `AIQ_MAGNET_ROOT`
  - `HELM_PRECOMPUTED_ROOT`
  - `AUDIT_RESULTS_ROOT`

Recommended capture commands:

```bash
git rev-parse HEAD
which "$AIQ_PYTHON"
dev/experiments/audit-helm-reproduction/scripts/check_env.sh
```

If you need to transfer the run to another machine for analysis, transfer:

- the manifest YAML
- the report directory for the experiment
- the raw results directory for the experiment

Minimum useful transfer set:

- report directory only, if you only need summaries
- report directory plus 1-2 representative raw job directories, if you need direct run artifact inspection
- full raw experiment directory, if you may need to rerun local comparisons later

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
- `configs/apples_manifest.yaml`
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

## Apples-To-Apples Smoke Batch

The first apples-to-apples control batch reuses the same 6 smoke entries, but
aligns `max_eval_instances` with the historic public bundle for those entries:

- `max_eval_instances=1000`
- experiment name: `audit-smoke-apples`
- suite: `audit-smoke-apples`
- devices are still controlled by the manifest `devices` field or `--devices`

This is the preferred first batch when the goal is reproduction fidelity rather
than just workflow validation.

Suggested operator flow:

```bash
dev/experiments/audit-helm-reproduction/scripts/make_apples_manifest.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/apples_manifest.generated.yaml \
  --devices 0,1

dev/experiments/audit-helm-reproduction/scripts/run_from_manifest.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/apples_manifest.generated.yaml

dev/experiments/audit-helm-reproduction/scripts/compare_batch.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/apples_manifest.generated.yaml
```

Raw reproduced outputs are written to:

```text
/data/crfm-helm-audit/audit-smoke-apples/
```

Comparison reports are written to:

```text
dev/experiments/audit-helm-reproduction/reports/audit-smoke-apples/
```

Files to inspect first:

- `management_summary_<timestamp>.txt`
- `compare_summary_<timestamp>.txt`
- `compare_cases_<timestamp>.jsonl`

Files to transfer back for local analysis:

- the entire report directory
- optionally the entire raw results directory if deeper run-by-run inspection is needed

Example transfer commands depend on your setup, but a simple pattern is:

```bash
ls -td dev/experiments/audit-helm-reproduction/reports/audit-smoke-apples/*
ls -td /data/crfm-helm-audit/audit-smoke-apples/*
```

Then transfer the newest report files and, if needed, the raw experiment root.

## Exact Reproduction Cases Used In This Research

The following cases were used as the first reproducibility controls in this
research thread.

### Apples-To-Apples Control Batch

Purpose:

- compare current local kwdagger reproductions against official public HELM with matched `max_eval_instances`

Manifest generation:

```bash
dev/experiments/audit-helm-reproduction/scripts/make_apples_manifest.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/apples_manifest.generated.yaml \
  --devices 0,1
```

Run:

```bash
dev/experiments/audit-helm-reproduction/scripts/run_from_manifest.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/apples_manifest.generated.yaml
```

Compare:

```bash
dev/experiments/audit-helm-reproduction/scripts/compare_batch.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/apples_manifest.generated.yaml
```

Outputs:

- raw results:
  - `/data/crfm-helm-audit/audit-smoke-apples/`
- reports:
  - `dev/experiments/audit-helm-reproduction/reports/audit-smoke-apples/`

### Pairwise Repeatability Control: BoolQ / Pythia

Purpose:

- measure ordinary local rerun drift on the same benchmark/model pair

Manifest files used:

- `dev/experiments/audit-helm-reproduction/configs/generated/boolq_pythia_r1.yaml`
- `dev/experiments/audit-helm-reproduction/configs/generated/boolq_pythia_r2.yaml`

Run:

```bash
dev/experiments/audit-helm-reproduction/scripts/run_from_manifest.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/boolq_pythia_r1.yaml

dev/experiments/audit-helm-reproduction/scripts/run_from_manifest.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/boolq_pythia_r2.yaml
```

Direct pairwise compare of the two completed runs:

```bash
dev/experiments/audit-helm-reproduction/scripts/compare_pair.sh \
  /data/crfm-helm-audit/audit-boolq-pythia-r1/helm/helm_id_13jkx9mm4k4n/benchmark_output/runs/audit-boolq-pythia-r1/boolq:model=eleutherai_pythia-6.9b,data_augmentation=canonical \
  /data/crfm-helm-audit/audit-boolq-pythia-r2/helm/helm_id_12jr5w48kge7/benchmark_output/runs/audit-boolq-pythia-r2/boolq:model=eleutherai_pythia-6.9b,data_augmentation=canonical \
  dev/experiments/audit-helm-reproduction/reports/pairwise/boolq-pythia-repeat
```

### Pairwise Official-vs-Local Control: BoolQ / Pythia

Purpose:

- compare one local reproduced run directly against the matched public HELM run

Direct compare:

```bash
dev/experiments/audit-helm-reproduction/scripts/compare_pair.sh \
  /data/crfm-helm-public/classic/benchmark_output/runs/v0.3.0/boolq:model=eleutherai_pythia-6.9b,data_augmentation=canonical \
  /data/crfm-helm-audit/audit-boolq-pythia-r1/helm/helm_id_13jkx9mm4k4n/benchmark_output/runs/audit-boolq-pythia-r1/boolq:model=eleutherai_pythia-6.9b,data_augmentation=canonical \
  dev/experiments/audit-helm-reproduction/reports/pairwise/boolq-pythia-historic
```

### Viewing The Key Reports

```bash
cat dev/experiments/audit-helm-reproduction/reports/pairwise/boolq-pythia-repeat-wide/pair_report_20260327T011202Z.txt
cat dev/experiments/audit-helm-reproduction/reports/pairwise/boolq-pythia-historic-wide/pair_report_20260327T011202Z.txt
```

These two reports are the current best compact illustration of:

- local repeatability drift being small
- official-vs-local drift being much larger
- global tolerance sweeps needing careful interpretation because metric scales differ across classes

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

## Pairwise Run Reports

To compare any two concrete HELM run directories directly, use:

```bash
dev/experiments/audit-helm-reproduction/scripts/compare_pair.sh \
  /path/to/run_a \
  /path/to/run_b \
  dev/experiments/audit-helm-reproduction/reports/pairwise
```

This writes:

- `pair_report_<timestamp>.json`
- `pair_report_<timestamp>.txt`

The pairwise report includes:

- strict diff diagnosis
- raw run-level distance distributions
- raw instance-level distance distributions
- tolerance sweeps across several preset thresholds

Example using the local repeated `boolq + gpt2` runs:

```bash
dev/experiments/audit-helm-reproduction/scripts/compare_pair.sh \
  /data/crfm-helm-audit/audit-boolq-gpt2-r1/helm/helm_id_lh2zobnkhuwi/benchmark_output/runs/audit-boolq-gpt2-r1/boolq:model=openai_gpt2,data_augmentation=canonical \
  /data/crfm-helm-audit/audit-boolq-gpt2-r2/helm/helm_id_lvb1vuf32m2g/benchmark_output/runs/audit-boolq-gpt2-r2/boolq:model=openai_gpt2,data_augmentation=canonical \
  dev/experiments/audit-helm-reproduction/reports/pairwise
```

Then inspect:

```bash
ls -td dev/experiments/audit-helm-reproduction/reports/pairwise/*
cat dev/experiments/audit-helm-reproduction/reports/pairwise/pair_report_<timestamp>.txt
```

## Notes

- This workflow is intended for execution on an external GPU machine.
- The current development environment does not need a GPU to generate manifests
  or comparison commands.
- Comparison relies on `HelmRunDiff` and will emit JSONL, JSON, text, and
  optional sankey artifacts when the required plotting dependencies are
  available.
