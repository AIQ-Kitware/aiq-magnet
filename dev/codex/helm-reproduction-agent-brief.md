# HELM Reproduction Agent Brief

Date: 2026-03-30
Workspace: `/home/joncrall/code/aiq-magnet`

## Mission

Prepare a fresh agent to continue the HELM reproducibility project quickly and safely.

The overarching goal is:

- determine how much of public HELM is independently reproducible with local open-weight execution
- quantify the remaining differences rather than treating reproducibility as a binary
- build a paper-quality methodology around:
  - same-machine repeatability
  - cross-machine repeatability
  - official-vs-local drift
  - benchmark-level and collection-level summaries

For publication framing, the likely paper story is:

- a positive result for a carefully defined runnable subset of HELM
- with quantified bounds on unavoidable residual differences
- plus a clear failure taxonomy for cases that are not currently runnable or not reproducible under the open local recipe

## Key Principle

Do **not** interpret every failed large-grid job as a reproducibility failure.

Many failures so far are recipe-scope failures:

- gated datasets
- missing external credentials
- dataset download failures
- model packaging issues
- scenarios requiring proprietary annotators

Those belong in the "not currently runnable under the recipe" bucket, not the "HELM is irreproducible" bucket.

## Current Scientific Conclusion

The strongest current positive signal is the corrected Vicuna no-chat recipe.

For:

- `boolq:model=lmsys/vicuna-7b-v1.3,data_augmentation=canonical`
- `mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model=lmsys/vicuna-7b-v1.3,data_augmentation=canonical`
- `narrative_qa:model=lmsys/vicuna-7b-v1.3,data_augmentation=canonical`

once `apply_chat_template: false` is used for the local HF deployment:

- same-machine reruns are stable
- cross-machine reruns are stable
- official-vs-local agreement on core metrics is excellent
- remaining drift is mostly output length/style rather than correctness

This is the current best evidence that a substantial subset of HELM is independently reproducible.

## Root Cause Already Found

There was a real false alarm earlier.

Problem:

- local HF Vicuna runs were auto-applying chat templating
- this caused severe empty-completion pathologies, especially on `narrative_qa`

Fix:

- use:
  - `dev/experiments/audit-helm-reproduction/configs/debug/vicuna_no_chat_template.yaml`

Effect:

- empty completions disappeared
- core metrics became close or identical to official public HELM

This is a critical lesson:

- local execution-path bugs can masquerade as reproducibility failures
- any strong irreproducibility claim should be preceded by recipe validation diagnostics

## Important Paths

Public historic bundle:

- `/data/crfm-helm-public`

Local reproduced runs:

- `/data/crfm-helm-audit`

Main audit workflow:

- `dev/experiments/audit-helm-reproduction`

Research notes:

- `dev/codex/helm-reproduction-research-journal.md`
- `dev/codex/reproduce-helm-session-v2.md`
- `dev/codex/kwdagger-notes.md`

## Operational State Of Recent Experiments

### 1. `audit-vicuna-nochat-overnight`

This is the fixed-config baseline on the main machine.

Status:

- successful
- core reproducibility result is positive

### 2. `audit-vicuna-nochat-server`

Initial attempt:

- failed due to GPU occupancy / OOM
- logs captured this clearly

Later successful attempt:

- matched the earlier fixed Vicuna result
- supports positive cross-machine reproducibility

### 3. `audit-namek-subset`

Subset intended for consumer-ish single-GPU cross-hardware checks.

Observed:

- BoolQ: complete
- MMLU: complete
- NarrativeQA: incomplete / hanging at time of inspection

Completed jobs match the same positive Vicuna no-chat pattern.

### 4. `audit-yardrat-subset`

Observed:

- all 3 target jobs completed
- terminal impression of failure was misleading
- copied artifacts show valid `DONE` files and healthy logs

Scientific interpretation:

- Yardrat supports the same positive cross-machine result as the other successful Vicuna no-chat runs

### 5. `audit-historic-grid`

Large broad candidate sweep built from `run_specs.yaml` / `run_details.yaml`.

Observed:

- total jobs: `286`
- completed jobs: `141`
- failed or incomplete: `145`

This grid is useful mainly for failure taxonomy and runnable-subset refinement.

## Large Grid Failure Taxonomy

Representative buckets already observed:

### Dataset / scenario access issues

- RAFT:
  - requires `trust_remote_code=True`
- MATH:
  - dataset resolution failure for `hendrycks/competition_math`
- bAbI:
  - failed on direct `wget` download
- Natural Questions:
  - failed on direct `wget` download

### Gated / credentialed cases

- XSTest:
  - gated dataset
- annotation-heavy runs:
  - may require OpenAI credentials

### Model packaging / environment issues

- Sea Lion:
  - missing dynamic import files such as `flash_attn_triton.py`

### Important interpretation

These are not currently evidence against reproducibility.

They are evidence that the current candidate suite is too broad and must be filtered to a reproducibility-ready subset.

## Runnable Recipe We Should Move Toward

The reproducibility recipe should prioritize:

- open-weight models
- local HF execution path
- models that fit on realistic consumer-accessible hardware
  - single large consumer GPU
  - or modest multi-GPU workstation
- datasets available without private credentials
- scenarios not requiring proprietary judge/annotator APIs
- configurations we can archive and rerun deterministically enough for audit purposes

In practice this likely means:

- do not center the paper on the entire raw HELM catalog
- instead define a reproducibility-ready subset with explicit inclusion criteria

## Existing Tooling Worth Reusing

### Historic candidate generation

- `dev/poc/inspect_historic_helm_runs.py`
- outputs:
  - `run_specs.yaml`
  - `run_details.yaml`

### Audit workflow

- `dev/experiments/audit-helm-reproduction/scripts/run_from_manifest.sh`
- `dev/experiments/audit-helm-reproduction/scripts/compare_batch.sh`
- `dev/experiments/audit-helm-reproduction/scripts/index_results.sh`
- `dev/experiments/audit-helm-reproduction/scripts/analyze_experiment_from_index.sh`
- `dev/experiments/audit-helm-reproduction/scripts/aggregate_core_reports.sh`

### Manifest building

- `dev/experiments/audit-helm-reproduction/scripts/make_historic_grid_manifest.sh`
- `dev/experiments/audit-helm-reproduction/scripts/make_machine_subset_manifest.sh`
- `dev/experiments/audit-helm-reproduction/scripts/make_machine_shard_manifest.sh`

### Important note

If running the analysis tooling on this machine, prefer:

- `/home/joncrall/code/aiq-magnet/.venv/bin/python`

because the default `python` here may not have `helm` installed.

## Reporting / Analysis State

The audit folder already contains:

- per-run-spec core metric reports
- management summaries
- threshold-vs-agreement plots
- raw score distribution plots
- ECDF plots
- run-level metric tables
- instance sample reports
- experiment-level summaries
- overall reproducibility rollups

Important conceptual improvement already adopted:

- focus on reproducibility distributions / envelopes, not single exact-equality claims

## What A Fresh Agent Should Do Next

### 1. Curate a reproducibility-ready subset

Turn `run_specs.yaml` into a filtered suite that excludes:

- gated datasets
- credentialed annotator cases
- obviously broken download paths
- models with known packaging failures

This should become the main benchmark set for the paper.

### 2. Quantify the distribution of differences

For the filtered subset:

- same-machine repeatability
- cross-machine repeatability
- official-vs-local drift

Summaries should include:

- per-instance agreement
- run-level metric deltas
- benchmark-family rollups
- model-family rollups

### 3. Keep correctness and style drift separate

Important current pattern:

- core correctness metrics often match exactly or nearly exactly
- output length can still differ substantially

This distinction should be central in the paper.

### 4. Make the runnable recipe explicit

Document:

- required configs
- model overrides
- environment assumptions
- hardware assumptions
- known excluded benchmark/model families and why they are excluded

### 5. Continue failure taxonomy

For the broad grid:

- classify each failure into:
  - unrunnable dataset
  - missing credential
  - model packaging/runtime issue
  - true reproduction drift
  - unknown

This can become a useful table or appendix.

## Current Bottom Line

The project has moved from:

- "maybe HELM is not reproducible"

to something more precise:

- a meaningful subset of HELM appears reproducible under a corrected local open-weight recipe
- many apparent failures are recipe or environment issues, not scientific counterexamples
- the next paper-quality step is to formalize the subset and quantify the reproducibility envelope
