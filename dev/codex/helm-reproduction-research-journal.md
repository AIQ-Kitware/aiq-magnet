# HELM Reproduction Research Journal

Date started: 2026-03-27
Project: `/home/joncrall/code/aiq-magnet`
Scope: reproduce historic public HELM results with local `kwdagger` pipelines, explain observed drift, and build reporting that supports both management summaries and maintainer-grade technical diagnosis.

## Research Goal

Determine whether current locally reproduced HELM runs meaningfully differ from official public HELM results, and if so:

- quantify how different they are
- separate ordinary rerun noise from structural drift
- identify likely causes such as deployment/provider changes, metric-spec changes, and execution-spec changes

## Data Sources

- Public historic bundle:
  - `/data/crfm-helm-public`
- Local audit runs:
  - `/data/crfm-helm-audit`
- Audit experiment workflow:
  - `dev/experiments/audit-helm-reproduction`

## Workflow Built During This Research

- Created a reusable audit experiment folder under:
  - `dev/experiments/audit-helm-reproduction`
- Added shell-first operator scripts for:
  - environment validation
  - manifest generation
  - scheduling runs with `kwdagger`
  - comparing completed batches to public HELM runs
  - direct pairwise comparison of two concrete run directories
- Added pairwise reporting support that writes:
  - JSON reports
  - text summaries
  - tolerance sweeps

## Current Direction Shift

We likely do not need more same-hardware repeat runs right now.

- The current evidence is already strong that repeated `kwdagger` runs on the
  same machine are very stable relative to the official-vs-local gap.
- The more interesting next source of variation is now cross-hardware /
  cross-machine drift.
- That means future repeatability work should prioritize:
  - different GPU architectures
  - different VRAM tiers
  - possibly different host environments

This should reduce time spent waiting on redundant same-machine reruns while
improving the causal story for a paper-quality reproducibility analysis.

## Important Audit / kwdagger Notes

- `kwdagger` is the right execution mechanism for these experiments, but it has a few behaviors worth remembering:
  - unset optional params may still be rendered into generated CLI commands
  - list-valued params are best passed as YAML strings, not `nargs='*'`
  - tmux queue name collisions can cause interactive prompts unless queue names are experiment-specific
- Pairwise and entry-based comparison tooling now validates finalized HELM run artifacts before attempting diffs.
  - This avoids misleading Python tracebacks when a job exists but its run directory is incomplete.

See also:
- `dev/codex/kwdagger-notes.md`
- `dev/codex/reproduce-helm-session-v2.md`

## Provenance / Multi-Machine Goal

Current status before this note:

- historical audit results mostly recorded logical run config
- they did **not** explicitly record machine / GPU provenance in a structured,
  analysis-friendly way

This is a problem if we want to:

- merge runs produced on different machines
- compare same-config runs across hardware
- avoid blocking on one machine just to obtain intermediate analyses

Implementation direction:

- record lightweight process provenance per materialized HELM job
- use `kwutil.ProcessContext` to capture:
  - host
  - user
  - cwd
  - OS / Python info
  - memory / CPU metadata when available
- augment that with best-effort `nvidia-smi` GPU details when available

Desired downstream behavior:

- pairwise / aggregate analysis should work against whatever runs exist
- missing runs should reduce coverage, not block intermediate analysis
- when cross-machine drift appears, we should be able to attribute it to a
  known machine / GPU context rather than reconstructing provenance manually

Open goal:

- teach the audit reporting layer to group / filter by machine provenance once
  enough multi-machine runs exist

## Deployment / Registry Findings

- Some target models already have built-in HELM Hugging Face deployments:
  - `eleutherai/pythia-6.9b`
  - `lmsys/vicuna-7b-v1.3`
  - `aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct`
- Some newer or different families appear Together-backed in the built-in registry:
  - example: `meta/llama-3-8b-chat`
- HELM’s experimental Hugging Face registration flags are not reliable enough by themselves for this workflow.
  - robust reproduction work should prefer explicit deployment control, e.g. via custom `model_deployments.yaml`

## Apples-To-Apples Finding

The original smoke batch was useful as a workflow check, but not apples-to-apples because public matches were often at `max_eval_instances=1000` while the reproduced smoke runs used `100`.

We then ran an apples-to-apples control batch aligned to `max_eval_instances=1000`.

Key result:

- the eval-size mismatch confounder was removed
- the remaining observed drift still persisted

That means the disagreement is not explained solely by comparing `100` local examples to `1000` historic examples.

## Representative Structural Drift

For apples-to-apples cases, the main remaining execution-path difference was:

- `adapter_spec.model_deployment`

Representative examples showed:

- historic public runs often record:
  - `adapter_spec.model_deployment = null`
- local reproduced runs often record:
  - `adapter_spec.model_deployment = huggingface/...`

Metric spec drift was also observed:

- historic often uses:
  - `helm.benchmark.metrics.basic_metrics.BasicMetric`
- local reproduced runs often use:
  - `BasicGenerationMetric`
  - `BasicReferenceMetric`
  - `InstancesPerSplitMetric`

This is important because it means the disagreement is not just at the output level. The effective evaluation configuration is also changing.

## Pairwise Comparison Baseline

To separate structural drift from normal stochastic variation, repeated local runs were compared directly.

Case studied in detail:

- `boolq:model=eleutherai/pythia-6.9b,data_augmentation=canonical`

Compared:

- local repeat 1:
  - `/data/crfm-helm-audit/audit-boolq-pythia-r1/helm/helm_id_13jkx9mm4k4n/benchmark_output/runs/audit-boolq-pythia-r1/boolq:model=eleutherai_pythia-6.9b,data_augmentation=canonical`
- local repeat 2:
  - `/data/crfm-helm-audit/audit-boolq-pythia-r2/helm/helm_id_12jr5w48kge7/benchmark_output/runs/audit-boolq-pythia-r2/boolq:model=eleutherai_pythia-6.9b,data_augmentation=canonical`
- official public HELM:
  - `/data/crfm-helm-public/classic/benchmark_output/runs/v0.3.0/boolq:model=eleutherai_pythia-6.9b,data_augmentation=canonical`

### Local Repeatability Result

`r1` vs `r2`:

- diagnosis:
  - `bookkeeping_metric_drift`
- strict run-level agree ratio:
  - `0.9552238805970149`
- strict instance-level agree ratio:
  - `0.9523809523809523`
- run-level max abs delta:
  - `0.0015118565559387176`
- instance-level max abs delta:
  - `0.4418189525604248`

Interpretation:

- local reruns are very close
- residual differences are mostly bookkeeping/runtime style noise, not large task-quality drift

### Official vs Local Result

Official `v0.3.0` vs local `r1`:

- diagnosis:
  - `multiple_primary_reasons`
- primary reason names:
  - `deployment_drift`
  - `execution_spec_drift`
- strict run-level agree ratio:
  - `0.4626865671641791`
- strict instance-level agree ratio:
  - `0.6577333333333333`
- run-level abs p90:
  - `4.0`
- run-level abs max:
  - `11.985`
- instance-level abs p90:
  - `4.0`
- instance-level abs max:
  - `75.73884344100952`

Interpretation:

- official-vs-local drift is much larger than local-vs-local drift
- that makes it very unlikely that ordinary nondeterminism is the main explanation

## High-Level Verdict

For the tested BoolQ/Pythia apples-to-apples case, we now have enough information to say:

- the current locally reproduced HELM result is significantly different from the official public HELM result
- the difference is much larger than the observed local repeatability noise
- the most likely explanation is structural/configurational drift rather than ordinary stochastic rerun variance

Important limitation:

- this claim is solid for the tested case
- it should not automatically be generalized to all HELM benchmarks or model families without more cases

## Tolerance Sweep Findings

The pairwise tool supports tolerance sweeps across several preset thresholds.

Current presets:

- `strict`: `abs_tol=0.0`, `rel_tol=0.0`
- `tiny`: `abs_tol=1e-12`, `rel_tol=1e-6`
- `small`: `abs_tol=1e-9`, `rel_tol=1e-4`
- `medium`: `abs_tol=1e-6`, `rel_tol=1e-3`
- `loose`: `abs_tol=1e-3`, `rel_tol=1e-2`
- `xloose`: `abs_tol=1e-2`, `rel_tol=1e-1`
- `xxloose`: `abs_tol=1e-1`, `rel_tol=1.0`
- `extreme`: `abs_tol=1.0`, `rel_tol=10.0`

For local BoolQ/Pythia repeats:

- similarity becomes nearly perfect by `loose` / `xloose`

For official vs local BoolQ/Pythia:

- similarity stays much lower through `loose` and `xloose`
- it only collapses to `1.0` at the very permissive `xxloose` threshold

Interpretation:

- official-vs-local mismatch is robust to modest tolerance relaxation
- forcing them to look identical requires very permissive tolerances

## Important Metric-Scale Caveat

A single global absolute tolerance is not easy to interpret because the comparison spans different metric families with very different numeric scales.

Observed for official vs local BoolQ/Pythia:

- `core` metrics are approximately `[0, 1]`
  - examples:
    - `exact_match`
    - `prefix_exact_match`
    - `quasi_exact_match`
  - run-level max abs delta in this class was only about:
    - `0.021`
- `bookkeeping` metrics are not bounded to `[0, 1]`
  - examples:
    - `num_bytes`
    - `num_output_tokens`
    - `logprob`
  - observed values included:
    - `num_bytes`: `15.432` vs `3.448`
    - per-instance `num_bytes`: `16.0` vs `3.0`
    - per-instance `num_output_tokens`: `5.0` vs `1.0`
    - per-instance `logprob`: values like `-3.64` vs `0.0`

Implication:

- `abs_tol=1.0` is extremely loose for bounded core metrics
- but it is not necessarily huge for bookkeeping metrics such as byte counts or token counts

Therefore:

- global tolerance sweeps are still useful as diagnostics
- but interpretation should move toward:
  - per metric class tolerances
  - and eventually per metric family tolerances

## Current Reporting Files Worth Inspecting

- Session journal:
  - `dev/codex/reproduce-helm-session-v2.md`
- kwdagger notes:
  - `dev/codex/kwdagger-notes.md`
- Pairwise report, local repeat:
  - `dev/experiments/audit-helm-reproduction/reports/pairwise/boolq-pythia-repeat-wide/pair_report_20260327T011202Z.txt`
- Pairwise report, official vs local:
  - `dev/experiments/audit-helm-reproduction/reports/pairwise/boolq-pythia-historic-wide/pair_report_20260327T011202Z.txt`

## Recommended Next Steps

- Add class-specific tolerance reporting:
  - at least separate `core` from `bookkeeping`
- Add effect-size style summaries:
  - compare official-vs-local distance to local-vs-local baseline distance
- Collect more repeat runs if formal significance estimates are desired
  - current data is enough for a strong effect-size-style argument
  - it is not enough for a stable p-value estimate
- Run a minimal Together-backed control on a representative case
  - this should help separate provider/deployment effects from general HELM evolution

## Direct NarrativeQA/Vicuna Debug Run

We ran a focused local debug job outside kwdagger scheduling:

- suite: `debug-narrative-vicuna-direct`
- run entry: `narrative_qa:model=lmsys/vicuna-7b-v1.3,data_augmentation=canonical`
- max_eval_instances: `20`

Main result:

- The direct run reproduces the same failure mode as the larger local NarrativeQA/Vicuna runs.
- This strongly argues against the issue being a kwdagger scheduling/orchestration bug.

Observed from the raw run:

- request count: `100`
- empty completions: `99`
- non-empty completions: `1`
- mean output token count: `0.13`
- token count histogram:
  - `0`: `99`
  - `13`: `1`

Observed from `stats.json`:

- `num_completion_tokens` mean on `test`: `0.0`
- `num_output_tokens` mean on `test`: `0.0`
- `finish_reason_unknown` mean on `test`: `1.0`
- `exact_match`, `quasi_exact_match`, `f1_score`, `rouge_l`: all `0.0`

Relevant log clues:

- HELM automatically set `apply_chat_template=True`
- HELM removed 4 in-context examples to fit the context window
- HELM logged stop/truncation warnings:
  - `truncate_sequence needs to strip "\\n"`
  - `truncate_sequence needs to strip "</s>"`

Current interpretation:

- Most likely this is a local HELM Hugging Face Vicuna execution/configuration issue.
- The strongest current suspect is chat-template application on a non-chat-style NarrativeQA prompt.
- Secondary suspects are newline stop-sequence handling and/or immediate EOS/empty-generation behavior in the HF Vicuna path.

Current follow-up experiment:

- rerun the same benchmark with a custom `model_deployments.yaml` override that sets:
  - `apply_chat_template: false`

## NarrativeQA/Vicuna Root Cause Update

The `apply_chat_template: false` rerun strongly supports a root-cause diagnosis.

Run:

- suite: `debug-narrative-vicuna-nochat`
- same benchmark/model family as the failing debug run
- same local Hugging Face deployment name
- overridden deployment config:
  - `client_spec.args.apply_chat_template: false`

Observed:

- request count: `500`
- empty completions: `0`
- non-empty completions: `500`
- mean output token count: `12.894`

Aggregate test metrics from the corrected local run:

- `exact_match`: `0.2727`
- `quasi_exact_match`: `0.4026`
- `f1_score`: `0.6422`
- `rouge_l`: `0.6442`
- `bleu_1`: `0.5138`
- `bleu_4`: `0.0722`

These are now close to the official public HELM run for the same benchmark/model pair.

Conclusion:

- The prior NarrativeQA/Vicuna failure was **not** good evidence of irreproducibility.
- It was caused by a local HELM/HuggingFace configuration issue.
- The main culprit appears to be automatic chat-template application for this run.

Implications for the audit:

- For local Hugging Face reproductions, `apply_chat_template` must be treated as an explicit controlled setting.
- Some earlier "failed reproductions" may need to be reinterpreted or rerun if they depended on HELM's automatic chat-template inference.
- The audit/reporting system should surface suspicious signals such as:
  - high empty-completion rate
  - near-zero `num_output_tokens`
  - pervasive `finish_reason_unknown`
