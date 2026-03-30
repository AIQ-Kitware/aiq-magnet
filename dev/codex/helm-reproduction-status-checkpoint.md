# HELM Reproduction Status Checkpoint

Date: 2026-03-30
Workspace: `/home/joncrall/code/aiq-magnet`

## Purpose

This document is a restart-safe checkpoint for the HELM reproduction project.

It is meant to answer:

- what we are trying to prove
- what we have already established
- what still needs to be done before we can make strong publication-quality claims
- what infrastructure work must be preserved as this project is separated from `magnet`

This document is intentionally redundant with other notes, but should be the best
single place to restart from if prior chat context is lost.

## Related Documents

Read these first for deeper context and provenance:

- `dev/codex/helm-reproduction-agent-brief.md`
- `dev/codex/helm-reproduction-research-journal.md`
- `dev/experiments/audit-helm-reproduction/README.md`
- `dev/codex/reproduce-helm-session-v2.md`
- `dev/codex/kwdagger-notes.md`

Important code and experiment roots:

- public historic HELM runs: `/data/crfm-helm-public`
- local reproduced runs: `/data/crfm-helm-audit`
- audit workflow: `dev/experiments/audit-helm-reproduction`

## Core Research Goal

The real paper question is not:

- "can we reproduce every HELM run exactly?"

The better question is:

- "to what extent are public HELM results independently reproducible under an explicit open-weight local recipe, and how should residual differences be interpreted?"

For a strong paper, we need a precise answer to four sub-questions:

1. Same-machine repeatability:
   how stable are local reruns under the same recipe?
2. Cross-machine repeatability:
   how stable are local reruns across different GPU hardware and host environments?
3. Official-vs-local reproducibility:
   when we compare to historic public HELM, how close are we on core benchmark outcomes?
4. Scope:
   for what subset of HELM can these comparisons be run fairly under an open-weight local recipe?

## Current Headline Conclusion

The strongest current result is positive:

- a corrected local Vicuna no-chat recipe reproduces a small but meaningful HELM subset very well
- this holds on multiple machines
- the remaining cross-machine differences are largely bookkeeping-style drift, not core-metric failure

The best current positive evidence is for:

- `boolq:model=lmsys/vicuna-7b-v1.3,data_augmentation=canonical`
- `mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model=lmsys/vicuna-7b-v1.3,data_augmentation=canonical`
- `narrative_qa:model=lmsys/vicuna-7b-v1.3,data_augmentation=canonical`

with:

- same-machine repeatability
- cross-machine repeatability
- excellent official-vs-local agreement on core metrics

This is the current basis for the paper framing:

- open-weight HELM reproduction is possible for a clearly defined subset
- exact equality is too strong a target
- core-metric agreement and bounded drift are the right scientific objects

## Critical Resolved Issue

An earlier apparent failure was real in the sense that the outputs were bad, but
it was not evidence against reproducibility.

Root cause:

- the local HF Vicuna execution path was auto-applying a chat template
- this caused severe empty-completion pathologies, especially on `narrative_qa`

Fix:

- use `dev/experiments/audit-helm-reproduction/configs/debug/vicuna_no_chat_template.yaml`
- specifically, the corrected local deployment must use `apply_chat_template: false`

Scientific lesson:

- execution-path bugs can masquerade as irreproducibility
- any claim of irreproducibility must be preceded by recipe validation diagnostics

## Questions We Have Answered Already

### 1. Are exact mismatches always evidence of irreproducibility?

No.

We have already shown that:

- some mismatches were caused by local recipe errors
- some mismatches are bookkeeping/runtime drift
- some large-grid failures are simply "not runnable under the current recipe"

### 2. Is the corrected Vicuna local recipe repeatable on the same machine?

Yes, for the current evaluated subset.

The corrected Vicuna no-chat runs show strong same-machine repeatability on the
current core tasks.

### 3. Is the corrected Vicuna local recipe repeatable across machines?

Yes, for the tested subset.

Current evidence supports positive cross-machine reproducibility across:

- the internal main machine
- `namek`
- `yardrat`

with the strongest evidence currently on:

- BoolQ
- MMLU `us_foreign_policy`
- NarrativeQA

### 4. Are cross-machine differences currently dominated by benchmark failure?

No, not for the successful Vicuna no-chat subset.

The current pairwise cross-machine reports are being diagnosed mainly as:

- `bookkeeping_metric_drift`

rather than core-metric collapse.

### 5. Are many large-grid failures actually recipe-scope failures rather than negative reproducibility evidence?

Yes.

Observed buckets include:

- gated datasets
- dataset download failures
- scenarios needing external credentials or proprietary annotators
- model packaging issues

These belong in a runnable-scope taxonomy, not in a blanket "HELM is irreproducible" claim.

### 6. Is apples-to-apples alignment on `max_eval_instances` enough by itself to explain the old official-vs-local drift?

No.

We already controlled this confounder, and substantial drift still persisted in
the older Pythia-style cases.

### 7. Is local-vs-local noise generally much smaller than the big official-vs-local failures seen in the older cases?

Yes.

That earlier result remains important:

- local repeat noise exists
- but the older structural official-vs-local failures were much larger than ordinary rerun variance

This is part of the causal story for why "recipe-corrected subset succeeds" is a
real scientific result rather than noise.

## Questions We Can Partially Answer But Not Yet Claim Broadly

### 1. How broad is the reproducible open-weight subset?

Partial answer:

- at least the current corrected Vicuna subset is reproducible on key tasks

Not yet established:

- how far this extends across more benchmarks
- how far this extends across more model families
- whether we can support a broad claim beyond this subset

### 2. Is reproducibility achievable on realistic consumer hardware?

Partial answer:

- yes for some tested runs on single- or modest-multi-GPU machines

Not yet established:

- the final size and diversity of the consumer-hardware reproducible subset
- a rigorous paper-ready inclusion criterion for "consumer-accessible"

### 3. Are the remaining cross-machine differences scientifically negligible?

Likely yes for the current successful subset, but we still need cleaner final summaries
and paper-ready language around:

- what is core metric drift
- what is bookkeeping drift
- what is acceptable reproducibility tolerance

## Questions We Still Need To Answer

These are the main open scientific questions.

### 1. What exact subset will the paper claim is reproducible?

We need a final inclusion rule, not an ad hoc list.

The paper subset should likely require:

- open-weight model
- local HF execution
- no proprietary judge/annotator dependency
- dataset available without private credentials
- runs that fit on realistic hardware
- at least one official-vs-local comparison
- at least one same-machine repeat
- at least one cross-machine repeat where feasible

### 2. How many benchmarks and scenarios in the candidate open-weight set are:

- reproducible
- runnable but not yet reproducible
- not currently runnable under the recipe

This needs a final table and failure taxonomy with counts, not just anecdotes.

### 3. How much of the positive result is specific to Vicuna?

This is still open.

We need to decide whether the paper makes:

- a model-specific claim
- a recipe-class claim
- or a broader open-weight HELM claim supported by more than one model family

### 4. What is the best scientific notion of agreement?

We already know strict equality is too brittle.

We still need to settle the paper language for:

- core-metric exact agreement
- agreement under tolerance sweeps
- benchmark-level reproducibility envelopes
- bookkeeping vs substantive drift

### 5. What should count as a reproducibility failure versus a recipe-scope exclusion?

This needs explicit policy language.

Without this, readers can reasonably misinterpret the large-grid failures.

### 6. What is the minimal cross-machine evidence needed for a strong claim?

We likely need a clean answer to:

- how many machines
- how different the machines need to be
- which benchmarks need cross-machine replication

### 7. What should the final paper-facing machine labels and hardware descriptions be?

We now have the infrastructure for this:

- `dev/experiments/audit-helm-reproduction/configs/paper_label_mappings.yaml`

But the final wording for figures, tables, and methods still needs to be chosen.

## Loose Ends That Need To Be Tied Up

### 1. Finish analysis on the latest external-machine results

We need final refreshed summaries and top-level rollups that include:

- `aiq-gpu` vs external-machine pairwise comparison lines
- paper-facing labels
- updated figure layouts

### 2. Confirm and document missing or incomplete runs explicitly

Example:

- `namek` had incomplete coverage for `narrative_qa` at one point

We need final status accounting so incomplete coverage is not confused with negative evidence.

### 3. Turn the large-grid failure list into a formal taxonomy table

Needed outputs:

- failure categories
- counts
- representative examples
- which categories are exclusion criteria vs genuine reproducibility negatives

### 4. Define the paper subset formally

We need one checked-in manifest or selection file that becomes the canonical:

- reproducibility-ready subset

This should not live only in our heads or only in generated outputs.

### 5. Decide whether to expand beyond Vicuna for the final claim

If yes:

- select one or more additional open-weight families
- run a smaller targeted grid
- compare whether the positive result generalizes

If no:

- narrow the paper claim clearly and honestly to the validated recipe scope

### 6. Build paper-ready aggregate tables

We still need tables that are easy to cite in a paper:

- per-run benchmark table
- per-machine comparison table
- failure taxonomy table
- runnable subset summary table

### 7. Finalize the methods story

The paper methods section should clearly explain:

- public historic bundle used
- local audit execution path
- local deployment override
- same-machine repeat protocol
- cross-machine repeat protocol
- analysis metrics and tolerance sweeps
- interpretation of bookkeeping vs core drift

### 8. Make sure the latest results and code are restart-safe

This means:

- current docs up to date
- canonical paths documented
- commands reproducible
- key generated reports easy to rebuild

## What Still Needs To Be Accomplished For Strong Claims

To make strong and defensible claims, we should complete the following sequence.

### Phase 1. Lock the subset and evidence base

- finalize the reproducibility-ready benchmark subset
- finish the pending external-machine runs
- ensure each claimed benchmark has the needed comparison types

Desired output:

- a clear list of benchmark/model pairs that the paper will center

### Phase 2. Quantify scope

- produce counts for:
  - candidate open-weight runs
  - runnable runs
  - successfully reproduced runs
  - excluded runs by failure category

Desired output:

- a paper table and figure showing scope and exclusions

### Phase 3. Quantify drift

- summarize:
  - same-machine repeatability
  - cross-machine repeatability
  - official-vs-local reproducibility

Desired output:

- benchmark-level and collection-level drift summaries
- agreement curves or summary statistics appropriate for a paper

### Phase 4. Generalize or narrow honestly

Choose one:

- broaden support by validating more open-weight families
- or keep the claim narrow and precise around the validated Vicuna no-chat subset

Desired output:

- final claim wording that matches the evidence exactly

### Phase 5. Write the paper narrative

The likely narrative arc is:

1. naive reproduction can fail for avoidable recipe reasons
2. once recipe confounders are fixed, a meaningful open-weight HELM subset becomes reproducible
3. reproducibility should be analyzed as bounded drift, not just exact equality
4. many apparent failures in broad sweeps are actually scope failures, not evidence against reproducibility

## Proposed Research Plan From Here

### Short-term plan

1. Finish refreshing the latest experiment summaries and overall aggregates.
2. Produce a final large-grid failure taxonomy with counts.
3. Define the canonical paper subset manifest and commit it.
4. Create one paper-facing summary table for:
   - same-machine
   - cross-machine
   - official-vs-local
5. Decide whether we need one additional model family for external validity.

### Medium-term plan

1. Run any last targeted experiments needed to close coverage gaps.
2. Freeze the methods and inclusion criteria.
3. Draft the core result section and the failure-taxonomy section in parallel.

### Paper-writing plan

The analysis should be organized around three claims:

1. Positive claim:
   a defined open-weight HELM subset is reproducible under a corrected local recipe.
2. Measurement claim:
   reproducibility is better characterized by core-metric agreement and bounded drift than by exact equality.
3. Scope claim:
   many failures in broad HELM sweeps are exclusions of the current recipe, not direct evidence of irreproducibility.

## Infrastructure / Repository Split Note

This experiment is likely to be separated into its own repository so it can be
independent from `magnet`.

That split should preserve two categories of work:

### 1. Audit-specific logic that belongs in the new repo

Examples:

- manifest generation
- run indexing
- pairwise comparison reports
- aggregate reporting
- failure taxonomy analysis
- paper-facing labeling and plotting utilities

### 2. General-purpose changes that should be kept and pushed upstream from `magnet`

Examples already relevant:

- `magnet.backends.helm.cli.materialize_helm_run`
  - support for `model_deployments_fpath`
  - robust handling of optional CLI placeholder values
  - provenance capture via `ProcessContext`
  - preservation of useful materialization metadata/logging behavior

These changes are not merely one-off experiment hacks. They improve the general
reproducibility and observability of HELM materialization and should not be lost
when the audit code is extracted.

## Practical Restart Checklist

If starting fresh:

1. Read:
   - `dev/codex/helm-reproduction-agent-brief.md`
   - `dev/codex/helm-reproduction-status-checkpoint.md`
   - `dev/codex/helm-reproduction-research-journal.md`
2. Inspect:
   - `dev/experiments/audit-helm-reproduction/README.md`
3. Confirm data roots:
   - `/data/crfm-helm-public`
   - `/data/crfm-helm-audit`
4. Rebuild the latest reports from:
   - `dev/experiments/audit-helm-reproduction/scripts/index_results.sh`
   - `dev/experiments/audit-helm-reproduction/scripts/analyze_experiment_from_index.sh`
   - `dev/experiments/audit-helm-reproduction/scripts/aggregate_core_reports.sh`
5. Review the latest experiment and aggregate summaries in:
   - `dev/experiments/audit-helm-reproduction/reports/`
6. Continue from the short-term plan above.

## Bottom Line

The project is no longer at the stage of asking whether anything works at all.

We already have a real positive reproducibility result for an open-weight HELM subset.

The next stage is to:

- formalize the subset
- quantify scope and exclusions
- tighten the drift analysis
- decide how broad the final claim should be
- separate the audit project cleanly from `magnet` without losing the upstream-worthy infrastructure improvements
