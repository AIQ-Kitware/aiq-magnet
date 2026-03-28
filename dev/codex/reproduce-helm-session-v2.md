# Reproduce HELM Session V2

Date: 2026-03-24
Workspace: `/home/joncrall/code/aiq-magnet`
Python env: `/home/agent/.local/uv/envs/uvpy3.13.2/bin/python`
HELM installation: assumed available in the Python environment used by `magnet`
Precomputed HELM root: `/data/crfm-helm-public`

## Task Summary

Take a second pass at reproducing historic HELM evaluations with the local `kwdagger` pipelines, with a focus on cases where historic runs used Together-hosted model deployments but could plausibly be rerun locally with HELM's Hugging Face client on the available 4 x 96 GB GPUs.

We also want to preserve useful discoveries here because rate limits are nearly exhausted for this session.

## Findings So Far

- The rough notes at the top of `dev/poc/inspect_historic_helm_runs.py` match the earlier workflow:
  - generate candidate run specs from `/data/crfm-helm-public`
  - filter them
  - schedule `magnet.backends.helm.pipeline.helm_single_run_pipeline()`
- `dev/poc/inspect_historic_helm_runs.py` already contains an important filtering heuristic:
  - it registers HELM built-in configs
  - inspects model metadata and deployments from HELM
  - keeps only models whose resolved deployment client is `helm.clients.huggingface_client.HuggingFaceClient`
  - it also limits to open-access text models with `num_parameters <= 10e9`
- That means the previous workflow may have avoided explicit Together-to-HuggingFace rewrites in many cases by selecting only models that HELM already knew how to run through a Hugging Face deployment.
- Concrete registry examples from this session:
  - `aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct`
    - default deployment available in HELM: `huggingface/llama3-8b-cpt-sea-lionv2.1-instruct`
    - client: `helm.clients.huggingface_client.HuggingFaceClient`
  - `lmsys/vicuna-7b-v1.3`
    - default deployment available in HELM: `huggingface/vicuna-7b-v1.3`
  - `eleutherai/pythia-6.9b`
    - default deployment available in HELM: `huggingface/pythia-6.9b`
  - `meta/llama-3-8b-chat`
    - built-in default deployment appears to be `together/llama-3-8b-chat`
    - no built-in Hugging Face deployment was present in the registry snapshot checked today

## Historic Together Cases

- `rg` over `/data/crfm-helm-public/**/run_spec.json` confirms many newer public bundles explicitly record Together deployments, e.g.:
  - `together/deepseek-v3`
  - `together/qwen2.5-7b-instruct-turbo`
  - `together/qwen2.5-72b-instruct-turbo`
  - `together/llama-3.2-11b-vision-instruct-turbo`
  - `together/llama-3.2-90b-vision-instruct-turbo`
  - `together/gpt-oss-20b`
- So the second-pass reproduction work really does need a principled local-deployment override path for some model families.

## Pipeline / Code Notes

- `magnet/backends/helm/pipeline.py` defines a thin kwdagger node around:
  - `python -m magnet.backends.helm.cli.materialize_helm_run`
- `magnet/backends/helm/cli/materialize_helm_run.py` behavior today:
  - tries to reuse a matching run from `precomputed_root`
  - otherwise executes:
    - `helm-run --run-entries <requested_desc> --suite <suite> ...`
  - no provider override or deployment rewrite option is exposed yet
- Matching of historic runs is based on run-entry token subset matching, with model normalization `/ -> _`
- Reuse search scans discovered `benchmark_output` trees and returns the best matching run directory

## Immediate Hypothesis

- If a historic run spec points at a Together-backed deployment, reproducing it locally likely requires one of:
  - rewriting the `model=` token in the run-entry to a HELM deployment name that uses `HuggingFaceClient`
  - or adding custom `prod_env/model_deployments.yaml` entries so the same logical model can resolve to a local Hugging Face deployment
  - or both

## Important HELM Behavior

- `helm-run` calls:
  - `register_builtin_configs_from_helm_package()`
  - then `register_configs_from_directory(args.local_path)`
- Default `--local-path` is `prod_env`
- Because `materialize_helm_run.py` runs `helm-run` with `cwd=out_dpath`, a relative `local_path=prod_env` naturally maps to:
  - `<job_dir>/prod_env`
- This gives us a clean way to inject per-job `model_deployments.yaml` overrides without modifying the shared HELM checkout.

## New Code Support Added This Session

- Updated `magnet/backends/helm/cli/materialize_helm_run.py` to support:
  - `local_path`
  - `model_deployments_fpath`
  - `enable_huggingface_models`
  - `enable_local_huggingface_models`
- Behavior:
  - before computing a run, the wrapper now prepares the HELM local config directory
  - if `model_deployments_fpath` is given, it copies that file to `<local_path>/model_deployments.yaml`
  - `helm-run` is now invoked with explicit `--local-path`
  - optional HELM Hugging Face registration flags are passed through when specified
- Updated `magnet/backends/helm/pipeline.py` so these parameters are available from kwdagger.
- Updated `materialize_helm_run.py` to defensively normalize optional values that may be rendered by `kwdagger` as CLI placeholders.
  - Real-world examples observed in generated commands:
    - `--precomputed_root=None`
    - `--model_deployments_fpath=None`
    - bare `--enable_huggingface_models`
    - bare `--enable_local_huggingface_models`
  - The wrapper now treats null-like placeholders and empty optional list inputs as truly unset.
- For list-shaped params that are directly exposed to `kwdagger`, prefer a single YAML-encoded string value instead of `nargs='*'`.
  - This fits kwdagger's key/value param model better.
  - The HELM wrapper now decodes those values with `kwutil.Yaml.coerce` before calling `helm-run`.

## Caveat About HELM's Experimental HuggingFace Registration

- HELM's `--enable-huggingface-models` is not a universal solution.
- Observed limitations from quick checks:
  - `meta/llama-3-8b-chat`
    - failed because the repo id is not directly usable as a public HF model id in this environment
    - likely also requires Hugging Face auth / correct repo id for gated models
  - `eleutherai/pythia-6.9b`
    - failed because HELM could not infer `model_max_length`
  - `aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct`
    - failed for the same `model_max_length` inference reason
  - `lmsys/vicuna-7b-v1.3`
    - failed during tokenizer conversion in this environment
- Conclusion:
  - custom `model_deployments.yaml` overrides are the more robust path for reproduction work
  - pass-through support for HELM's flags is still useful for models where they happen to work

## Likely Reproduction Strategy

- Prefer models that already have built-in Hugging Face deployments when possible.
- For Together-only model families that are still locally runnable:
  - create a custom HELM `model_deployments.yaml`
  - point kwdagger jobs at it via `helm.model_deployments_fpath`
  - keep the logical `adapter_spec.model` stable where possible, and change deployment resolution rather than renaming the model everywhere

## Next Checks

- Inspect a concrete historic run spec from `/data/crfm-helm-public`
- Inspect HELM registry entries for the corresponding model/deployment
- Determine whether a Hugging Face deployment already exists for those model names
- If not, decide whether to extend `materialize_helm_run.py` and/or add custom model deployment registration support

## kwdagger Observations

- The smoke-test audit manifest scheduled as expected:
  - 6 total jobs
  - 2 tmux workers
  - `CUDA_VISIBLE_DEVICES` split across GPU 0 and GPU 1
- For the HELM audit node, overriding the node `command` renderer is the cleanest way to suppress unset optional args.
  - This avoids generated commands like:
    - `--precomputed_root=None`
    - `--model_deployments_fpath=None`
    - bare `--enable_huggingface_models`
- `kwdagger` / `cmd_queue` may stop for an interactive prompt if older tmux queue sessions with the same queue name are still running.
  - That behavior matters for unattended reproduction runs and should be accounted for in future script refinements.
- The smoke-test comparison is now working end to end for the 6-run control batch.
  - Historic rows found: 6
  - kwdagger rows found: 6
  - current diagnosis label across all six cases: `multiple_primary_reasons`
  - primary reason counts:
    - `deployment_drift`: 6
    - `execution_spec_drift`: 6
  - full reason counts:
    - `completion_content_drift`: 6
    - `core_metric_drift`: 6
    - `dataset_instance_drift`: 6
    - `dataset_variant_drift`: 6
    - `deployment_drift`: 6
    - `evaluation_spec_drift`: 6
    - `execution_spec_drift`: 6
    - `request_prompt_drift`: 1
  - Interpretation:
    - this smoke batch validates the audit workflow plumbing
    - it does not yet validate close behavioral reproduction of the historic runs
    - the dominant differences are consistent with local HF execution diverging materially from the historic HELM deployment / execution configuration
- Representative case inspected:
  - `mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model=eleutherai/pythia-6.9b,data_augmentation=canonical`
  - historic run selected by matcher:
    - `/data/crfm-helm-public/classic/benchmark_output/runs/v0.2.4/...`
  - primary drift:
    - `deployment_drift`
      - historic `adapter_spec.model_deployment`: `null`
      - reproduced `adapter_spec.model_deployment`: `huggingface/pythia-6.9b`
    - `execution_spec_drift`
      - historic `adapter_spec.max_eval_instances`: `1000`
      - reproduced `adapter_spec.max_eval_instances`: `100`
  - additional evidence that this is not a close reproduction yet:
    - dataset base coverage differs: historic 111 vs reproduced 100
    - variant coverage differs: historic 327 vs reproduced 294
    - metric spec set differs substantially:
      - historic uses `BasicMetric`
      - reproduced uses `BasicGenerationMetric`, `BasicReferenceMetric`, and `InstancesPerSplitMetric`
    - core metric agreement is effectively zero in this case
      - `core_agree_ratio: 0.0`
    - completion agreement is only partial
      - 294 comparable completions
      - 28 mismatches
      - equal ratio about `0.905`
- Reporting improvements added after inspecting the smoke batch:
  - `compare_batch.py` now writes a management-facing text summary in addition to the technical summary and JSON outputs
  - summaries now expose:
    - `primary_reason_name_counts`
    - `reason_counts`
    - high-level findings suitable for executive reporting
  - historic candidate selection in the compare step is now more explicit and records whether an exact historic `adapter_spec.max_eval_instances` match exists
- Current important finding:
  - for the smoke manifest, the public HELM bundle does not appear to contain exact historic matches with requested `max_eval_instances=100`
  - available matched public runs for the inspected `pythia-6.9b` MMLU case record `adapter_spec.max_eval_instances=1000`
  - this means the current smoke comparison is useful for auditing drift, but it is not yet an apples-to-apples reproduction check on evaluation size
- Apples-to-apples smoke batch (`audit-smoke-apples`) result:
  - transferred report bundle shows:
    - historic rows found: 6
    - kwdagger rows found: 6
    - all 6 compared successfully
  - all 6 cases report `historic_exact_requested_max_eval_match = True`
  - the actual run-spec execution-path drift in the transferred case JSONL is now only:
    - `adapter_spec.model_deployment`
  - primary reasons across all 6 apples-to-apples cases:
    - `deployment_drift`
    - `execution_spec_drift`
  - recurring secondary reasons across all 6 apples-to-apples cases:
    - `evaluation_spec_drift`
    - `core_metric_drift`
    - `completion_content_drift`
  - one case also shows:
    - `request_prompt_drift`
  - Important correction:
    - the initial apples executive summary still claimed a requested `max_eval_instances` mismatch in all 6 cases
    - inspection of the case JSONL indicates that was a reporting bug from how the reproduced-side value was extracted, not the real comparison outcome
    - `compare_batch.py` was updated to read reproduced `run_spec.json` directly for `adapter_spec.max_eval_instances`

## Apples-To-Apples Report

Date: 2026-03-25
Experiment: `audit-smoke-apples`

### Executive Summary

- The apples-to-apples smoke batch compared 6 reproduced runs against 6 matched historic public HELM runs.
- All 6 runs were paired and compared successfully.
- After aligning `max_eval_instances` to the historic public bundle (`1000`), the remaining dominant drift is not evaluation size.
- The main recurring technical differences are:
  - explicit Hugging Face `model_deployment` in reproduced runs vs `null` deployment in historic runs
  - evaluation-spec drift caused by metric class changes
  - completion-content drift
  - core-metric drift
- This means the audit workflow is functioning correctly, and the remaining mismatch is now focused on how HELM resolves and evaluates these runs, not on the basic scheduling / pairing machinery.

### High-Level Conclusion

- The earlier non-apples smoke batch was confounded by comparing reproduced `max_eval_instances=100` runs against historic `max_eval_instances=1000` runs.
- The apples-to-apples batch removes that confounder.
- In the apples batch, the real remaining execution drift appears to be:
  - `adapter_spec.model_deployment`
- The real remaining evaluation drift appears to be:
  - `BasicMetric` in historic runs vs newer split metric classes in reproduced runs
- The current evidence suggests that at least part of the reproduction difference is due to HELM config / registry / metric behavior differences across environments or versions, not merely operator error in kwdagger scheduling.

### Apples Batch Artifacts

- Report bundle transferred locally to:
  - `/home/joncrall/code/aiq-magnet/audit-smoke-apples`
- Representative reproduced job directories transferred locally to:
  - `/home/joncrall/code/aiq-magnet/helm_id_10qfn238081w`
  - `/home/joncrall/code/aiq-magnet/helm_id_kydvdl7oawex`

### Apples Batch Technical Findings

- All 6 case rows in the transferred JSONL indicate:
  - `historic_exact_requested_max_eval_match = True`
- The union of reported run-spec execution-path drift across the transferred apples case rows is:
  - `adapter_spec.model_deployment`
- Recurring primary reasons in the transferred apples report:
  - `deployment_drift`
  - `execution_spec_drift`
- Recurring secondary reasons in the transferred apples report:
  - `evaluation_spec_drift`
  - `core_metric_drift`
  - `completion_content_drift`
- One case additionally reported:
  - `request_prompt_drift`

### Important Correction To Earlier Reporting

- The first apples executive summary incorrectly said all 6 cases had a requested `max_eval_instances` mismatch.
- Inspection of the transferred apples case JSONL and reproduced job directories shows:
  - historic `max_eval_instances = 1000`
  - reproduced `max_eval_instances = 1000`
- Therefore that headline was a reporting bug, not a real result.
- Root cause of the bug:
  - compare-side extraction of reproduced `adapter_spec.max_eval_instances` used the wrong view of the run spec
- Fix:
  - `compare_batch.py` now reads reproduced `run_spec.json` directly via `load_run_spec_json(...)`

### Representative Evidence: MMLU / Pythia-6.9b

Run entry:
- `mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model=eleutherai/pythia-6.9b,data_augmentation=canonical`

Historic matched run:
- `/data/crfm-helm-public/classic/benchmark_output/runs/v0.2.4/mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model=eleutherai_pythia-6.9b,data_augmentation=canonical`

Reproduced transferred job:
- `/home/joncrall/code/aiq-magnet/helm_id_10qfn238081w`

Observed config differences:
- historic `adapter_spec.model_deployment`: `null`
- reproduced `adapter_spec.model_deployment`: `huggingface/pythia-6.9b`
- historic `adapter_spec.max_eval_instances`: `1000`
- reproduced `adapter_spec.max_eval_instances`: `1000`

Observed metric class differences:
- historic:
  - `helm.benchmark.metrics.basic_metrics.BasicMetric`
- reproduced:
  - `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`
  - `helm.benchmark.metrics.basic_metrics.BasicReferenceMetric`
  - `helm.benchmark.metrics.basic_metrics.InstancesPerSplitMetric`

Observed behavior-level consequences from the transferred case JSONL:
- completion equal ratio about `0.896`
- core metric agree ratio about `0.5`

Interpretation:
- once eval size is aligned, this case still differs materially
- the remaining differences are consistent with deployment resolution and metric/evaluation config changes

### Representative Evidence: BoolQ / Vicuna-7b-v1.3

Run entry:
- `boolq:model=lmsys/vicuna-7b-v1.3,data_augmentation=canonical`

Historic matched run:
- `/data/crfm-helm-public/classic/benchmark_output/runs/v0.3.0/boolq:model=lmsys_vicuna-7b-v1.3,data_augmentation=canonical`

Reproduced transferred job:
- `/home/joncrall/code/aiq-magnet/helm_id_kydvdl7oawex`

Observed config differences:
- historic `adapter_spec.model_deployment`: `null`
- reproduced `adapter_spec.model_deployment`: `huggingface/vicuna-7b-v1.3`
- historic `adapter_spec.max_eval_instances`: `1000`
- reproduced `adapter_spec.max_eval_instances`: `1000`

Observed metric class differences:
- historic:
  - `helm.benchmark.metrics.basic_metrics.BasicMetric`
  - multiple `helm.benchmark.metrics.bias_metrics.BiasMetric`
- reproduced:
  - `helm.benchmark.metrics.basic_metrics.BasicGenerationMetric`
  - `helm.benchmark.metrics.basic_metrics.BasicReferenceMetric`
  - `helm.benchmark.metrics.basic_metrics.InstancesPerSplitMetric`
  - the same family of `helm.benchmark.metrics.bias_metrics.BiasMetric`

Observed behavior-level consequences from the transferred case JSONL:
- completion equal ratio about `0.083`
- core metric agree ratio `0.0`

Interpretation:
- this is a very strong mismatch even under apples-to-apples eval size
- the local reproduction is not merely “slightly off”; it is behaviorally very different for this case

### What We Can Say To Management

- The reproduction audit workflow is working end to end.
- We now have a real apples-to-apples control batch, not just a workflow smoke test.
- Even under apples-to-apples eval size, reproduced results still differ materially from historic public HELM runs.
- The dominant remaining differences are concentrated in deployment resolution and evaluation/metric configuration, with downstream output and metric drift.
- This is enough evidence to justify deeper technical follow-up with HELM maintainers.

### What We Can Say To HELM Maintainers

- The reproduced runs use explicit Hugging Face deployments where the matched historic public runs record `adapter_spec.model_deployment = null`.
- The reproduced runs also use a different metric-spec layout:
  - `BasicMetric` historically
  - `BasicGenerationMetric` + `BasicReferenceMetric` + `InstancesPerSplitMetric` in reproduced runs
- These config-level differences correlate with substantial completion-content and core-metric drift even after aligning requested eval size.
- The next maintainer-facing step should focus on why the same logical run entry resolves to these different `adapter_spec` / `metric_specs` structures across environments or HELM versions.

### Suggested Next Step

- Re-run the apples comparison after the `compare_batch.py` bugfix so the management summary no longer falsely reports requested eval-size mismatch.
- Then produce:
  - one management-facing memo from the executive summary
  - one maintainer-facing technical note with the two representative cases above and direct `run_spec.json` evidence

### Tooling Note: Pairwise Comparison Artifact Validation

- On 2026-03-27, `resolve_run.py`, `compare_entry.sh`, and `compare_pair.py` were hardened to validate finalized HELM run artifacts before diffing.
- A kwdagger job can exist with:
  - `job_config.json`
  - local caches
  - scenario assets
  while still lacking the finalized run outputs needed for diffing.
- Required files checked now:
  - `run_spec.json`
  - `scenario_state.json`
  - `stats.json`
  - `per_instance_stats.json`
- When those are missing, the tooling now reports a structured status such as:
  - `artifact_status: incomplete_run_dir`
  - `missing_files: [...]`
  instead of surfacing a `FileNotFoundError` from deep inside `HelmRunDiff`.

## Pairwise BoolQ / Pythia Result Snapshot

Compared:

- official HELM public run:
  - `/data/crfm-helm-public/classic/benchmark_output/runs/v0.3.0/boolq:model=eleutherai_pythia-6.9b,data_augmentation=canonical`
- local kwdagger repeats:
  - `/data/crfm-helm-audit/audit-boolq-pythia-r1/helm/helm_id_13jkx9mm4k4n/benchmark_output/runs/audit-boolq-pythia-r1/boolq:model=eleutherai_pythia-6.9b,data_augmentation=canonical`
  - `/data/crfm-helm-audit/audit-boolq-pythia-r2/helm/helm_id_12jr5w48kge7/benchmark_output/runs/audit-boolq-pythia-r2/boolq:model=eleutherai_pythia-6.9b,data_augmentation=canonical`

Repeatability baseline (`r1` vs `r2`):

- diagnosis: `bookkeeping_metric_drift`
- strict run-level agree ratio: `0.9552238805970149`
- strict instance-level agree ratio: `0.9523809523809523`
- run-level max abs delta: `0.0015118565559387176`
- instance-level max abs delta: `0.4418189525604248`
- interpretation:
  - repeated local runs are very close
  - residual drift is mostly bookkeeping/runtime noise

Historic vs local (`v0.3.0` vs `r1`):

- diagnosis: `multiple_primary_reasons`
- primary reason names:
  - `deployment_drift`
  - `execution_spec_drift`
- strict run-level agree ratio: `0.4626865671641791`
- strict instance-level agree ratio: `0.6577333333333333`
- run-level abs p90: `4.0`
- run-level abs max: `11.985`
- instance-level abs p90: `4.0`
- instance-level abs max: `75.73884344100952`
- interpretation:
  - historic vs local drift is much larger than local repeatability drift
  - this strongly suggests the remaining mismatch is structural/configurational rather than ordinary nondeterminism
