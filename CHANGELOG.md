# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

## Version 0.1.0 - Unreleased


### Added

* Added `magnet evaluate_new` for KWDagger-native evaluation recipes, including matrix overrides, dry runs, configurable evidence scope, and run provenance.
* Added schema validation for evaluation cards and recipes.
* Added logging with `--verbose` flag
* Added model, dataset, and metric metadata, including metric aggregation and optimization objectives.
* Added links from empirical evaluations to theoretical statements and named premises, with theory annotations also available through the standalone `aiq-magnet-theory` package.
* Added per-node container execution and optional infer-stack endpoint leasing for KWDagger pipelines.
* Expanded HELM materialization to support replay from resolved `run_spec.json` files, local model-deployment substitution, registry metadata passthrough, and materialization from local or public results.

### Deprecated

* Deprecated the legacy `pipeline:` card block. New pipeline-backed evaluations should use `kwdagger:` with `magnet evaluate_new`.

### Changed

* Python 3.11 is now the minimum supported Python version.
* KWDagger 0.4.1 or newer is now required.
* HELM integration and infer-stack leasing are now optional dependencies. Install `aiq-magnet[helm]`, `aiq-magnet[leasing]`, or `aiq-magnet[optional]` when those features are needed.
* `magnet evaluate` remains the legacy evaluator. KWDagger-native recipes should use `magnet evaluate_new`.
* Evaluation cards are now validated by default and require `version`, `organizations`, `submitter`, `tags`, and `links` in addition to the existing core card fields.
* Predictors now warn and use the available evaluation instances when fewer instances are available than requested. This behavior can be configured to error, warn, or ignore.
* The Llama KWDagger example has moved to `magnet/examples/llama_consistency/`.

### Removed

* HELM run-diff and Sankey analysis utilities have moved from `aiq-magnet` to `helm_audit`.

### Fixed

* Fixed metric resolution during parallel evaluation.
* Fixed metric metadata and aggregation handling for pipeline-backed evaluations.
* Improved HELM run matching across run-name and model-deployment variations.
* Fixed `--override` handling for quoted and list-valued values.
* Improved static typing.


## Version 0.0.2 -- Released 2026-05-08

### Added

* Added per-instance predictor base class (`InstancePredictor`) and random example
* User can now specify patterns to helm runs, suites, or all outputs as predictor input
* Added symbol sweeping capability to evaluation card evaluator
* Added modal CLI for `evaluation.py` script
* Added support for KWDagger pipelines in evaluation cards (both as explicit pipelines, and YAML defined pipelines)
* Added support for symbol overrides to `magnet evaluate` with the `--override` argument
* Added parallelization to `magnet evaluate` with the `--jobs` (and `--parallel_backend`) arguments
* Added claim resolution and final result file output to `magnet evaluate`
* Added support for `claim_aggregation_strategy` to evaluation cards (supporting `any`, `all`, and `fraction` strategies)

### Changed

* Switched to single argument path input for example predictors
* Cleaned up predicted vs. actual code for predictors
* HelmRuns.coerce can now accept a more expressive set of inputs
* BREAKING: You must how specify `helm_runs` when calling the predictor.
* `magnet download helm` can now download multiple benchmarks

### Fixed

* Fixed doctests and README wrt predictor refactors
* Updated `predict_inputs_exploration.ipynb` notebook wrt API updates

## Version 0.0.1 -- Released 2025-10-28

* Initial release; includes minimum working implementations for:
  * Evaluation card specification and evaluation
  * [HELM](https://github.com/stanford-crfm/helm) benchmark output downloading and data interfaces
  * Benchmark `Predictor` class (with random, and perturbation based examples)
  * Utility for "offline" HELM perturbation application
  * Ad-hoc inference and direct model access through HELM
  * Command-line wrapper for `helm-run` supporting runs against "offline" dataset instances
