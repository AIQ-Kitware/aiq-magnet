# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

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
