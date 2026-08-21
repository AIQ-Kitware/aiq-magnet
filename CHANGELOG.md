# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

## Unreleased

### Added

* Cards declare `kwdagger.result_node`: the node whose output is the card's
  result. Every configured instance of it is one cell, identified by that
  instance's kwdagger `process_id` and evaluated separately.
* A result node's values reach a claim as `metrics.<node>.<name>`. A card that
  declares a symbol of the same name still gets it unqualified, which is how a
  `define_metric` symbol is supplied.
* A verdict records the `cell` it belongs to and the results it `consumed`.
* A card reports the cells its run computed. A result node instance that
  produced nothing is skipped rather than failing the card, and recorded in
  `incomplete_cells.json` as `failed` (with its exit code), `pending`, or
  `empty`.
* `--params` merges a YAML/JSON blob (or a file of one) into a card's
  backend block, in the same language as `kwdagger schedule --params`. A
  card's matrix is a default grid an evaluator overrides, so running a card
  against models it does not name no longer means forking the card. The
  merged card is written to the run directory.
* The queue backend is selectable via `--queue_backend` and defaults to tmux.

### Deprecated

* A card's `pipeline:` block. Prefer `kwdagger:` with a `result_node`. Its
  semantics are unchanged and still supported; it now warns.

### Removed

* A `kwdagger` card must declare `result_node`; the schema requires it and
  a card without one is rejected as it loads.
  The path that rediscovered verdicts by globbing the run tree is gone, along
  with the node a pipeline had to carry to write them.

### Changed

* Requires `kwdagger>=0.4.0`.
* A cell's identity no longer depends on the values it measured, so a metric
  that moves replaces its verdict instead of writing a second one beside it.
* Node artifacts live in `<output>/_kwdagger`, shared across card versions, so
  editing a card no longer recomputes the nodes it did not change. Both routes
  ask each configured instance for its own artifact rather than globbing that
  root. `<run>/kwdagger` links there for consumers that read a run.
* An unchanged card reuses its run directory instead of stamping a new one.
* A relative pipeline path in a card resolves against the card.
* The tmux queue is named after the run directory rather than `schedule-eval`.

### Fixed

* `--override` accepts list and quoted values; both raised `RepresenterError`
  when the card was written back out.
* Accept `depends` as an alias for `depends_on` in symbol dependencies.
* Warn on unrecognized symbol-spec keys.

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
