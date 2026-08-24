# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

## Unreleased


### Added

* Added `magnet evaluate_new`, a kwdagger-only migration path that forwards selected `kwdagger schedule` controls directly: `--params`, `--backend`, `--tmux_workers`, `--skip_existing`, `--cache`, and `--max_configs`. It rejects legacy `pipeline:` computation and symbol sweeps while still feeding result-node values into the existing claim/verdict tail.
* Added `magnet evaluate_legacy` as the explicit name for the historical evaluator; `magnet evaluate` remains its compatibility alias.
* New evaluation recipes declare `kwdagger.result_node`: the node whose output
  supplies the recipe's result cells. `evaluate_new` requires it; the shared
  legacy schema leaves it
  optional for compatibility with card parsing. `evaluate` /
  `evaluate_legacy` reject kwdagger execution with a pointer to `evaluate_new`.
  Every configured instance of the result node is one cell, identified by its
  kwdagger `process_id` and evaluated separately.
* A result node's values reach a claim as `metrics.<node>.<name>`. A card that
  declares a symbol of the same name still gets it unqualified, which is how a
  `define_metric` symbol is supplied.
* A verdict records the `cell` it belongs to and the results it `consumed`.
* A new evaluation result reports the cells its run computed. A result node
  instance that produced nothing is skipped rather than failing the recipe,
  and recorded in
  `incomplete_cells.json` as `failed` (with its exit code), `pending`, or
  `empty`.
* `magnet evaluate_new --params` merges a YAML/JSON blob (or a file of one)
  into a recipe's `kwdagger:` block, in the same language as `kwdagger
  schedule --params`. The merged recipe is written to the run directory.

### Deprecated

* A card's `pipeline:` block. Prefer `kwdagger:` with a `result_node`. Its
  semantics are unchanged and still supported; it now warns.

### Changed

* Requires `kwdagger>=0.4.0`.
* The replacement Python API now uses `NewEvaluationRecipe` for input,
  `NewEvaluationCellResult` for one kwdagger result-node cell, and
  `NewEvaluationResultCard` for the aggregate output. `NewEvaluationTask` is
  removed; per-cell claim evaluation is a direct transformation from a recipe
  and kwdagger result values into a cell result.
* A cell's identity no longer depends on the values it measured, so a metric
  that moves replaces its verdict instead of writing a second one beside it.
* Under `evaluate_new`, node artifacts live in `<output>/_kwdagger`, shared
  across card versions, so editing a card does not recompute unchanged nodes.
  `<run>/kwdagger` links there for consumers that read a run. The legacy
  evaluator keeps its historical per-run DAG layout.
* Under `evaluate_new`, an unchanged card reuses its run directory instead of
  stamping a new one.
* `evaluate_new` resolves a relative kwdagger pipeline path against the card.
* `evaluate_new` passes scheduling options directly to KWDagger; MAGNET no longer resolves queue backends, synthesizes queue names, or translates worker settings in `_kwdagger.py`.
* The Llama kwdagger example embeds its declarative `nodes` / `edges`
  pipeline directly in the card; the separate Python pipeline definition is
  removed.

### Fixed

* The inline Llama kwdagger card addresses result fields through `metrics.llama_compare`, matching its declared `result_node`.
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
