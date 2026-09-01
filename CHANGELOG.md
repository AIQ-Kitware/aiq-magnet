# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

## Unreleased


### Added

* Added `magnet evaluate_new`, a kwdagger-only migration path that forwards selected `kwdagger schedule` controls directly: `--params`, `--backend`, `--tmux_workers`, `--skip_existing`, `--cache`, and `--max_configs`. It rejects legacy `pipeline:` computation and symbol sweeps while still feeding result-node values into the existing claim/verdict tail.
* Added `magnet evaluate_legacy` as the explicit name for the historical evaluator; `magnet evaluate` remains its compatibility alias.
* New evaluation recipes declare `kwdagger.result_node`: the node whose
  accumulated KWDagger aggregate rows provide claim evidence. `evaluate_new`
  requires it; the shared legacy schema leaves it optional for compatibility
  with card parsing. `evaluate` / `evaluate_legacy` reject kwdagger execution
  with a pointer to `evaluate_new`.
* KWDagger aggregate rows reach claims with their native qualified namespace,
  including `metrics.<node>.*`, `params.<node>.*`,
  `resolved_params.<node>.*`, and available lineage/context fields. A card
  symbol of the same leaf name can still be filled unqualified for the
  transitional `define_metric` behavior.
* `evaluate_new` resolves a recipe's `theory:` block and writes `theory.json`
  into the run directory, as `evaluate_legacy` already did. Links resolve
  before anything is scheduled, so a broken annotation or index fails before
  jobs run. The theory-link examples are now kwdagger recipes on this path.
* A short name -- a node-qualified or bare symbol, or a claim's node view --
  ranges only over `metrics`, `params` and `resolved_params`. `machine`,
  `resources`, `context` and the always-1 `specified` flags describe the run
  rather than the result and stay reachable only by qualified name; this keeps
  `machine.<node>.error`, which appears only when kwdagger's CPU probe fails,
  from colliding with an error a node measured.
* A Python claim can address an aggregate column by node alone --
  `llama_compare.gap` -- as well as by its qualified name --
  `metrics.llama_compare.gap`. The short form resolves wherever the node
  reports that name once; agreeing columns across namespaces collapse to their
  shared value, disagreeing ones raise and name the alternatives. Either
  spelling is recorded as the qualified column. A declared symbol of the same
  name as a node keeps precedence, so existing cards are unaffected.
* A declared symbol can name its evidence column by node --
  `llama_compare.base_score` -- the same spelling a claim uses. Short names
  match on segment boundaries, so the bare names legacy cards carry keep
  working and no name reaches across a partial segment. Where a short name
  matches columns that disagree, filling still warns and proceeds, since a
  symbol labels evidence rather than deciding a verdict.
* The claim namespace object is introspectable: `dir()`, `keys()`, `items()`,
  `values()`, `len()`, `in`, `[...]` indexing, and a repr that names its
  location and children. Inspecting a view does not mark evidence as consumed.
* A per-evidence claim record stores the source artifact, stable computation
  cell, and qualified fields the claim consumed.
* `requested_runs.json` records the current scheduling request separately from
  evidence: new-submission/skipped/disabled state, attempt status, return
  code, expected output, and whether that output is available. Failed or pending
  execution does not itself falsify a claim.
* New recipes can set `evidence.scope` to `all` (default) or `requested`.
  `requested` still discovers results through KWDagger aggregate, then keeps
  only available result-node computations requested by the current invocation;
  cached/skipped requested outputs are included.
* `evaluate_new` run bundles retain the visualization dashboard's existing
  `card.yaml` / `log` / `results/*/verdict.json` / `verdict.json` contract.
  The legacy `symbols` field in each per-evidence verdict is populated from
  resolved recipe symbols plus the qualified KWDagger leaves consumed by the
  claim, while the complete aggregate row remains available under `evidence`.
  An empty `results/` directory is retained for zero-evidence snapshots.
* `magnet evaluate_new --params` merges a YAML/JSON blob (or a file of one)
  into a recipe's `kwdagger:` block, in the same language as `kwdagger
  schedule --params`. The merged recipe is written to the run directory.

### Deprecated

* A card's `pipeline:` block. Prefer `kwdagger:` with a `result_node`. Its
  semantics are unchanged and still supported; it now warns.

### Changed

* Requires `kwdagger>=0.4.0`.
* The replacement Python API now uses `NewEvaluationRecipe` for input,
  `NewEvaluationCellResult` for a claim evaluated against one available
  KWDagger aggregate row, and `NewEvaluationResultCard` for the aggregate
  output. `NewEvaluationTask` is removed; claim evaluation is a direct
  transformation from a recipe and available evidence into a cell result.
* A cell's identity no longer depends on the values it measured, so a metric
  that moves replaces its verdict instead of writing a second one beside it.
* Under `evaluate_new`, node artifacts live in `<output>/_kwdagger`, shared
  across card versions, so editing a card does not recompute unchanged nodes.
  `<run>/kwdagger` links there for consumers that read a run. The legacy
  evaluator keeps its historical per-run DAG layout.
* Each `evaluate_new` invocation gets a distinct MAGNET run directory so its
  requested-work snapshot is preserved. KWDagger computation artifacts remain
  shared and reusable under `<output>/_kwdagger`.
* `evaluate_new` discovers evidence with KWDagger's aggregate loader after
  scheduling. The finite campaign requested by one invocation therefore does
  not bound the result rows available to the claim; prior successful campaigns
  remain visible in the shared result store.
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
