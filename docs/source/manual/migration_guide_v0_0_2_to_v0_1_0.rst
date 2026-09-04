AIQ-MAGNET Evaluation-Card Migration Guide
===========================================

:Baseline release: ``v0.0.2`` (commit ``9141dc1146ef9b55045527dae95bf41fed837996``,
                   2026-05-08)
:Target revision: ``v0.1.0``
:Audience: Authors and maintainers of MAGNET evaluation cards
:Primary focus: Legacy Python-symbol cards, symbol metadata, and migration to
                KWDagger recipes run with ``magnet evaluate_new``

Purpose and scope
-----------------

This guide describes the evaluation-card changes made after ``v0.0.2`` and
through the target ``v0.1.0`` version. It is based on the merged pull
requests in that interval, with particular attention to `PR 78`_, and on the
resulting schema, evaluator implementations, and Llama consistency example at
the pinned target revision.

The guide distinguishes three card shapes:

``Python-symbol card``
   A legacy card in which ``symbols`` contain ``value``, ``sweep``, and/or
   ``python`` definitions. MAGNET resolves those symbols and evaluates the
   Python claim.

``Legacy pipeline card``
   A legacy card with a *top-level* ``pipeline:`` block. This form remains
   executable by the legacy evaluator, but is soft-deprecated.

``KWDagger Recipe Card``
   A card with a *top-level* ``kwdagger:`` block that contains
   ``kwdagger.pipeline``, ``kwdagger.result_node``, and usually
   ``kwdagger.matrix``. This is the format accepted by ``evaluate_new``.


Executive summary
-----------------

Existing Python-symbol cards do not have to be rewritten immediately. Continue
to run them with ``magnet evaluate_legacy``; ``magnet evaluate`` remains a
compatibility alias. The main author-visible changes are schema validation,
optional symbol metadata, aggregate metric definitions, a ``depends`` alias,
and corrected aggregate metrics when legacy execution is parallelized.

Symbol metadata is additive. It is not required to resolve or execute a legacy
symbol. Add it when the dashboard should identify, label, or display a symbol,
or when MAGNET should calculate an aggregate metric across evaluation rows.
The ``kind`` field added by `PR 93`_ accepts ``model``, ``dataset``, or
``metric`` and is emitted in ``symbol_metadata.json`` for dashboard consumers.
It does not change symbol execution.

KWDagger migration is an explicit change of evaluator and execution model.
``magnet evaluate_new`` requires a ``kwdagger:`` recipe and a
``kwdagger.result_node``; it rejects both the legacy top-level ``pipeline:``
executor and legacy symbol sweeps. Experimental variation belongs in
``kwdagger.matrix``. KWDagger owns execution and accumulated results, while
MAGNET selects evidence rows, evaluates the existing Python claim once per row,
and writes the result-card/dashboard bundle.

Choose a migration path
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 23 22 24 31

   * - Current card shape
     - Command
     - Support status
     - Recommended action
   * - ``symbols`` with Python/value/sweep definitions
     - ``magnet evaluate_legacy`` or ``magnet evaluate``
     - Supported
     - Validate the card, add useful metadata, and migrate only when KWDagger's
       execution/evidence model is needed.
   * - Top-level legacy ``pipeline:``
     - ``magnet evaluate_legacy`` or ``magnet evaluate``
     - Supported, soft-deprecated
     - Add metadata-only declarations for pipeline outputs as needed. Plan a
       move to nested ``kwdagger.pipeline``.
   * - Top-level ``kwdagger:``
     - ``magnet evaluate_new``
     - New migration path
     - Declare ``result_node``, move all experimental axes to ``matrix``, and
       make node inputs/outputs explicit.
   * - ``kwdagger:`` passed to ``evaluate`` / ``evaluate_legacy``
     - Not applicable
     - Rejected by the current CLI
     - Use ``magnet evaluate_new``.
   * - Legacy ``pipeline:`` or symbol ``sweep`` passed to ``evaluate_new``
     - Not applicable
     - Rejected
     - Migrate the executor and move variation into ``kwdagger.matrix``, or
       remain on the legacy evaluator.

Changes that can require edits to an old card
---------------------------------------------

Schema validation is now on by default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both evaluators accept the validation modes ``only``, ``error``, ``warning``,
and ``off``. The default is ``error``. Use validation-only mode before running
expensive work:

.. code-block:: console

   magnet evaluate_legacy path/to/card.yaml --validate=only
   magnet evaluate_new path/to/recipe.yaml --validate=only

At the target revision, these top-level fields are required by the shared card
schema:

* ``title``
* ``description``
* ``claim`` with ``claim.python``
* ``version``
* ``organizations``
* ``submitter`` with ``name`` and ``email``
* ``tags``
* ``links``

``category``, ``claim_aggregation_strategy``, ``symbols``, and ``theory`` are
optional in the schema, subject to backend-specific constraints. A card with
neither ``pipeline`` nor ``kwdagger`` must provide ``symbols``.

Most checked-in ``v0.0.2`` examples already contain the required descriptive
fields. Private cards that depended on permissive parsing may now need those
fields added. ``--validate=warning`` can be useful during a staged cleanup, but
``--validate=error`` should be the acceptance criterion.

The evaluator command now matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current CLI exposes two paths:

.. code-block:: console

   magnet evaluate_legacy card.yaml   # explicit historical evaluator
   magnet evaluate card.yaml          # compatibility alias for the historical evaluator
   magnet evaluate_new recipe.yaml    # KWDagger-only path

This intentionally separates the new execution/evidence semantics path while it is still experimental. In `0.1.0` the `evaluate` command resolves to `evaluate_legacy` to maintain backwards compatibility during the migration. A future version of MAGNET will remove `evaluate_new` and `evaluate_legacy` and `evaluate` will be the canonical entrypoint for kwdagger-style recipe cards.

Top-level ``pipeline:`` is deprecated, not nested ``kwdagger.pipeline``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are different constructs:

.. code-block:: yaml

   # Legacy form, run by evaluate/evaluate_legacy
   pipeline:
     score:
       executable: "python score.py"
       # ...

.. code-block:: yaml

   # KWDagger form, run by evaluate_new
   kwdagger:
     result_node: score
     pipeline:
       nodes:
         score:
           executable: "python score.py"
           # ...
       edges: []
     matrix:
       score.model: [model-a, model-b]

The legacy form still runs, but emits a deprecation warning. The KWDagger form
is the recommended API.

Migration path A: keep a Python-symbol card
-------------------------------------------

This is the lowest-risk path when the existing ``symbols`` graph and Python
claim already express the evaluation correctly.

Add symbol metadata where it carries meaning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each symbol can now contain an optional ``metadata`` mapping.

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Field
     - Allowed value
     - Meaning
   * - ``kind``
     - ``model``, ``dataset``, or ``metric``
     - Semantic classification written to ``symbol_metadata.json`` for the
       dashboard. It does not compute, coerce, or validate the symbol's value.
   * - ``display``
     - Boolean
     - Marks a symbol for display in the dashboard.
   * - ``display_name``
     - String
     - Human-readable label. For a defined aggregate metric, this is also the
       output key used in the aggregate metrics (in the top level ``verdict.json`` output).
   * - ``define_metric``
     - Mapping
     - Requests a numeric aggregate across all selected evaluation rows.
   * - ``define_metric.objective``
     - ``maximize`` or ``minimize``
     - Describes the desired optimization direction.
   * - ``define_metric.aggregation_strategy.type``
     - ``mean``, ``max``, or ``min``
     - Reducer applied across rows. Although ``custom`` is accepted by the
       schema, custom reducers are not currently supported (raises ``NotImplementedError``).

A practical metadata patch for a Python-symbol card looks like this:

.. code-block:: yaml

   symbols:
     base_model:
       metadata:
         kind: model
         display: true
       sweep:
         - meta/llama-2-13b
         - meta/llama-2-70b

     base_score:
       metadata:
         kind: metric
         display: true
         display_name: "Average Exact Match"
         define_metric:
           objective: maximize
           aggregation_strategy:
             type: mean
       type: float
       depends_on:
         - base_model
         - exact_match_scores
       python: |
         base_score = [
             score
             for name, score in exact_match_scores
             if name == base_model
         ][0]

     helm_runs_path:
       metadata:
         kind: dataset
         display: true
         display_name: "HELM Data Path"
       type: str
       value: "./data/crfm-helm-public/lite/benchmark_output"

Important metadata rules
~~~~~~~~~~~~~~~~~~~~~~~~

``kind: metric`` and ``define_metric`` are independent
   ``kind`` classifies the symbol for the dashboard. ``define_metric`` tells
   MAGNET to calculate a cross-row aggregate. Use both when both meanings are
   intended; do not assume either one implies the other.

A defined metric must exist in every aggregated row
   MAGNET gathers the value of the named symbol from every row. If the value is
   absent in any row, that metric is not calculated. In the legacy evaluator,
   the rows are resolved sweep combinations. In ``evaluate_new``, they are the
   selected KWDagger evidence rows.

Metric aggregation is across rows, not inside one run
   A ``mean`` under ``define_metric`` reduces the values from all evaluation
   rows/cells. If one KWDagger node must average multiple files, subjects, or
   samples *within* a cell, perform that reduction in the DAG and expose the
   resulting scalar as a node result.

Metadata is preserved separately
   When metadata is present, the evaluator writes ``symbol_metadata.json`` in
   the MAGNET run directory. This is the dashboard-facing metadata artifact.

Metadata-only declarations are valid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A symbol with only ``metadata`` is accepted. This is most useful with a legacy
pipeline or KWDagger recipe, where an executor already produced the value and
the card only needs to describe it:

.. code-block:: yaml

   symbols:
     base_score:
       metadata:
         kind: metric
         display: true
         display_name: "Average Exact Match"
         define_metric:
           objective: maximize
           aggregation_strategy:
             type: mean

Do not convert an ordinary Python-computed symbol to metadata-only form. Keep
its ``value``, ``sweep``, or ``python`` definition when MAGNET itself is still
responsible for resolving it.

Recommended legacy-card acceptance sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   # 1. Strict schema check, no evaluation
   magnet evaluate_legacy card.yaml --validate=only

   # 2. Small or overridden serial run
   magnet evaluate_legacy card.yaml \
       --override='base_model: [meta/llama-2-13b]' \
       --jobs=1 \
       --output_path=./migration-smoke

   # 3. Intended parallel mode
   magnet evaluate_legacy card.yaml \
       --jobs=4 \
       --parallel_backend=loky \
       --output_path=./migration-full

Confirm that the run contains ``card.yaml``, per-row ``results/*/verdict.json``,
the overall ``verdict.json``, and ``symbol_metadata.json`` when metadata was
declared. Confirm that each expected metric appears in the overall result.

Migration path B: convert to KWDagger and ``evaluate_new``
---------------------------------------------------------

The central design change from `PR 78`_ is that KWDagger is treated as an
experimental execution and result engine. Scheduling,
accumulated result discovery, evidence selection, and claim evaluation are
separate stages.

The current flow is:

#. Compile and schedule the finite request described by ``kwdagger.pipeline``
   and ``kwdagger.matrix``.
#. Record the requested processes in ``requested_runs.json``.
#. Ask KWDagger aggregate for available rows from ``kwdagger.result_node`` in
   the shared result store.
#. Apply ``evidence.scope``.
#. Evaluate the existing Python claim once per selected evidence row.
#. Aggregate those cell verdicts and declared metrics into the MAGNET result.

A failed, queued, or not-yet-started process remains execution provenance. It
is not automatically interpreted as evidence that the claim is false.

Step 1: add a recipe identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A KWDagger recipe SHOULD declare ``name`` in addition to the human-readable
``title``:

.. code-block:: yaml

   name: llama_consistency
   title: "In-domain Model Consistency for Llama Family"

``name`` is optional and is derived from the filename when omitted. When
provided, it must contain only letters, digits, underscores, and hyphens:
``^[A-Za-z0-9_-]+$``. It is used in paths and queue/session names, so do not put
spaces, periods, or colons in it.

Step 2: keep only static, non-sweep values in ``symbols``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``evaluate_new`` can still resolve non-sweep recipe symbols, including
``value`` and ``python`` symbols, before evaluating a claim. It deliberately
rejects every symbol with ``sweep``.

Move experimental axes from this legacy form:

.. code-block:: yaml

   symbols:
     base_model:
       sweep: [model-a, model-b]
     comp_model:
       sweep: [model-a, model-b]

into KWDagger's matrix:

.. code-block:: yaml

   kwdagger:
     matrix:
       compare.base_model: [model-a, model-b]
       compare.comp_model: [model-a, model-b]

Constants that belong to node execution usually fit better under a node's
``algo_params``. Constants used only by the Python claim may remain ordinary
non-sweep symbols.

Map the old Python symbol graph to KWDagger concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pure Python-symbol card often hides an execution graph inside
``depends_on``/``python`` definitions. Make that graph explicit rather than
copying every old symbol into the new recipe.

.. list-table::
   :header-rows: 1
   :widths: 37 63

   * - Legacy Python-card construct
     - KWDagger migration
   * - Literal ``value`` consumed by a node
     - Put it in that node's ``algo_params``. Keep it as a static MAGNET symbol
       only when it belongs to the claim rather than node execution.
   * - ``sweep``
     - Move it to one or more ``kwdagger.matrix`` axes.
   * - ``python`` symbol that finds, downloads, or materializes data
     - Make it an executable materialization node with explicit output paths.
   * - ``python`` symbol that computes a measurement
     - Make it an executable scoring node whose artifact can be loaded into
       aggregate columns.
   * - ``depends_on`` between computed symbols
     - Represent artifact flow with KWDagger edges. Use matrix/configuration
       dependencies for parameter relationships rather than Python globals.
   * - ``python`` symbol that only parses a node's result artifact
     - Prefer a node ``load_result`` function. If it performs a real empirical
       transformation, use a separate downstream node instead.
   * - Symbol used only to label an executor-produced result
     - Keep a metadata-only symbol named after the aggregate column.
   * - Final Boolean assertion
     - Keep it in ``claim.python``, rewritten against the aggregate namespace.

A useful boundary test is reproducibility: if a symbol's value depends on the
filesystem, a model call, a dataset query, or another empirical artifact, it
usually belongs in a KWDagger node. A pure deterministic expression over static
recipe constants can remain a MAGNET Python symbol.

Step 3: replace the top-level legacy executor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``v0.0.2``-style pipeline card could look like this:

.. code-block:: yaml

   claim:
     python: |
       assert abs(comp_score - base_score) < threshold

   pipeline:
     llama_predict:
       executable: "python magnet/examples/llama_consistency/llama_predict.py"
       algo_params:
         helm_runs_path: "./data/crfm-helm-public/lite/benchmark_output"
         base_model: [model-a, model-b]
         comp_model: [model-a, model-b]
       out_paths:
         results_fpath: results.json

   symbols:
     threshold:
       value: 0.1

For a ``v0.1.0`` recipe card using ``evaluate_new``, place the execution graph under ``kwdagger.pipeline`` and
use KWDagger's declarative ``nodes`` and ``edges`` structure. The following is
a shortened, annotated form of the current Llama example:

.. code-block:: yaml

   name: llama_consistency
   title: "In-domain Model Consistency for Llama Family"
   description: |
     Performance should remain within a consistency bound.
   version: 1.0
   organizations: [Kitware]
   submitter:
     name: Kitware TA2 Team
     email: aiq-ta2@kitware.com
   tags: [example, llama, helm, mmlu]
   links:
     - title: MAGNET
       url: https://github.com/AIQ-Kitware/aiq-magnet
       type: software

   claim:
     python: |
       c = llama_compare
       assert c.gap < c.threshold, (
           f"{c.comp_model} score ({c.comp_score:.2f}) exceeds "
           f"the consistency bound on {c.base_model} ({c.base_score:.2f})"
       )

   evidence:
     scope: requested

   symbols:
     metrics.llama_compare.base_score:
       metadata:
         kind: metric
         display: true
         display_name: "Average Exact Match"
         define_metric:
           objective: maximize
           aggregation_strategy:
             type: mean

     metrics.llama_compare.comp_model:
       metadata:
         kind: model
         display: true
         display_name: "Comparison model"

     llama_predict.scored_runs:
       metadata:
         display: true
         display_name: "HELM runs scored"

   kwdagger:
     result_node: llama_compare

     pipeline:
       nodes:
         materialize_run:
           executable: >-
             python -m
             magnet.examples.llama_consistency.materialize_run
           algo_params:
             model: null
             subject: null
             precomputed_root: "./data/crfm-helm-public"
             mode: reuse_only
           out_paths:
             out_dpath: "."
             done_fname: DONE
           primary_out_key: done_fname
           load_result: >-
             magnet.examples.llama_consistency.materialize_run.load_kwdagger_result

         llama_predict:
           executable: >-
             python -m
             magnet.examples.llama_consistency.llama_predict
           in_paths: [run_dpaths]
           algo_params:
             base_model: null
             comp_model: null
             threshold: 0.1
           out_paths:
             results_fpath: results.json
           primary_out_key: results_fpath
           load_result: >-
             magnet.examples.llama_consistency.llama_predict.load_kwdagger_result

         llama_compare:
           executable: >-
             python -m
             magnet.examples.llama_consistency.llama_compare
           in_paths: [scores_fpath]
           out_paths:
             out_fpath: comparison.json
           primary_out_key: out_fpath
           load_result: >-
             magnet.examples.llama_consistency.llama_compare.load_kwdagger_result

       edges:
         - src: materialize_run.out_dpath
           dst: llama_predict.run_dpaths
           gather:
             group_by: []
             order_by: [model, subject]
             require: all_success
         - llama_predict.results_fpath -> llama_compare.scores_fpath

     matrix:
       materialize_run.model: [model-a, model-b]
       materialize_run.subject: [abstract_algebra, college_chemistry]
       llama_predict.base_model: [model-a, model-b]
       llama_predict.comp_model: [model-a, model-b]

This example uses fully qualified metadata keys where the namespace is part of
the intended meaning. The checked-in Llama recipe uses the shorter
``llama_compare.base_score`` form, which is also valid.

While this example uses the inline KWDagger specification, it's also
possible to reference an external pipeline definition as well.  See
`KWDagger Tutorials`_ for more details.

Step 4: make artifacts and dependencies explicit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each node, identify:

``executable``
   The command that performs one node computation.

``algo_params``
   Ordinary node parameters. A ``null`` value is commonly filled by a matrix
   axis or another configuration source.

``in_paths``
   Inputs supplied by upstream edges or on-disk dependencies. These are either supplied by the user directly (i.e. by specifying the values in the parameter matrix) or by connecting (drawing an edge) from an out_path to an in_path.

``out_paths``
   Named output artifacts. The default values for these should be relative paths for where they are stored in a kwdagger output node. 

``primary_out_key``
   The output KWDagger treats as the node's primary completion/result artifact. This is how kwdagger finds the filepath to pass to your load result function. 

``load_result``
   An optional import path used to load an artifact into aggregate columns.
   Use it when the generic result envelope is not sufficient. A materializer
   that produces an artifact but no measurement can provide a loader that
   explicitly returns no result rather than allowing a generic loader to parse
   a sentinel file.

Step 5: use gather edges for fan-in
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A gather edge turns many upstream artifacts into a collection-valued input for
one downstream cell:

.. code-block:: yaml

   - src: materialize_run.out_dpath
     dst: llama_predict.run_dpaths
     gather:
       group_by: []
       order_by: [model, subject]
       require: all_success

In the Llama example, ``group_by: []`` intentionally gives each comparison cell
the entire declared corpus; the cell then selects the two models it compares.
Grouping by model would be wrong because one comparison needs data for two
models. Choose grouping keys from the data dependency, not merely from the
matrix shape.

Step 6: choose ``result_node`` deliberately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``kwdagger.result_node`` is required by ``evaluate_new``. It names the node
whose available aggregate rows are candidate evidence for the claim:

.. code-block:: yaml

   kwdagger:
     result_node: llama_compare

Choose the last node that emits one claim-ready result per desired evaluation
cell. If the scoring node emits two raw scores but the claim concerns their gap,
add a comparison/reduction node and make that the result node. This keeps the
claim simple and makes the measurement represented by each evidence row clear.

Step 7: rewrite the claim against aggregate namespaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KWDagger aggregate rows expose qualified names such as:

* ``metrics.<node>.<field>``
* ``params.<node>.<field>``
* ``resolved_params.<node>.<field>``

Where ``<node>`` is the name of the KWDagger node
(e.g. ``llama_compare``), and ``<field>`` is the name of the field or
parameter (e.g. ``base_score``). A Python claim can use the fully
qualified path or a node view:

.. code-block:: yaml

   claim:
     python: |
       # Convenient node view
       assert llama_compare.gap < llama_compare.threshold

       # Fully qualified lookup is unambiguous
       assert (
           metrics.llama_compare.gap
           < metrics.llama_compare.threshold
       )

The fully qualified path is safest and always states which aggregate column the
claim consumed. A node-qualified short path, such as
``llama_compare.base_score``, is accepted when that field resolves uniquely or
matching namespaces agree. A disagreeing ambiguity in a claim is an error,
because a verdict must not depend on arbitrary resolution order.

Only result-like namespaces participate in short-name resolution. Run-context
namespaces such as machine/resource data remain available by qualified name and
do not compete with measured fields.

Step 8: convert result labels to metadata-only symbols
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a KWDagger recipe, a result already produced by a node should normally not be
recomputed in ``symbols``. Declare a metadata-only symbol whose key names the
evidence column:

.. code-block:: yaml

   symbols:
     metrics.llama_compare.base_score:
       metadata:
         kind: metric
         display: true
         display_name: "Average Exact Match"
         define_metric:
           objective: maximize
           aggregation_strategy:
             type: mean

     resolved_params.llama_predict.helm_runs_path:
       metadata:
         kind: dataset
         display: true
         display_name: "Resolved HELM data path"

A shorter node-qualified or bare key can be used. However, a metadata-only
symbol that matches several disagreeing columns currently warns and chooses a
candidate, whereas a claim raises. Use a fully qualified name whenever the
namespace distinguishes the intended value.

Static, non-sweep symbols can coexist with metadata-only evidence labels, but a
static symbol name may not collide with a pipeline result bound into the claim
context.

Step 9: set the evidence scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``evaluate_new`` separates the work requested by one invocation from the rows
already accumulated in the shared KWDagger store.

.. code-block:: yaml

   evidence:
     scope: requested

``requested``
   Evaluate only result-node computations requested by this invocation,
   including computations satisfied from existing cached output. This is the
   recommended default for a reproducible per-invocation result snapshot.

``all``
   Evaluate every compatible result-node row currently available in the shared
   store. Use this when successive partial campaigns are intentionally building
   a cumulative evidence set.

The schema default is ``all``. Therefore, write ``scope: requested`` explicitly
when an old artifact from a previous or superseded matrix must not continue to
vote in the current result. The CLI can override the recipe with
``--evidence_scope=requested`` or ``--evidence_scope=all``.

Step 10: validate, inspect, and run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   # Schema only
   magnet evaluate_new recipe.yaml --validate=only

   # Compile the matrix and inspect requested_runs.json; run nothing
   magnet evaluate_new recipe.yaml \
       --output_path=./results_kwdagger \
       --dry_run

   # Execute a single matrix configuration as a smoke test
   magnet evaluate_new recipe.yaml \
       --output_path=./results_kwdagger \
       --backend=serial \
       --max_configs=1

   # Run the intended local campaign
   magnet evaluate_new recipe.yaml \
       --output_path=./results_kwdagger \
       --backend=serial

A dry run compiles and records the campaign, but submits no work, loads no
evidence, evaluates no claim, returns ``NOT_EVALUATED``, and writes no
``verdict.json``. Do not use it to infer what an existing shared store currently
proves.

``evaluate_new`` defaults to the ``tmux`` backend. ``--tmux_workers`` controls
KWDagger workers; ``auto`` derives a bound from local GPUs when available. Use
``--backend=serial`` for a deterministic foreground smoke test.

Reuse and override semantics
----------------------------

Legacy and new overrides are not interchangeable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The legacy evaluator's ``--override`` replaces values or sweep lists in the
card's ``symbols`` mapping:

.. code-block:: console

   magnet evaluate_legacy card.yaml \
       --override='threshold: 0.05'

The new evaluator's ``--params`` deep-merges YAML/JSON into the recipe's
``kwdagger`` block, using KWDagger's matrix/configuration language:

.. code-block:: console

   magnet evaluate_new recipe.yaml \
       --params='matrix: {llama_predict.base_model: [model-a]}' \
       --backend=serial

Mappings merge recursively; a list or other non-mapping value replaces the
existing leaf.

Caching controls
~~~~~~~~~~~~~~~~

``--skip_existing=1`` avoids submitting nodes whose expected products already
exist. ``--cache=1`` (the default) lets submitted commands skip work when their
outputs already exist. To request recomputation using KWDagger's native
semantics, disable both:

.. code-block:: console

   magnet evaluate_new recipe.yaml \
       --backend=serial \
       --skip_existing=0 \
       --cache=0

These controls affect requested execution. They do not select evidence; that is
the job of ``evidence.scope``.

Result and dashboard behavior
-----------------------------

KWDagger computations accumulate under a shared ``<output_path>/_kwdagger``
store. Each MAGNET invocation receives a separate timestamped run directory and
a link to that store. The run directory preserves the existing dashboard
contract, including:

* ``card.yaml``
* ``log``
* ``results/*/verdict.json`` for evaluated evidence cells
* overall ``verdict.json`` for non-dry runs
* ``symbol_metadata.json`` when metadata is declared
* ``requested_runs.json`` for the invocation's operational request
* ``theory.json`` when theory links are declared

The per-cell dashboard ``symbols`` mapping contains static resolved symbols and
the qualified KWDagger fields actually consumed by the claim. The shared
``_kwdagger`` directory is not needed in a dashboard upload bundle.

``requested_runs.json`` and ``verdict.json`` answer different questions:

``requested_runs.json``
   What this invocation requested and what operational state those processes
   reached.

``verdict.json``
   What the selected, currently available evidence rows imply about the claim.

Llama consistency migration notes
---------------------------------

The checked-in KWDagger example is the most complete migration reference:
``magnet/examples/llama_consistency/llama_kwdagger.yaml`` and its ``README.md``.
Its graph is:

.. code-block:: text

   materialize_run -> llama_predict -> llama_compare

The three stages exist for concrete compatibility reasons:

* ``materialize_run`` creates one explicit artifact per ``(model, subject)``.
* ``llama_predict`` consumes a gathered manifest and calculates model scores.
  The executable is shared with the legacy pipeline card.
* ``llama_compare`` reduces those scores to the gap used by the claim and is
  therefore the ``result_node``.

The port intentionally changes data-selection semantics. The legacy Python card
globs a HELM cache and averages every matching MMLU subject it finds. The
KWDagger example declares two subjects in its matrix. This improves
reproducibility, but means the values are not numerically identical even though
the conceptual evaluation and 6-by-6 model-pair grid are retained.

The 6-by-6 grid has 36 cells but only 15 distinct nontrivial unordered model
comparisons: six self-comparisons pass by construction and 15 cells mirror the
other direction because the claim uses an absolute gap. This is inherited from
the legacy example, not a recommended new design. A new card could instead use
a node that reads the declared set once and emits only the comparisons the
claim actually needs.

Because the gather uses ``group_by: []``, run one model family per invocation
unless the DAG is extended with an explicit family grouping/tag. Otherwise,
multiple families would be pooled into every comparison's input collection.

Optional additions since ``v0.0.2``
-----------------------------------

Theory links
~~~~~~~~~~~~

Cards may include a structured ``theory:`` block connecting empirical code to
formal or prose theoretical objects. Both evaluators resolve those links before
expensive execution and write ``theory.json``. A broken source annotation or
index therefore fails early. This is optional and does not need to be added as
part of a symbol/KWDagger migration unless the project uses the feature.

Containers and inference-stack leasing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`PR 94`_ added optional execution controls around ``evaluate_new``, including
containerized node commands and inference-stack leasing. They do not change the
core recipe requirements described above. A card can migrate to KWDagger first
and adopt those controls separately when a stable node runtime or managed model
endpoints are needed.

Migration checklists
--------------------

Legacy Python-symbol card
~~~~~~~~~~~~~~~~~~~~~~~~~

.. checklist entries are deliberately written as ordinary bullets so this file
   renders in standard docutils/Sphinx installations without an extension.

* The card passes ``magnet evaluate_legacy CARD --validate=only``.
* All required descriptive fields are present.
* The card uses ``evaluate_legacy``/``evaluate``, not ``evaluate_new``.
* Symbols still computed by MAGNET retain ``value``, ``sweep``, or ``python``.
* Model, dataset, and metric symbols have ``metadata.kind`` where useful.
* Dashboard-facing symbols have intentional ``display`` and ``display_name``.
* Every ``define_metric`` has an explicit ``objective`` and one of
  ``mean``, ``min``, or ``max``.
* Each defined metric resolves in every sweep row.
* The output includes ``symbol_metadata.json`` when expected.

KWDagger recipe
~~~~~~~~~~~~~~~

* The recipe passes ``magnet evaluate_new RECIPE --validate=only``.
* It has a top-level ``kwdagger:`` block and no top-level legacy ``pipeline:``.
* ``kwdagger.result_node`` identifies a claim-ready result node.
* No symbol contains ``sweep``.
* Experimental axes are declared under ``kwdagger.matrix``.
* Nodes declare their executable, inputs, outputs, and completion/result
  artifact intentionally.
* Fan-in uses an explicit gather edge with correct grouping and ordering.
* Claims use node-qualified or fully qualified aggregate columns.
* Metadata-only symbols name the evidence columns they describe.
* Ambiguous metadata names are fully qualified.
* ``evidence.scope`` is explicit and matches the intended snapshot semantics.
* ``--dry_run`` reports the expected campaign size.
* ``--max_configs=1`` produces a valid one-cell smoke result.
* ``requested_runs.json`` and the selected evidence-cell count are reviewed
  separately.
* A repeated run exercises the intended cache/reuse behavior.

Source references
-----------------

* `v0.0.2 tag`_
* `v0.0.2 to target-main comparison`_
* `target main commit`_
* `current schema`_
* `legacy evaluator`_
* `new evaluator`_
* `current changelog`_
* `v0.0.2 Llama Python card`_
* `v0.0.2 Llama pipeline card`_
* `current Llama Python card`_
* `current Llama pipeline card`_
* `current Llama KWDagger recipe`_
* `current Llama KWDagger README`_
* `KWDagger Tutorials`_

.. _v0.0.2 tag: https://github.com/AIQ-Kitware/aiq-magnet/releases/tag/v0.0.2
.. _v0.0.2 to target-main comparison: https://github.com/AIQ-Kitware/aiq-magnet/compare/9141dc1146ef9b55045527dae95bf41fed837996...5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13
.. _target main commit: https://github.com/AIQ-Kitware/aiq-magnet/commit/5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13
.. _current schema: https://github.com/AIQ-Kitware/aiq-magnet/blob/5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13/magnet/schema.py
.. _legacy evaluator: https://github.com/AIQ-Kitware/aiq-magnet/blob/5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13/magnet/evaluation.py
.. _new evaluator: https://github.com/AIQ-Kitware/aiq-magnet/blob/5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13/magnet/evaluation_new.py
.. _current changelog: https://github.com/AIQ-Kitware/aiq-magnet/blob/5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13/CHANGELOG.md
.. _v0.0.2 Llama Python card: https://github.com/AIQ-Kitware/aiq-magnet/blob/9141dc1146ef9b55045527dae95bf41fed837996/magnet/cards/llama.yaml
.. _v0.0.2 Llama pipeline card: https://github.com/AIQ-Kitware/aiq-magnet/blob/9141dc1146ef9b55045527dae95bf41fed837996/magnet/cards/llama_pipeline.yaml
.. _current Llama Python card: https://github.com/AIQ-Kitware/aiq-magnet/blob/5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13/magnet/cards/llama.yaml
.. _current Llama pipeline card: https://github.com/AIQ-Kitware/aiq-magnet/blob/5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13/magnet/cards/llama_pipeline.yaml
.. _current Llama KWDagger recipe: https://github.com/AIQ-Kitware/aiq-magnet/blob/5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13/magnet/examples/llama_consistency/llama_kwdagger.yaml
.. _current Llama KWDagger README: https://github.com/AIQ-Kitware/aiq-magnet/blob/5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13/magnet/examples/llama_consistency/README.md

.. _PR 56: https://github.com/AIQ-Kitware/aiq-magnet/pull/56
.. _PR 58: https://github.com/AIQ-Kitware/aiq-magnet/pull/58
.. _PR 59: https://github.com/AIQ-Kitware/aiq-magnet/pull/59
.. _PR 60: https://github.com/AIQ-Kitware/aiq-magnet/pull/60
.. _PR 61: https://github.com/AIQ-Kitware/aiq-magnet/pull/61
.. _PR 62: https://github.com/AIQ-Kitware/aiq-magnet/pull/62
.. _PR 63: https://github.com/AIQ-Kitware/aiq-magnet/pull/63
.. _PR 64: https://github.com/AIQ-Kitware/aiq-magnet/pull/64
.. _PR 65: https://github.com/AIQ-Kitware/aiq-magnet/pull/65
.. _PR 67: https://github.com/AIQ-Kitware/aiq-magnet/pull/67
.. _PR 70: https://github.com/AIQ-Kitware/aiq-magnet/pull/70
.. _PR 71: https://github.com/AIQ-Kitware/aiq-magnet/pull/71
.. _PR 75: https://github.com/AIQ-Kitware/aiq-magnet/pull/75
.. _PR 77: https://github.com/AIQ-Kitware/aiq-magnet/pull/77
.. _PR 78: https://github.com/AIQ-Kitware/aiq-magnet/pull/78
.. _PR 79: https://github.com/AIQ-Kitware/aiq-magnet/pull/79
.. _PR 80: https://github.com/AIQ-Kitware/aiq-magnet/pull/80
.. _PR 90: https://github.com/AIQ-Kitware/aiq-magnet/pull/90
.. _PR 93: https://github.com/AIQ-Kitware/aiq-magnet/pull/93
.. _PR 94: https://github.com/AIQ-Kitware/aiq-magnet/pull/94

.. _KWDagger Tutorials: https://gitlab.kitware.com/computer-vision/kwdagger/-/tree/main/docs/source/manual/tutorials
