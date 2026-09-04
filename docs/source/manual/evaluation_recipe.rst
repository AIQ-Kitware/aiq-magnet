Evaluation Recipe Card Schema
=============================

An **Evaluation Recipe Card** is a YAML file declaring a falsifiable empirical
claim, the symbols the claim reads, and how the values behind those symbols are
produced. Evaluating one produces an **Evaluation Result Card**: a run bundle
holding a verdict, per-cell claim records, and the recipe that produced them.

This page enumerates every field a recipe card may contain.

Normative source
    ``magnet/schema.py``. ``EvaluationCardSchema`` is the base document;
    ``NewEvaluationRecipeSchema`` narrows it for ``magnet evaluate_new``. Class
    names still say *card* where this page says *recipe card*.

Validating a card
    ``--validate=only`` checks a card and exits. Under ``evaluate_legacy`` it
    also resolves the ``theory`` block; under ``evaluate_new`` it runs schema
    validation alone, and theory references are resolved at evaluation time
    (including under ``--dry_run``). Validation modes and the acceptance
    sequence are covered in :doc:`migration_guide_v0_0_2_to_v0_1_0`.


Execution forms
---------------

A card resolves its symbols in exactly one of three forms. The chosen form
decides which top-level keys are legal and which evaluator will accept the
card.

.. list-table::
   :header-rows: 1
   :widths: 8 26 22 44

   * - Form
     - Selected by
     - Evaluator
     - Values come from
   * - 1
     - ``symbols``, no execution block
     - ``evaluate_legacy``
     - Python evaluated in-process, per symbol
   * - 2
     - top-level ``pipeline``
     - ``evaluate_legacy``
     - A DAG MAGNET generates from the block
   * - 3
     - ``kwdagger``
     - ``evaluate_new``
     - A KWDagger DAG and its aggregate result store

The base schema accepts all three, but the CLIs do not overlap: passing a
form-3 card to ``evaluate_legacy`` exits with an error naming ``evaluate_new``,
and ``evaluate_new`` rejects forms 1 and 2. Form 2 additionally emits a
``DeprecationWarning``; nested ``kwdagger.pipeline`` is not deprecated.

Forms 2 and 3 are mutually exclusive, and a card with neither must define
``symbols``.

To convert a card between forms, see
:doc:`migration_guide_v0_0_2_to_v0_1_0`.


Common fields
-------------

Every form shares the fields in this section.


Top level
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``title``
     - string
     - yes
     - One line mapping the claim to its context.
   * - ``description``
     - string
     - yes
     - Prose explanation. Normally a block scalar.
   * - ``version``
     - string
     - yes
     - YAML numbers are coerced, so ``version: 1.0`` becomes ``"1.0"``.
   * - ``organizations``
     - list of string
     - yes
     - Organizations accountable for the card.
   * - ``submitter``
     - object
     - yes
     - See `submitter`_.
   * - ``tags``
     - list of string
     - yes
     - May be empty; the key must be present.
   * - ``links``
     - list of object
     - yes
     - May be empty; the key must be present. See `links`_.
   * - ``claim``
     - object
     - yes
     - See `claim`_.
   * - ``category``
     - string
     - no
     - Free-form grouping label.
   * - ``claim_aggregation_strategy``
     - object
     - no
     - See `claim_aggregation_strategy`_. Defaults to ``{type: all}``.
   * - ``symbols``
     - map
     - conditional
     - Required in form 1. See `symbols`_.
   * - ``theory``
     - object
     - no
     - See `theory`_.

``title``, ``description``, ``version``, ``organizations``, ``submitter``,
``tags``, ``links``, and ``category`` are validated and recorded, never read
during evaluation. They are not inert: the whole card is copied into the result
card, and the run directory is named from a hash of the entire card, so editing
any of them yields a distinct run.


submitter
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``name``
     - string
     - yes
     - Person or team.
   * - ``email``
     - string
     - yes
     - Not format-checked.


links
~~~~~

Each element of ``links``:

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``title``
     - string
     - yes
     - Display name.
   * - ``url``
     - string
     - yes
     - Not format-checked.
   * - ``type``
     - string
     - yes
     - Free-form. Shipped cards use ``software``, ``dataset``, ``model``,
       ``asset``.


claim
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``python``
     - string
     - yes
     - Executable Python, normally ``assert`` statements. Not syntax-checked
       at validation time.

The claim is evaluated once per cell. A holding assertion gives ``VERIFIED``,
a failing one ``FALSIFIED``, any other exception ``INCONCLUSIVE``.

What the claim can name depends on the form. In forms 1 and 2 it names resolved
symbols. In form 3 it additionally addresses the KWDagger aggregate row:

``<namespace>.<node>.<field>``
    Fully qualified, where namespace is ``metrics``, ``params``, or
    ``resolved_params``. Always unambiguous, and always what the result card
    records as consumed evidence.

``<node>.<field>``
    Short form. Resolves where the node reports that field in exactly one
    namespace. Where two namespaces report it and disagree, the claim refuses
    and names them rather than choosing.


claim_aggregation_strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduces per-cell verdicts to the single result-card verdict.

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``type``
     - string
     - yes
     - ``all``, ``any``, or ``fraction``. Typed as an open string, so an
       unrecognized value passes validation and raises at reduce time.
   * - ``parameters.threshold``
     - float
     - conditional
     - Required when ``type`` is ``fraction``.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Type
     - Verdict
   * - ``all``
     - ``FALSIFIED`` if any cell falsified, else ``INCONCLUSIVE`` if any cell
       inconclusive, else ``VERIFIED``. Applied when the key is omitted.
   * - ``any``
     - ``VERIFIED`` if any cell verified, else ``INCONCLUSIVE`` if any cell
       inconclusive, else ``FALSIFIED``.
   * - ``fraction``
     - ``VERIFIED`` when verified/total meets ``threshold``, else
       ``FALSIFIED``. Inconclusive cells count in the denominator only.

Zero cells reduces to ``INCONCLUSIVE``. Unknown keys in this block are kept.


symbols
~~~~~~~

``symbols`` maps a symbol name to its definition. The name is what the claim
sees.

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``type``
     - string
     - no
     - Documentary annotation. Not enforced against the resolved value.
   * - ``value``
     - any
     - no
     - A literal.
   * - ``sweep``
     - list
     - no
     - Expands to one evaluation per combination. Forms 1 and 2 only;
       rejected by ``evaluate_new``.
   * - ``depends_on``
     - list of string
     - no
     - Symbols resolved first, whose bindings are visible to this symbol's
       ``python``. ``depends`` is an accepted alias. Defaults to ``[]``.
   * - ``python``
     - string
     - no
     - Executable Python assigning the symbol's own name.
   * - ``metadata``
     - object
     - no
     - See `symbol metadata`_.

A symbol must define at least one of ``type``, ``value``, ``sweep``, or
``python`` — unless it carries ``metadata`` and nothing else, which declares a
label over a value an executor already produced.

In form 3, a symbol with no ``value``, ``sweep``, or ``python`` is filled from
the evidence row using the same addressing as the claim: a qualified
``<namespace>.<node>.<field>``, or a shorter tail match on segment boundaries.
Unlike the claim, an ambiguous symbol match warns and picks one rather than
refusing. Name the column outright where it could be ambiguous.


symbol metadata
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``kind``
     - string
     - no
     - ``model``, ``dataset``, or ``metric``. Classifies what the symbol
       denotes.
   * - ``display``
     - boolean
     - no
     - Surface this symbol in downstream displays. Read by the dashboard, not
       by the evaluators.
   * - ``display_name``
     - string
     - no
     - Human-readable label. Also the key under which this symbol's computed
       metric appears in the verdict; falls back to the symbol name.
   * - ``define_metric``
     - object
     - no
     - Promotes the symbol to a computed metric. See `define_metric`_.

Symbols carrying ``metadata`` are written to ``symbol_metadata.json`` in the
result card.


define_metric
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``objective``
     - string
     - no
     - ``minimize`` or ``maximize``. Records direction of improvement; does not
       affect the computed value.
   * - ``aggregation_strategy``
     - object
     - yes
     - Reducer over this symbol's per-cell values.
   * - ``aggregation_strategy.type``
     - string
     - yes
     - ``mean``, ``max``, ``min``, or ``custom``. ``custom`` raises
       ``NotImplementedError``.
   * - ``aggregation_strategy.parameters``
     - map of string to float
     - no
     - Accepted; not consumed by any reducer.

``aggregation_strategy`` is required whenever ``define_metric`` is present.
``kind: metric`` and ``define_metric`` are independent; neither implies the
other.

The reducer runs *across* cells, not within one. See
:doc:`migration_guide_v0_0_2_to_v0_1_0` for in-cell reduction and the
requirement that a defined metric be present in every aggregated row.


theory
~~~~~~

Static links from the card's empirical code to formal statements, optional and
supported identically in all three forms. Every object under ``theory`` rejects
unknown keys. For the relation vocabulary, source annotations, index file
format, and the generated ``theory.json``, see :doc:`theory_links`.

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``links``
     - list of object
     - no
     - Card-level relations to whole statements.
   * - ``empirical_sources``
     - list of string
     - no
     - Paths, relative to the card, read statically for annotations.
   * - ``entries``
     - list of object
     - no
     - Theory entries declared inline.
   * - ``indexes``
     - list of string
     - no
     - Paths, relative to the card, to theory index files.

``theory.links[]``:

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``relation``
     - string
     - yes
     - ``tests``, ``approximates``, or ``motivates``.
   * - ``ref``
     - string
     - yes
     - A theory entry id. May not contain ``::``; premise relations belong in
       source annotations.
   * - ``note``
     - string
     - no
     - Prose justifying the relation.

``theory.entries[]``:

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``id``
     - string
     - yes
     - Reference identity. May not contain ``::``.
   * - ``kind``
     - string
     - yes
     - ``theorem``, ``conjecture``, ``question``, or ``definition``.
   * - ``statement``
     - string
     - conditional
     - At least one of ``statement`` or ``declaration`` is required.
   * - ``declaration``
     - string
     - conditional
     - Fully-qualified formal declaration name.
   * - ``formalization``
     - object
     - no
     - ``system`` (required), ``repository``, ``revision``. A ``repository``
       without a ``revision`` is an error.
   * - ``source_path``
     - string
     - no
     - Path to the file holding the formal statement.
   * - ``premises``
     - list of object
     - no
     - Each has ``id`` (required), ``type``, ``statement``. Ids must be unique
       within an entry.


Form 1: Python symbols
----------------------

Selected by defining ``symbols`` with no execution block. Adds no fields
beyond the common set. Every symbol resolves in-process, in dependency order,
via ``value``, ``sweep``, or ``python``.

.. code:: yaml

    symbols:
      int_range_even:
        type: List[int]
        metadata:
          display_name: "Set of Even Numbers"
        python: |
          int_range_even = [n for n in range(-10, 11) if n % 2 == 0]

      int_range_odd:
        type: List[int]
        depends_on: [int_range_even]
        python: |
          int_range_odd = [n + 1 for n in int_range_even]

Reference: ``magnet/cards/simple.yaml``.


Form 2: generated pipeline (deprecated)
---------------------------------------

Selected by a top-level ``pipeline`` block. MAGNET generates a KWDagger DAG
from it. The block is an unvalidated mapping of node name to node definition;
MAGNET reads ``out_paths.results_fpath`` from the generated node.

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``pipeline``
     - map of node name to object
     - yes
     - Not schema-validated beyond being a mapping.

Node definitions use the same vocabulary as form 3, typically ``executable``,
``algo_params``, ``out_paths``. A list value under ``algo_params`` is expanded
as a sweep axis.

Reference: ``magnet/cards/llama_pipeline.yaml``.


Form 3: KWDagger
----------------

Selected by a ``kwdagger`` block. This is the only form ``evaluate_new``
accepts, and the only one that separates the campaign a run requests from the
evidence a claim is judged on.


Form 3 top level
~~~~~~~~~~~~~~~~

Two top-level keys exist only in this form.

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``name``
     - string
     - no
     - Short machine identifier, distinct from ``title``. Must match
       ``^[A-Za-z0-9_-]+$``: it becomes a tmux session name and a path
       component. Derived as ``<parent-dir>_<stem>`` with a ``UserWarning``
       when absent. Declare it so it survives the file moving.
   * - ``evidence``
     - object
     - no
     - Evidence selection policy. See `evidence`_.

``name`` is defined on ``NewEvaluationRecipeSchema`` only; under
``evaluate_legacy`` it is an unknown top-level key and dropped.


kwdagger
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``pipeline``
     - string or object
     - yes
     - The DAG. Three spellings, below.
   * - ``result_node``
     - string
     - conditional
     - Node whose accumulated aggregate rows are candidate evidence. Optional
       on the base schema, **required** by ``evaluate_new``.
   * - ``matrix``
     - object
     - no
     - Parameter grid. See `matrix`_.
   * - *any other key*
     - any
     - no
     - Forwarded verbatim to ``kwdagger schedule --params``. Not validated by
       MAGNET, so a typo here is not caught.

Everything except ``result_node`` becomes the ``params`` payload handed to
KWDagger. MAGNET validates only what it reads.

Name nodes carefully. A claim can address a node by its bare name, so a node
named for a Python builtin shadows it; the shipped example uses
``enumeration`` rather than ``enumerate``.


kwdagger.pipeline
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Spelling
     - Meaning
   * - mapping
     - The DAG inline. Keys: ``nodes`` (required), ``edges``, ``__doc__``.
   * - ``"path.yaml"``
     - A declarative pipeline file. Relative paths resolve against the card's
       directory. Recognized by a ``.yaml``, ``.yml``, or ``.json`` suffix.
   * - ``"pkg.mod.factory()"``
     - A Python callable returning a ``kwdagger.Pipeline``. Required when nodes
       need behavior that is not expressible as data.

The inline and file spellings share one vocabulary, defined by KWDagger's
declarative loader.


Declarative node keys
~~~~~~~~~~~~~~~~~~~~~

Each entry under ``nodes`` is keyed by node name — do not set ``name`` inside
the spec. Any key outside this table is a load error.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``executable``
     - Shell command for the node. Required (``command`` is an alias).
   * - ``algo_params``
     - Parameters that affect results. Part of the node's process identity.
   * - ``perf_params``
     - Parameters that do not affect results, such as worker counts.
   * - ``in_paths``
     - Input port names, as a list or set.
   * - ``out_paths``
     - Output port name to filename, relative to the node directory.
   * - ``primary_out_key``
     - Which output identifies the node's completion.
   * - ``config``
     - Static configuration merged into the node.
   * - ``group``
     - Grouping label for scheduling.
   * - ``node_dpath`` / ``group_dpath``
     - Override the node's or group's output directory.
   * - ``slurm_options``
     - Options forwarded to the Slurm backend.
   * - ``setup`` / ``teardown``
     - Shell string or list run before / always after the command.
   * - ``result``
     - Inline result mapping for the aggregate loader.
   * - ``load_result``
     - Dotted path to a callable that reads this node's result. Use it when
       the primary output is not JSON the generic loader can parse.
   * - ``metrics``
     - Names within the loaded result to expose under ``metrics.<node>``.
   * - ``vantage_points``
     - Aggregate viewpoints declared on the node.
   * - ``class``
     - Dotted path to a ``kwdagger.ProcessNode`` subclass backing this node.
       See `MAGNET node capabilities`_.


Edges
~~~~~

``edges`` is a list; each element is a string or a mapping.

.. code:: yaml

    edges:
      - llama_predict.results_fpath -> llama_compare.scores_fpath

      - src: materialize_run.out_dpath
        dst: llama_predict.run_dpaths
        gather:
          group_by: []
          order_by: [model, subject]
          require: all_success

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``src`` / ``dst``
     - ``node.port`` endpoints. ``src_key`` / ``dst_key`` may name the ports
       separately instead.
   * - ``gather.group_by``
     - Parameter names shared by source and target. ``[]`` gathers every
       source instance into each target. An entry may be
       ``{src: ..., dst: ...}`` when the sides name it differently.
   * - ``gather.order_by``
     - Source parameters ordering manifest members. Defaults to process id.
   * - ``gather.require``
     - Completion policy. Currently ``all_success`` only.

A gather delivers its sources to the target as a newline-delimited path
manifest.


matrix
~~~~~~

``matrix`` maps ``<node>.<param>`` to a value or list of values, expanded as a
cross product. It also accepts ``include`` and ``exclude`` lists of partial
assignments, following GitHub Actions matrix semantics.

.. code:: yaml

    matrix:
      estimate.samples: 20000
      estimate.seed: [1, 2, 3, 4, 5]

``--params`` deep-merges into the ``kwdagger`` block at the command line; see
:doc:`migration_guide_v0_0_2_to_v0_1_0` for override and caching semantics.


MAGNET node capabilities
~~~~~~~~~~~~~~~~~~~~~~~~

``class: magnet.process_node.MagnetProcessNode`` gives a node containerization
and endpoint leasing. It widens the declarative node keys by four:

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Key
     - Default
     - Meaning
   * - ``endpoint_params``
     - ``()``
     - Names of node parameters whose values are infer-stack endpoint aliases.
       Leasing does nothing without at least one.
   * - ``lease_ttl``
     - ``8h``
     - Lease lifetime. A backstop for a hard-killed process; normal completion
       releases the lease.
   * - ``lease_timeout``
     - ``1800``
     - Seconds to wait for capacity.
   * - ``lease_queue``
     - ``true``
     - Wait behind other leases rather than failing on unavailable capacity.

Container settings are invocation-level flags (``--container_image`` and
companions), not card fields; leasing is enabled with ``--per_node_leasing``.

A node without this capability cannot be containerized, and passing
``--container_image`` when no node can use it is an error rather than a
silently inert setting. When both apply, the lease wraps the container
command.


evidence
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 40

   * - Field
     - Type
     - Required
     - Notes
   * - ``scope``
     - string
     - no
     - ``all`` (default) judges every row the result store holds for this
       recipe, whoever produced it. ``requested`` judges only rows for
       result-node computations this invocation asked for, cached or fresh.

Unknown keys in this block are rejected. ``--evidence_scope`` overrides the
card's value for one invocation.

Use ``requested`` when the verdict must describe this invocation. Under
``all``, an artifact left in the root by an earlier, superseded grid keeps
voting.


Cross-field rules
-----------------

- At most one of ``kwdagger`` and ``pipeline``.
- With neither, ``symbols`` is required.
- A symbol needs one of ``type``, ``value``, ``sweep``, ``python`` — or
  ``metadata`` alone.
- ``depends_on`` and ``depends`` must agree when both are given.
- ``claim_aggregation_strategy.type: fraction`` requires
  ``parameters.threshold``.
- ``theory.entries[]`` needs a ``statement`` or ``declaration``; its ``id`` may
  not contain ``::``; premise ids must be unique.
- ``theory.entries[].formalization.repository`` requires ``revision``.
- ``theory.links[].ref`` may not contain ``::``.
- Under ``evaluate_new``: ``kwdagger.result_node`` is required, ``pipeline``
  must be absent, no symbol may declare ``sweep``, ``name`` must match
  ``^[A-Za-z0-9_-]+$``, and ``evidence.scope`` must be ``all`` or
  ``requested``.


Unknown keys
------------

The blocks disagree on what happens to a key the schema does not define. This
matters when a card validates but misbehaves.

.. list-table::
   :header-rows: 1
   :widths: 40 16 44

   * - Block
     - Unknown keys
     - Consequence
   * - card top level
     - dropped
     - Typos pass validation silently.
   * - ``claim``, ``submitter``, ``links[]``, ``symbols[]``, ``metadata``,
       ``define_metric``
     - dropped
     - Same, one level down.
   * - ``kwdagger``
     - kept
     - Forwarded to KWDagger by design.
   * - ``claim_aggregation_strategy``
     - kept
     - Preserved so extra reducer parameters survive.
   * - ``evidence``, ``theory`` and below
     - rejected
     - Validation error.
   * - ``kwdagger.pipeline`` nodes and top level
     - rejected
     - Load error from KWDagger's declarative loader, naming the allowed keys.

Note that ``name`` is dropped under ``evaluate_legacy`` and meaningful under
``evaluate_new``.


The Evaluation Result Card
--------------------------

Evaluating a recipe card writes a run bundle under
``<output_path>/<recipe-hash>_<timestamp>/`` containing the verdict, one record
per cell, the recipe as evaluated, and — when the recipe declares a ``theory``
block — the resolved theory report. Both evaluators write result cards; the
newer one records more. See :doc:`evaluation_result_card`.

Under ``--dry_run`` no evidence is loaded and no claim is evaluated: the result
is ``NOT_EVALUATED`` and no verdict is written.
