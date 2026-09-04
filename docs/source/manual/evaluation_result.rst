Evaluation Result Card Schema
=============================

Evaluating an :doc:`Evaluation Recipe Card <evaluation_recipe_card>` writes an
**Evaluation Result Card**: a run bundle recording a verdict, the evidence
behind it, and the recipe that produced it.

Result cards are written by MAGNET and read by downstream tooling. Nothing here
is hand-edited, and none of it is schema-validated on write — this page
documents what is produced, not what is accepted.

Both evaluators write to ``<output_path>/<hash>_<timestamp>/``, where the hash
covers the entire recipe card. Editing any field of a recipe, including fields
never read during evaluation, yields a distinct bundle.


Bundle layout
-------------

.. list-table::
   :header-rows: 1
   :widths: 32 12 56

   * - Path
     - Written by
     - Contents
   * - ``card.yaml``
     - both
     - The recipe card as evaluated, after any ``--params`` merge.
   * - ``verdict.json``
     - both
     - The card-level verdict. See `verdict.json`_.
   * - ``results/<id>/verdict.json``
     - both
     - One record per cell. See `Cell records`_.
   * - ``symbol_metadata.json``
     - both
     - The ``metadata`` block of every symbol declaring one, keyed by symbol
       name. Written only when at least one symbol has metadata.
   * - ``log``
     - both
     - Run log. ``INFO``, or ``DEBUG`` under ``--verbose``; ``MAGNET_LOG_LEVEL``
       overrides both.
   * - ``theory.json``
     - both
     - Written when the recipe has a ``theory`` block. Format in
       :doc:`theory_links`.
   * - ``requested_runs.json``
     - ``evaluate_new``
     - Per-process execution state for this invocation. See
       `requested_runs.json`_.
   * - ``kwdagger``
     - ``evaluate_new``
     - Symlink to the shared KWDagger artifact root, which lives outside the
       bundle at ``<output_path>/_kwdagger``.

``<id>`` is a 12-character hash of the cell's resolved symbols under
``evaluate_legacy``, and the artifact/computation identity under
``evaluate_new``.


verdict.json
------------

Both evaluators write the same three keys. ``evaluate_new`` adds three more,
and one further key is optional in both.

.. list-table::
   :header-rows: 1
   :widths: 26 14 12 48

   * - Field
     - Evaluator
     - Always
     - Contents
   * - ``result``
     - both
     - yes
     - ``VERIFIED``, ``FALSIFIED``, ``INCONCLUSIVE``, or ``NOT_EVALUATED``.
   * - ``claim_aggregation_strategy``
     - both
     - yes
     - The strategy applied, as resolved. ``{"type": "all"}`` when the recipe
       omitted it.
   * - ``claims``
     - both
     - yes
     - Cell ids, matching the directory names under ``results/``.
   * - ``metrics``
     - both
     - no
     - Computed metric values, keyed by ``display_name`` or by symbol name.
       Present only when a symbol declared ``define_metric``.
   * - ``evidence``
     - new
     - yes
     - ``scope`` (``all`` or ``requested``), ``available`` (rows judged), and
       ``discovered`` (rows found before scope was applied).
   * - ``requested_work``
     - new
     - no
     - Summary of the campaign this invocation requested. Omitted when nothing
       was requested.
   * - ``provenance``
     - new
     - no
     - The ``--provenance`` mapping, recorded verbatim.

``available`` and ``discovered`` differ only under ``evidence.scope:
requested``, where the gap is the number of accumulated rows this invocation
did not ask for.

No ``verdict.json`` is written under ``--dry_run``. The in-memory result card
reports ``NOT_EVALUATED``; ``theory.json`` and ``requested_runs.json`` are
still written, which makes a dry run a way to check a recipe's theory links
without executing anything.

The two evaluators produce the same filename with different key sets and no
version marker. Read ``evidence`` as the discriminator: its presence means the
bundle came from ``evaluate_new``.


Cell records
------------

One record per evaluated cell, at ``results/<id>/verdict.json``.

.. list-table::
   :header-rows: 1
   :widths: 26 14 12 48

   * - Field
     - Evaluator
     - Always
     - Contents
   * - ``status``
     - both
     - yes
     - ``VERIFIED``, ``FALSIFIED``, or ``INCONCLUSIVE`` for this cell.
   * - ``output``
     - both
     - yes
     - The assertion message or exception text. Empty on success.
   * - ``symbols``
     - both
     - yes
     - The concrete claim inputs. Under ``evaluate_new`` this combines
       resolved recipe symbols with the qualified evidence leaves the claim
       consumed.
   * - ``timestamp``
     - both
     - yes
     - ISO-8601 completion time.
   * - ``cell``
     - new
     - yes
     - Artifact/computation identity, stable across MAGNET runs that judge the
       same evidence.
   * - ``artifact``
     - new
     - no
     - Path to the primary KWDagger result the row was loaded from.
   * - ``consumed``
     - new
     - no
     - Qualified evidence columns the claim actually accessed.
   * - ``evidence``
     - new
     - no
     - The complete qualified aggregate row offered to the claim.

``consumed`` is a subset of the keys in ``evidence``: what the claim read,
versus what it was given.


requested_runs.json
-------------------

A list of per-process records describing this invocation's finite request.
Operational provenance only: it records what was requested and what state those
processes reached, where ``verdict.json`` records what the evidence implies.
Nothing here changes a verdict.

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Field
     - Always
     - Contents
   * - ``process_id``
     - yes
     - KWDagger process identity.
   * - ``node``
     - yes
     - Node name within the DAG.
   * - ``schedule_status``
     - yes
     - ``new_submission``, ``skipped`` (KWDagger had it already), or
       ``disabled``.
   * - ``attempt_status``
     - yes
     - ``not_attempted``, ``not_started``, ``running``, ``passed``,
       ``skipped``, or ``failed``.
   * - ``returncode``
     - yes
     - Exit status, or ``null`` when the process did not finish.
   * - ``output_available``
     - yes
     - Whether the expected primary output exists on disk.
   * - ``enabled``
     - yes
     - Whether KWDagger considered the process for submission.
   * - ``expected_output``
     - no
     - Path to the primary output, when the node declares one.
   * - ``stat_fpath`` / ``log_fpath``
     - no
     - Job status and log paths, when a job was submitted.

``verdict.json``'s ``requested_work`` is the summary of this file: a
``processes`` count, histograms of ``schedule_status`` and ``attempt_status``,
and an ``outputs_available`` count.

A ``skipped`` schedule status with ``output_available: true`` is the normal
cached case, not a failure. The shared ``_kwdagger`` store is not part of a
dashboard upload bundle.
