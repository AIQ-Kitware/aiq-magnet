Saved evaluation results
========================

Serialized results are authoritative; logs describe execution. Existing dashboard
bundles and older claim records containing only resolved ``symbols`` are supported.

Read and present
----------------

.. code-block:: console

   magnet results list evaluation_runs --format=json
   magnet results show evaluation_runs/RECORD --output=report.md

``list`` defaults to text and skips caches and mutable aliases. Choose an explicit
record, not ``latest``. ``show`` defaults to Markdown on stdout; ``--format=text``
or ``--format=json`` also works. It reads ``card.yaml``, aggregate ``verdict.json``
and referenced ``results/`` claim records without importing pipelines, executing
claims or scheduling work. Missing claim files are reported. A missing aggregate
verdict displays as ``NOT_RECORDED``, not a scientific verdict.

``evaluate_new`` writes regenerable ``report.md`` and ``report.txt``. Reports need
neither original paths, logs nor caches. They show declared metrics, terminal-node
measurements and nested records. Claim records retain qualified evidence and
consumed fields; flattened ancestor fields do not preserve every producer
instance. Export supporting artifacts for refitting or deeper audits.

Recover a readout
-----------------

.. code-block:: console

   magnet evaluate_new saved_card.yaml --output_path=evaluation_runs \
       --backend=serial --readout_only=1 --evidence_scope=requested

Use the original output root and parameters (``--params`` if needed). Readout-only
requires the original pipeline and dependencies, compiles without submission,
judges available cached evidence under the card's selection rules, and writes a
new timestamped record. It runs no missing nodes, inference or model leases.
``--dry_run`` instead judges no evidence and writes no verdict; the flags cannot
be combined. Use ``results show`` for an already recorded verdict.

Export and verify
-----------------

.. code-block:: console

   magnet results export evaluation_runs/RECORD review/RECORD
   magnet results verify review/RECORD

The destination must not exist. Exports preserve native scientific bytes, add
reports, and record SHA-256 hashes and sizes in ``bundle_manifest.json``. Core
exports need no DAG copy. Optional export arguments:

* ``--supporting=terminal``: primary artifacts referenced by claim records.
* ``--supporting=requested``: additionally, expected outputs in
  ``requested_runs.json``; ``--nodes=features,summary`` selects requested nodes.
* ``--include_logs``: available application and launcher console logs.
* ``--attachments='{"analysis.json": "/path/to/analysis.json"}'``: explicit
  supplemental files under ``attachments/``, covered by the manifest. A JSON/YAML
  mapping file also works; attachments remain separate from native results.

Available ``job_config.json`` files accompany supporting outputs. Missing optional
files are listed without changing the verdict; omitted logs do not imply an
incomplete bundle. Exports neither scan entire caches nor copy serving settings.
Recorded readout-time hashes reject changed supporting outputs/configurations;
older bundles without hashes remain usable.

Absolute source paths remain provenance. Portable manifest locations, resolved by
``ResultBundle.artifact_path(original_path)``, survive relocation and original-file
deletion; portable bundles can be re-exported. ``verify`` checks included bytes
and required claim references, returning nonzero on errors. Hashes establish
integrity relative to the manifest, not scientific validity or manifest authenticity.

Invocation identity and provenance
---------------------------------

.. code-block:: console

   magnet evaluate_new card.yaml --record_path_file=/path/to/attempt/record.json

Before scheduling, MAGNET atomically writes the exact record ID/directory to this
per-attempt descriptor outside the result directory. Launchers should use it
rather than newest-directory timestamps. ``execution.json`` records the backend,
dry-run/readout flags, software/source snapshots, timestamps and state:
``evaluated``, ``dry_run``, ``readout``, ``interrupted`` or ``error``. State describes
this invocation, not completion of the accumulated DAG. CLI SIGINT/SIGTERM retain
interruption metadata.

Arbitrary runtime settings, container options and environments are omitted;
MAGNET does not strip credentials. Caller ``--provenance`` is preserved unchanged.
Providers/adapters must supply actual weights, precision and remote-service facts;
aliases do not establish them. Output/configuration hashes observe readout-time
bytes, not the execution settings that produced earlier cached computations.
