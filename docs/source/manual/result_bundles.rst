Reading, recovering and exporting evaluation results
===================================================

Serialized results are the persistent evaluation output. Console and worker
logs explain execution; they are not required to recover measurements or
display a recorded verdict. Existing dashboard bundles remain supported.

Find the evaluation records under an output directory:

.. code-block:: console

   magnet results list evaluation_runs
   magnet results list evaluation_runs --format=json

This lists timestamped records and prunes shared caches and mutable aliases.
Select an explicit record, rather than depending on ``latest``. Present it
without loading its pipeline, executing its Python claim or scheduling work:

.. code-block:: console

   magnet results show evaluation_runs/RECORD
   magnet results show evaluation_runs/RECORD --format=text
   magnet results show evaluation_runs/RECORD --format=json
   magnet results show evaluation_runs/RECORD --output=report.md

The reader uses ``card.yaml``, the aggregate ``verdict.json`` and the claim
records referenced under ``results/``. Native claim records preserve their
complete qualified evidence row and consumed fields. Older records with only
resolved ``symbols`` still render. An interrupted invocation without a verdict
is displayed as ``NOT_RECORDED``; this is not a scientific verdict. Missing
referenced claim files are reported explicitly.

``evaluate_new`` now writes ``report.md`` and ``report.txt`` beside its result.
These are derived presentations. They can be regenerated from serialized
results without the original machine, logs or cache. Declared metrics and
terminal-node measurements are shown, including nested records. Inherited
ancestor fields in a flattened gather row are not a lossless archive of all
producer instances; use supporting artifacts for refitting or deeper audits.

Recover a readout from surviving computation outputs
--------------------------------------------------

If a previous invocation stopped before writing its verdict, but its shared
computation store survives, use the normal evaluator in readout-only mode:

.. code-block:: console

   magnet evaluate_new saved_card.yaml --output_path=evaluation_runs \
       --backend=serial --readout_only=1 --evidence_scope=requested

Use the original output root and parameterization. The command compiles the
requested graph with submission disabled, then evaluates the available
evidence and writes a new timestamped record. It does not run missing nodes,
acquire model leases or regenerate inference. The card's existing evidence
selection rules still apply. ``--params`` is available when needed.

This differs from ``--dry_run``: a dry run judges no evidence and writes no
verdict. The two flags cannot be combined. Reading a completed bundle with
``results show`` is simpler and does not require the original pipeline or
its dependencies; readout-only evaluation does require those dependencies.

Portable exports
----------------

.. code-block:: console

   magnet results export evaluation_runs/RECORD review/RECORD
   magnet results verify review/RECORD

An export keeps native scientific files byte-for-byte, adds human-readable
reports, and writes ``bundle_manifest.json`` with SHA-256 hashes and sizes.
The destination must not exist. A core bundle needs no shared-DAG copy.

Choose additional evidence explicitly:

.. code-block:: console

   magnet results export evaluation_runs/RECORD review/RECORD \
       --supporting=terminal
   magnet results export evaluation_runs/RECORD review/RECORD \
       --supporting=requested --nodes=features,summary --include_logs

``terminal`` includes primary artifacts referenced by the claim records.
``requested`` additionally includes expected outputs listed by the saved
``requested_runs.json``; ``--nodes`` optionally selects node names. Available
``job_config.json`` files are included. Missing optional supporting files are
listed in the manifest without changing the recorded scientific verdict.
No recursive cache or serving-environment copy occurs. Absolute source paths
remain provenance; the manifest supplies portable artifact locations.
Where readout-time output/configuration hashes were recorded, an export
rejects supporting files that have changed since that snapshot. Historical
bundles without those hashes remain readable and exportable.

Supplemental analyses can be attached without replacing native verdicts:

.. code-block:: console

   magnet results export evaluation_runs/RECORD review/RECORD \
       --attachments='{"analysis.json": "/path/to/analysis.json"}'

A JSON/YAML mapping file also works. Explicit files are copied under
``attachments/`` and covered by the integrity manifest. They remain distinct
from the native result. ``--include_logs`` includes available application and
launcher console logs; omitted logs do not make a bundle incomplete.

``results verify`` checks included bytes and required claim references, and
returns a nonzero status on integrity errors. SHA-256 detects changes relative
to the export manifest, not scientific validity or authenticity of the
manifest itself. The library ``ResultBundle.artifact_path(original_path)``
resolves exported artifacts after moving a bundle, even if original paths
are unavailable. Re-exporting portable bundles is supported.

Invocation identity and provenance
---------------------------------

.. code-block:: console

   magnet evaluate_new card.yaml --record_path_file=/path/to/attempt/record.json

Before scheduling, MAGNET atomically writes its exact record ID and directory
to that descriptor. Launchers should allocate a separate descriptor per
attempt outside the result directory, not find the newest directory by
modification time. The native ``execution.json`` records the scheduler backend,
dry-run/readout-only flags, software/source snapshots,
timestamps and evaluated/dry-run/readout/interrupted/error state. These states
describe an invocation, not whether an entire accumulated DAG is complete.
SIGINT and SIGTERM received by the CLI preserve interruption metadata.

Arbitrary runtime settings, container options and environments are not copied
into this snapshot. MAGNET does not strip credentials from supplied data.
Caller-supplied ``--provenance`` belongs to the caller and is preserved
unchanged. Providers/adapters must supply facts
about actual weights, precision and remote services; MAGNET does not infer
those from aliases. Output/configuration hashes in the operational inventory
are observations at readout. They do not assign current execution settings
to earlier cached computations.
