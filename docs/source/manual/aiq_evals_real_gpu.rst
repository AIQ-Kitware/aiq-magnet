Real-GPU aiq-magnet-evals acceptance
====================================

Purpose
-------

``dev/ci/aiq_evals_real_gpu.sh`` is the opt-in end-to-end acceptance test for
MAGNET's aiq-magnet-evals backend.  It is intentionally stronger and more
expensive than ``dev/ci/aiq_evals_integration.sh``.

The ordinary integration gate uses infer-stack's null serving backend so it can
exercise lease bookkeeping deterministically without needing a GPU.  This gate
uses the production-shaped path instead::

    MAGNET EvaluationNode
      -> infer-stack run
      -> Compose
      -> vLLM on a real NVIDIA GPU
      -> Dockerized MAGNET node
      -> aiq-magnet-evals
      -> Inspect worker
      -> OpenAI-compatible generation
      -> native metric/evidence
      -> MAGNET card

The evaluator request contains ``http://127.0.0.1:9/v1`` as a deliberately dead
``base_url``.  The measurement can complete only when the infer-stack lease
environment supplies the real operational endpoint to the Dockerized node.

What constitutes a pass
-----------------------

The test asserts all of the integration invariants that matter:

* Docker and NVIDIA GPUs are actually available.
* The selected infer-stack controller reports ``backend: compose``.
* The selected endpoint exists in the real infer-stack catalog.
* MAGNET schedules an ``EvaluationNode`` with endpoint leasing enabled.
* infer-stack records a new lease claiming that endpoint.
* The Dockerized MAGNET node reaches the real endpoint despite the dead URL in
  the declarative evaluator request.
* aiq-magnet-evals records the measurement as ``status=succeeded`` and produces
  an attempt plus canonical run artifact.
* Inspect returns eligible evidence to MAGNET.
* MAGNET verifies the integration claim ``metrics.evaluate.eligible``.
* Rescheduling the already materialized measurement takes no second GPU lease.

The model's benchmark *score* is printed as a diagnostic but is deliberately
not a pass condition.  Whether one particular model answers one benchmark item
correctly is model behavior, not an infrastructure invariant.

Prerequisites
-------------

The host needs:

* Python 3.13 available to ``uv``;
* Docker plus the Compose plugin;
* NVIDIA driver/runtime visible to Docker and ``nvidia-smi``;
* infer-stack on ``PATH`` and configured with the Compose backend;
* an OpenAI-compatible real GPU endpoint already present in the infer-stack
  catalog; and
* an aiq-magnet-evals checkout.

The default endpoint is ``qwen2.5-7b`` with model revision
``Qwen/Qwen2.5-7B-Instruct``.  ``infer-stack catalog suggest --apply`` creates
that endpoint on hosts where the suggested catalog includes it.  Any other real
OpenAI-compatible vLLM endpoint can be selected explicitly.

Run it
------

From the aiq-magnet repository root::

    AIQ_MAGNET_EVALS_DIR=~/code/aiq-magnet-evals \
    INFER_STACK_DIR=~/code/infer_stack \
    MAGNET_REAL_GPU_ALLOWED_GPUS=0 \
    dev/ci/aiq_evals_real_gpu.sh

The ``INFER_STACK_DIR`` override is optional.  When supplied, the runner installs
that checkout editable into its Python 3.13 MAGNET environment, so the test
covers the local infer-stack source rather than an installed release.

To use another endpoint::

    AIQ_MAGNET_EVALS_DIR=~/code/aiq-magnet-evals \
    INFER_STACK_DIR=~/code/infer_stack \
    MAGNET_REAL_GPU_ENDPOINT=smol-135 \
    MAGNET_REAL_GPU_MODEL_REVISION=HuggingFaceTB/SmolLM2-135M-Instruct \
    MAGNET_REAL_GPU_ALLOWED_GPUS=0 \
    dev/ci/aiq_evals_real_gpu.sh

The runner creates reusable Python 3.13 environments under its work directory
and a unique timestamped artifact directory for each invocation.  It does not
set shell error options in the caller, and it does not require ``pipefail`` in
an interactive terminal.  The pytest command is logged with ``tee`` and its
actual exit status is recovered from ``PIPESTATUS[0]`` inside the child script.

Control-plane behavior
----------------------

This test intentionally uses the infer-stack control plane selected by the
caller's environment/default configuration.  It does not create a second
Compose controller because infer-stack Compose controllers share the same
``infer-stack`` Docker project namespace.  If ``INFER_STACK_CONFIG_DIR`` or
``INFER_STACK_DATA_DIR`` is intentionally set before invoking the runner, those
values are preserved and the test prints the resulting ``infer-stack status``
before doing work.

Potentially stale lease-derived environment variables are removed inside the
runner before pytest starts: ``INFER_STACK_BACKEND``, ``INFER_STACK_CATALOG``,
``INFER_STACK_LEASE_ID``, ``OPENAI_BASE_URL``, and every
``INFER_STACK_ENDPOINT_*`` variable.  This prevents a previous lease from making
the acceptance test pass without acquiring a fresh one.

Failure interpretation
----------------------

A failure before the MAGNET process graph is produced is a prerequisite or
control-plane setup problem.  A failure while infer-stack converges the endpoint
is a serving/GPU problem.  A failed evaluator record is an engine/transport
problem.  A missing new lease is a MAGNET-to-infer-stack integration failure.
An ineligible evidence row is an aiq-magnet-evals contract failure.

A score of zero by itself is *not* an integration failure.  The artifact log
prints the selected native evaluator metrics so content-level behavior can be
reviewed separately from infrastructure correctness.
