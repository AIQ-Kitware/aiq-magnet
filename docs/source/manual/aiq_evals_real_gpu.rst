Real-GPU aiq-magnet-evals acceptance
====================================

Purpose
-------

``dev/ci/aiq_evals_real_gpu.sh`` is the opt-in end-to-end acceptance test for
MAGNET's aiq-magnet-evals backend.  It is intentionally stronger and more
expensive than ``dev/ci/aiq_evals_integration.sh``.

The ordinary integration gate uses infer-stack's null serving backend so it can
exercise lease bookkeeping deterministically without needing a GPU.  This gate
uses the production-shaped path instead and covers three distinct evaluation
modes::

    Inspect one-shot benchmark
      -> MAGNET EvaluationNode
      -> infer-stack real lease
      -> Compose/vLLM on a physical GPU
      -> Dockerized node
      -> aiq-magnet-evals / Inspect
      -> native metric/evidence

    Inspect agentic evaluation
      -> forced model tool call
      -> real Python tool execution
      -> post-tool model turn
      -> normalized Inspect trajectory

    OLMo Eval agentic evaluation
      -> OpenAI Agents scaffold
      -> real model-selected tool call
      -> real OLMo registered tool execution
      -> subsequent model turn
      -> normalized OLMo trajectory

Every evaluator request contains ``http://127.0.0.1:9/v1`` as a deliberately
dead ``base_url``.  A measurement can complete only when the infer-stack lease
environment supplies the real operational endpoint to the Dockerized node.

What constitutes a pass
-----------------------

All three cases must satisfy the common integration invariants:

* Docker and NVIDIA GPUs are actually available.
* The selected infer-stack controller reports ``backend: compose``.
* MAGNET schedules a Dockerized ``EvaluationNode`` with endpoint leasing enabled.
* infer-stack records a fresh lease claiming the endpoint requested by that case.
* The node reaches the real vLLM endpoint despite the dead URL in the declarative
  evaluator request.
* aiq-magnet-evals records ``status=succeeded`` and materializes attempt, native,
  and canonical run artifacts.
* Native coverage is complete enough to produce eligible evidence.
* MAGNET verifies ``metrics.evaluate.eligible``.

The two agentic cases additionally inspect aiq-magnet-evals' normalized sample
trajectories and require evidence of an *executed tool result*.  Advertising a
tool to the model, configuring an agent scaffold, or merely emitting a tool-call
request is not enough to pass.

Reuse is intentionally not part of this expensive hardware gate.  Canonical-run
reuse and the rule that a reused evaluation should not acquire another lease are
covered by the deterministic MAGNET/aiq-magnet-evals lease integration tests.
Keeping that concern separate prevents a reuse/caching regression from obscuring
the question this gate answers: whether a *fresh* Inspect or OLMo evaluation can
reach a real GPU-served model and, for agentic cases, actually execute a tool.

The model's benchmark score is printed as a diagnostic but is not itself an
infrastructure pass condition.  A model can answer the tiny benchmark item
incorrectly while the transport/evaluator integration is healthy.  Conversely,
an agentic test that never reaches a tool-result turn fails even if its evaluator
process exits successfully.

Tool-capable endpoint
---------------------

vLLM automatic tool choice requires a model-specific parser.  The default real
endpoint is ``qwen2.5-7b`` and Qwen2.5 uses vLLM's ``hermes`` tool-call parser.
The runner therefore prepares a temporary endpoint alias named
``qwen2.5-7b-agentic-e2e`` by copying the base endpoint and appending::

    --enable-auto-tool-choice
    --tool-call-parser=hermes

The alias also receives ``reclaim: stop`` so its GPU process is removed after
the lease ends.  ``runtime.extra_args`` is used intentionally: those launch
arguments participate in infer-stack's deployment identity, preventing the
tool-enabled endpoint from accidentally coalescing with a resident non-tool
vLLM process.

The original catalog is copied into the timestamped artifact directory before
this temporary alias is written and is restored byte-for-byte by an EXIT trap.
Automatic catalog mutation is refused when infer-stack reports another active
lease.  If ``MAGNET_REAL_GPU_AGENTIC_ENDPOINT`` already exists, the runner does
not rewrite the catalog but verifies that its ``runtime.extra_args`` contain the
required tool flags.

For a non-Qwen model, set the parser appropriate to that model, for example::

    MAGNET_REAL_GPU_ENDPOINT=<base-alias> \
    MAGNET_REAL_GPU_AGENTIC_ENDPOINT=<tool-enabled-alias> \
    MAGNET_REAL_GPU_TOOL_CALL_PARSER=<vllm-parser> \
    dev/ci/aiq_evals_real_gpu.sh

Python and evaluator workers
----------------------------

The gate is Python 3.13-only.  It creates reusable environments under its work
directory for:

* MAGNET + aiq-magnet-evals + infer-stack;
* Inspect with the OpenAI client;
* the Dockerized MAGNET node; and
* the pinned OLMo Eval checkout.

OLMo Eval is checked out at
``73ade80e24f796af55caeb8fd7b75a7f3fd607fd`` and synced from its frozen lock
with the ``litellm`` and ``agents`` extras on Python 3.13.  The OLMo project at
that revision declares support for Python 3.13, so the real-GPU gate does not
retain the older Python 3.12-only setup used by previous integration scripts.

The Inspect agent test uses a test-only task in
``tests/aiq_evals_real_gpu_tasks.py``.  Its first action is forced to the
``aiq_real_gpu_double`` function using Inspect's ``ToolFunction`` directive, so
whether the model invokes a tool is deterministic at the API level.  The OLMo
agent case uses aiq-magnet-evals' ``aiq_example_tool`` task with the real
``openai_agents`` scaffold and asserts the resulting OLMo trajectory contains a
tool-result turn.

Run it
------

From the aiq-magnet repository root::

    AIQ_MAGNET_EVALS_DIR=~/code/aiq-magnet-evals \
    INFER_STACK_DIR=~/code/infer_stack \
    MAGNET_REAL_GPU_ALLOWED_GPUS=0 \
    dev/ci/aiq_evals_real_gpu.sh \
        ~/.cache/aiq-real-tests/magnet-aiq-evals-real-gpu-agentic

``INFER_STACK_DIR`` is optional.  When supplied, that checkout is installed
editable into the Python 3.13 MAGNET environment, so the gate covers the local
infer-stack source rather than an unrelated installed release.

A successful run reports three pytest tests: Inspect generation, Inspect agent,
and OLMo agent.  The runner prints a final VERIFIED marker only when pytest
returns zero.  The two agent tests print normalized
trajectories under ``pytest -s`` so the actual tool turn can be inspected in the
saved log.

The runner is a child shell.  It does not change error-handling options in the
caller's interactive terminal.  Its pytest output is captured with ``tee`` and
the real pytest exit status is recovered from ``PIPESTATUS[0]`` inside the child
script.

Failure interpretation
----------------------

A failure before the process graph is produced is a prerequisite, Python-worker,
or control-plane setup problem.  A failure while infer-stack converges an
endpoint is a serving/GPU problem.  A failed evaluator record is an
engine/transport problem.  A missing fresh lease is a MAGNET-to-infer-stack
integration failure.  Ineligible evidence is an aiq-magnet-evals contract
failure.  An agentic run with no normalized tool-result turn is specifically an
agent/scaffold/tool-calling failure and must not be counted as an end-to-end
agentic pass.
