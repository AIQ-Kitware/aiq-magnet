# aiq-magnet-evals recipes (integration plan M7)

These recipes obtain their evaluations through
[aiq-magnet-evals](https://github.com/Erotemic/aiq-magnet-evals) using
`magnet.backends.aiq_evals.EvaluationNode`. Each exposes `metrics.evaluate.score`
only when its `select` names exactly one metric and the evidence is eligible.

| Recipe | Engine | Kind |
| --- | --- | --- |
| `helm_generation.yaml` | HELM (built-in `simple_mcqa`, local `simple/model1`) | generation |
| `olmo_generation.yaml` | OLMo Eval (mock provider) | generation |
| `inspect_generation.yaml` | Inspect (deterministic `aiq_example` provider) | generation |
| `olmo_agent.yaml` | OLMo Eval (OpenAI Agents scaffold, OpenAI-compatible endpoint) | agent/tool |
| `inspect_agent.yaml` | Inspect (`use_tools`, tool runs in the `local` sandbox) | agent/tool |
| `mixed_engines.yaml` | HELM and Inspect, compared through an explicit node | comparison |

The OLMo and Inspect tasks are the installed `magnet_evals.examples` modules, so
no aiq-magnet-evals source checkout is needed.

## Running

Engines never run in MAGNET's interpreter. Install MAGNET with
`aiq-magnet[aiq-magnet-evals]`, then give every evaluation node its engine's worker
interpreter:

```bash
magnet evaluate_new magnet/examples/aiq_evals/inspect_generation.yaml \
    --output_path ./results_aiq_evals \
    --params='matrix: {evaluate.worker_python: /path/to/inspect-venv/bin/python}'
```

`mixed_engines.yaml` has two evaluation nodes: set `helm.worker_python` and
`inspect.worker_python`.

A worker environment needs its engine at the verified pin: HELM
`crfm-helm==0.5.14`, Inspect `inspect-ai==0.3.272`, or an OLMo Eval checkout at
`73ade80e` synced from its lock. It does not need aiq-magnet-evals installed:
the worker imports the caller's copy.

`olmo_agent.yaml` expects an OpenAI-compatible endpoint at
`http://127.0.0.1:8000/v1`. The deterministic example endpoint is

```bash
python -m magnet_evals.examples.chat_server --port 8000 &
export OPENAI_API_KEY=example-local-key   # any value; it is never persisted
```

To use a leased model instead, set `perf_params.endpoint` to an infer-stack
alias and the model binding to the name it serves, and run with leasing enabled.
Other roles can be leased too, e.g. an Inspect grader:
`perf_params.endpoints: {grader: judge-alias}`. One lease holds every alias. The
node decides at run time, under the store's acquisition lock, whether a lease
is needed at all: a stored result is reused without one.

## Identity, reuse, and dry runs

`magnet evaluate_new <recipe> --dry-run` compiles the request shape and checks
the selector and coverage policy. It resolves nothing, starts no model, and runs
no task code. A real run first resolves every request in the node's own
environment (host, or its container), and the resolved measurement identity
becomes part of the kwdagger node identity.

Nodes that differ only in `select` share one native evaluation. If they run
concurrently (for example under `--backend tmux`), one executes it and the
others wait and reuse it.
