# aiq-evals recipes (integration plan M7)

These recipes get their evaluations through
[aiq-evals](https://github.com/AIQ-Kitware/aiq-evals) using
`magnet.backends.aiq_evals.EvaluationNode`. Each exposes `metrics.evaluate.score`
only when its `select` names exactly one metric and the evidence is eligible.

| Recipe | Engine | Kind |
| --- | --- | --- |
| `helm_generation.yaml` | HELM (local simple model) | generation |
| `olmo_generation.yaml` | OLMo Eval (mock provider) | generation |
| `inspect_generation.yaml` | Inspect (fixture provider) | generation |
| `olmo_agent.yaml` | OLMo Eval (OpenAI Agents scaffold, OpenAI-compatible endpoint) | agent/tool |
| `inspect_agent.yaml` | Inspect (`use_tools`, local sandbox) | agent/tool |
| `mixed_engines.yaml` | HELM and Inspect, compared through an explicit node | comparison |

Engines never run in MAGNET's interpreter. Set `perf_params.worker_python` on
every evaluation node to that engine's worker environment (for example with
`--params`). The OLMo and Inspect recipes use aiq-evals' deterministic fixture
tasks, so their workers need an aiq-evals source checkout. HELM needs only
`crfm-helm`.

`magnet evaluate_new <recipe> --dry-run` compiles the request shape only. It
resolves nothing, starts no model, and runs no task code. A real run resolves
every request in its worker before kwdagger assigns node identities.
