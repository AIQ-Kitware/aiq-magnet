# Coding benchmark smoke recipes

Harbor and SWE-bench use `EvaluationNode` and the existing kwdagger scheduler.
Prepare a concrete Verified recipe with the evaluator's five frozen instances:

```bash
python -m magnet.examples.aiq_evals.swebench --kind verified \
  --endpoint coding --worker_python /path/to/verified-worker/bin/python \
  --evals_root /path/to/aiq-magnet-evals --output verified.yaml
magnet evaluate_new verified.yaml --output_path results --per_node_leasing=True
```

For Pro V2, use `--kind pro --pro_root /path/to/swe-bench-pro --image_pins pins.json`
and the Harbor worker interpreter. `pins.json` maps the frozen Ansible smoke
instance to its physical `sha256:` image ID. The evaluator's `profile_request`
pins the upstream revision, manifest, locked agent and fresh replay protocol.
Put the checkout's `v2/tooling` on `PYTHONPATH` for the worker. This recipe uses
the `local-harbor/pro-v2-offline-replay/v1` protocol, not a publisher score.

For a generic Harbor task, use `--kind harbor --task_path /path/to/task --agent
python:your_module:Agent` with the Harbor worker. The default oracle is useful
for protocol checks. A reusable task needs pinned sandbox images and a fully
identified agent runtime; installed agents currently remain non-reusable.

The generated claim checks evidence eligibility. Replace it with a scientific
claim appropriate to the experiment. These recipes run tiny smoke sets and do
not launch Verified's 500 tasks, HARD-51, or Pro's 642 tasks.

Leased nodes query configured scientific serving provenance before scheduling,
bind its digest as the model cache token, and verify the lease's public JSON
descriptor before inference. A catalog change also prevents reuse of an old
queued measurement. Placement, lease IDs, endpoint URLs and credentials are
operational. Mutable weights, images or other unresolved inputs disable reuse.
An explicit conflicting cache token is rejected; `verify_serving_provenance:
false` is the explicit opt-out for caller-owned identity assertions.

For outer Docker workers, mount the worker environment and its Python runtime,
both source checkouts, task inputs, output/store directories, Docker CLI/plugins
and `/var/run/docker.sock` at their host paths. Forward the Docker socket's
group with `--group-add`. The public lease descriptor is mounted read-only at
execution time. Preflight uses the same container wrapper as execution.

Inspect patches can be exported from the referenced run and checked in fresh
official grader containers using the evaluator's
`dev/regrade_swebench_verified.py`. Pro generation patches and replay lineage
stay in the referenced native bundle. Errors and incomplete grading never
supply a claim-facing zero score.

Run CPU/Docker acceptance with:

```bash
AIQ_MAGNET_EVALS_DIR=/path/to/aiq-magnet-evals \
INFER_STACK_DIR=/path/to/infer_stack dev/ci/swebench_vm_acceptance.sh
```

This uses real Harbor containers, deterministic tool calls and infer-stack's
null serving backend. It proves protocol and scheduling behavior; it does not
load model weights or establish GPU/model capability.
