# SmolLM example — leasing and containers, with nothing else in the way

A card whose subject is the *execution machinery*. There is no benchmark, no
download and no HELM: two SmolLM2 checkpoints are asked eight generated
addition questions, and the card claims only that every question came back.

That claim is the point. It is true or false for reasons that live entirely in
the plumbing — a lease that never resolved, a container that cannot reach the
gateway, an endpoint alias with no catalog entry — and it is unaffected by
whether the models are any good at arithmetic. So it stays meaningful when you
run it against a simulator, which is how you can run it with no GPU.

```
items ──> ask (× 2 endpoints, LEASED) ──gather──> compare
```

## What it demonstrates

**Only the node that needs a model leases one.** `items` writes the dataset and
`compare` reduces the answers; neither can use a GPU, and neither holds one.
Wrapping the whole evaluation in a single lease — the obvious alternative —
would hold both SmolLM models from the first node to the last.

**The lease is outside the container.** Acquiring one needs the Docker daemon
and the shared ledger, both on the host; consuming the endpoint happens inside.
Being inside is also what lets the container inherit `OPENAI_BASE_URL` and
`OPENAI_API_KEY` with no extra plumbing. The rendered command:

```
test -e answers.json || \
infer-stack run --endpoint smol-135 --ttl 8h --timeout 1800 --queue ${SLURM_JOB_GPUS:+...} -- \
    docker run --rm --network host --user 1000:1000 -v /repo:/repo -w "$PWD" \
        -e OPENAI_BASE_URL -e OPENAI_API_KEY -e PYTHONPATH ... aiq-eval-node:latest \
        python -m magnet.examples.smollm_example.ask_model --endpoint=smol-135 ...
```

Cache guard outermost, so a node whose output exists neither leases nor starts a
container. The `${SLURM_JOB_GPUS...}` word is unexpanded on purpose: the
allocation it names does not exist yet on the host that rendered the string.

**The endpoint is an ordinary matrix axis.** `ask.endpoint` sweeps like any
other parameter. `nodes.AskModel` declares it in `endpoint_params`, which is
what also makes its value the catalog alias that cell acquires. Two endpoints,
two cells, each holding only the model it is using.

**A gather edge turns N cells into one comparison.** `group_by: []` hands the
single `compare` cell every endpoint's answers as a newline-delimited manifest,
so the comparison names exactly what it read instead of globbing a directory.

**No `load_result` anywhere.** Every node writes `result.metrics`, which is
what kwdagger's generic loader reads. Nothing here hand-rolls a loader.

## Running it

Two catalogs sit beside the card with **the same two aliases**, which is the
demonstration: the card never learns which one answered.

```bash
# No GPU: a simulator serves both aliases.
docker pull ghcr.io/llm-d/llm-d-inference-sim:v0.9.0
export INFER_STACK_BACKEND=compose
export INFER_STACK_CATALOG=$PWD/magnet/examples/smollm_example/catalog-mock.yaml

python -m magnet.evaluation_new \
    --path=magnet/examples/smollm_example/smollm_kwdagger.yaml \
    --output_path=./runs/smollm --backend=serial --per_node_leasing=1
```

Swap `INFER_STACK_CATALOG` to `catalog.yaml` for real weights on a GPU — vLLM,
SmolLM2-135M and -360M. That is the only change. Add
`--container_image=<image>` to run the node commands in a container too; the
image needs magnet installed.

Observed on the guest VM (2026-09-02, mock catalog, no GPU):

```
RESULT:      VERIFIED
smol-135: answered 8/8, rate 1.0, exact 0.0, mean 0.188s
smol-360: answered 8/8, rate 1.0, exact 0.0, mean 0.188s
compare : coverage 1.0, agreement 1.0, endpoints smol-135,smol-360
```

`exact_rate` is 0.0 because the simulator returns random text — the first
answer to "What is 7 + 7?" was `Alas, poor Yorick! I`. `agreement` is 1.0
because both simulated endpoints return the same canned sequence, seed
regardless. **Neither number means anything here**, which is why the card
claims about neither. `coverage` does mean something, and it is what the claim
is about.

## Files

| | |
|---|---|
| `smollm_kwdagger.yaml` | the card: three nodes, one gather edge, one claim |
| `nodes.py` | `AskModel` (leases) and `PlainStep` (containerized, no lease) |
| `make_items.py` | the dummy dataset, generated from a seed |
| `ask_model.py` | one leased endpoint, every question, over plain `urllib` |
| `compare_answers.py` | the gather manifest reduced to coverage and agreement |
| `catalog.yaml` | `smol-135` / `smol-360` on vLLM, on a GPU |
| `catalog-mock.yaml` | the same two aliases, simulated, no GPU |

`tests/test_smollm_example.py` covers both halves: what the card *renders*
(which nodes lease, how the wrappers nest) and what the nodes *compute* (all
three run end to end against a stub OpenAI server, no container or model
needed).
