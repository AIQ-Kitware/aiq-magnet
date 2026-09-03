# SmolLM example — leasing and containers, with nothing else in the way

```bash
./run.sh
```

That is the whole workflow. By default it serves the real SmolLM2 135M and
360M checkpoints with vLLM on an NVIDIA GPU, and runs the MAGNET node commands
inside the example container.

```bash
./run.sh --mock                    # GPU-free simulator; nodes stay containerized
./run.sh --no-container            # real models, but run node commands on the host
./run.sh --mock --no-container     # simulator plus host node commands
./run.sh --dry_run=1               # compile using the default GPU/container mode
./run.sh --help                    # show wrapper options without doing setup work
```

`--mock` and `--no-container` are independent and combine. `--help` only
prints the wrapper's usage text: it does not require MAGNET, infer-stack, a GPU,
or Docker. Anything else goes through to `magnet evaluate_new`.

`run.sh` expects `magnet` and `infer-stack` to already be installed in the
active environment. Installing MAGNET with its leasing extra provides both:

```bash
pip install 'aiq-magnet[leasing]'
```

The default path also requires a usable NVIDIA GPU visible to `nvidia-smi` and
Docker for the node container. `run.sh` checks both before scheduling anything.
Use `--mock` when no GPU is available, and `--no-container` when the node
commands should run directly on the host.

Then infer-stack needs one machine-level backend choice. `run.sh` checks this
before building or scheduling anything and prints the command if it is missing:

```bash
infer-stack config init --yes --backend compose
```

## What it is

Two SmolLM2 checkpoints — the 135M and 360M instruct models — are each asked
the same eight generated addition questions, and the answers are compared.

```
items ──> ask (× 2 endpoints, LEASED) ──gather──> compare
```

Nothing here is a benchmark. The card claims one thing: **every endpoint
answered every question**. That is true or false for reasons that live entirely
in the plumbing — a lease that never resolved, a container that cannot reach
the gateway, an alias with no catalog entry — and it is unaffected by whether
the models can add. Which is what makes it worth running against a simulator.

## What it demonstrates

**Only the node that needs a model leases one.** `items` writes the dataset and
`compare` reduces the answers; neither can use a GPU, and neither holds one.
Wrapping the whole evaluation in a single lease — the obvious alternative —
would hold both models from the first node to the last.

**The lease is outside the container.** Acquiring one needs the Docker daemon
and the shared ledger, both on the host; consuming the endpoint happens inside.
Being inside is also what lets the container inherit `OPENAI_BASE_URL` and
`OPENAI_API_KEY` with no extra plumbing:

```
test -e answers.json || \
infer-stack run --endpoint smol-135 --ttl 8h --queue ${SLURM_JOB_GPUS:+...} -- \
    docker run --rm --network host -v /repo:/repo -e OPENAI_BASE_URL ... <image> \
        python -m smollm_example.cli.ask_model --endpoint=smol-135 ...
```

Cache guard outermost, so a node whose output already exists neither leases nor
starts a container. The `${SLURM_JOB_GPUS...}` word is unexpanded on purpose:
the allocation it names does not exist yet on the host that rendered the string.

**The endpoint is an ordinary matrix axis.** `ask.endpoint` sweeps like any
other parameter, and naming it in `endpoint_params` is what also makes its
value the catalog alias that cell acquires.

### Override the matrix

`run.sh` passes `--params` through to `magnet evaluate_new`, where it is
merged into the card's `kwdagger:` block. That makes the shipped matrix a
default rather than a fixed model list. For example, run only the 135M endpoint:

```bash
./run.sh --params='matrix: {ask.endpoint: [smol-135]}'
```

The matrix contains **endpoint aliases**, not Hugging Face model IDs. To try a
model that is not already in this example's real catalog, first use infer-stack's
catalog CLI to ensure the model and endpoint entries exist. From this directory,
this adds a small Qwen instruct model under the stable alias `qwen-05`:

```bash
infer-stack catalog model add qwen05 \
    --catalog="$PWD/catalog.yaml" \
    --source=hf://Qwen/Qwen2.5-0.5B-Instruct

infer-stack catalog endpoint add qwen-05 \
    --catalog="$PWD/catalog.yaml" \
    --model=qwen05 \
    --max-model-len=2048 \
    --gpu-mem=0.2 \
    --extra-args='--enforce-eager --dtype=half' \
    --reclaim=stop
```

Those `catalog ... add` commands are safe to rerun. If the named entry already
has exactly that definition, infer-stack reports it as up to date; if it exists
with a different definition, the command fails and shows the differing fields
instead of overwriting it. The catalog editor validates the result before it
writes the file.

Now the kwdagger matrix can select the new endpoint just like either shipped
SmolLM endpoint:

```bash
./run.sh --params='matrix: {ask.endpoint: [qwen-05]}'
```

or compare models from different families in one sweep:

```bash
./run.sh --params='matrix: {ask.endpoint: [smol-135, qwen-05]}'
```

No pipeline or node code changes. `ask.endpoint` controls both the kwdagger
matrix cell and the alias leased by infer-stack. The example node uses the
OpenAI chat-completions API, so an instruct/chat model such as the Qwen example
is the simplest substitution; a base completion model such as GPT-2 would need
additional serving/chat-template decisions that distract from this example.

**A gather edge turns N cells into one comparison.** `group_by: []` hands the
single `compare` cell every endpoint's answers as a manifest, so the comparison
names exactly what it read rather than globbing a directory.

**Nothing is restated.** Each node's inputs, outputs and parameters come from
its CLI's own kwconf declaration, so the card carries none of them — only the
claim, the evidence scope and the sweep. And no node hand-rolls a
`load_result`: they write `result.metrics`, which kwdagger's generic loader
reads.

## Containers

`./run.sh` builds the `Dockerfile` beside this file and runs every node command
inside it by default. `./run.sh --no-container` opts out and runs those commands
on the host instead. The image is built rather than named, because an image tag
that exists on the machine that built it is exactly the kind of thing that does
not travel.

The image is small — a slim Python base plus MAGNET's core, no GPU and no
weights — because parsing HELM output and leasing endpoints are both extras
that a card like this one does not need. Note what it does *not* install:
infer-stack. `infer-stack run` executes on the host, outside the container;
all that reaches inside is `OPENAI_BASE_URL` and `OPENAI_API_KEY`, and the node
talks to that endpoint over plain HTTP. If the image ever needs infer-stack,
something has been layered the wrong way round.

Observed with the default container mode on the guest VM: four containerized
node cells, two leases, and the leases only on `ask`.

## Simulated or real

The two catalogs beside this file declare **the same two aliases**, which is
the demonstration: the card names `smol-135` and `smol-360` and never learns
which one answered. `run.sh` picks `catalog.yaml`; `--mock` picks
`catalog-mock.yaml`. Switching between real weights and a simulator is a
catalog choice, not an edit to the card.

Observed on the guest VM (2026-09-02, `--mock`, no GPU):

```
RESULT:      VERIFIED
smol-135: answered 8/8, rate 1.0, exact 0.0, mean 0.188s
smol-360: answered 8/8, rate 1.0, exact 0.0, mean 0.188s
compare : coverage 1.0, agreement 1.0
```

`exact_rate` is 0.0 because the simulator returns random text — the first
answer to "What is 7 + 7?" was `Alas, poor Yorick! I`. `agreement` is 1.0
because both simulated endpoints return the same canned sequence, seed
regardless. **Neither number means anything here**, which is why the card
claims about neither.

### CI smoke test

GitHub Actions runs the exact containerized mock path with `./run.sh --mock`
on an ordinary `ubuntu-latest` runner. The job installs MAGNET with the
`leasing` extra, initializes infer-stack's Compose backend, disables Open WebUI
(the UI is outside this example's execution path), and leaves the LiteLLM
gateway enabled. This exercises `run.sh`, the node image build, kwdagger,
per-node leasing, the simulator containers, the gateway, and the final claim
without requiring a GPU.

## Why the DAG is Python

A leasing node has to be told which of its parameters holds a catalog alias,
and kwdagger's node-spec allow-list is closed, so a declarative card cannot say
`endpoint_params` until kwdagger 0.4.1 ships `extra_node_spec_keys`. In Python
it is a class attribute, so this runs on the released 0.4.0 today.

Once 0.4.1 lands, `pipeline.py` can go and the card can say it directly:

```yaml
ask:
  class: magnet.leasing.LeasedYamlProcessNode
  endpoint_params: [endpoint]
  executable: "python -m smollm_example.cli.ask_model"
```

Nothing in `cli/` changes when it does. `tests/test_yaml_container_nodes.py`
pins that destination and skips itself on a kwdagger that cannot do it yet.

## Files

This example lives in `examples/` beside magnet, not inside the package — it is
a *consumer* of magnet, the way a team's own repository is, and it is imported
the same way: by being on the path. `run.sh` arranges that, and the image COPYs
it in. Nothing here ships in the magnet wheel.

Scaffolding and executables are separate on purpose. Everything in `cli/` is an
ordinary command-line program — reads files, talks to an endpoint, writes JSON
— runnable by hand, debuggable without a scheduler, and knowing nothing about
kwdagger, containers or leases.

| | |
|---|---|
| `run.sh` | the whole workflow |
| `Dockerfile` | the node image `run.sh` builds by default: slim Python plus MAGNET core |
| `smollm_kwdagger.yaml` | the card: claim, evidence scope, sweep |
| `pipeline.py` | the DAG: three nodes, one gather edge, which one leases |
| `cli/make_items.py` | the dummy dataset, generated from a seed |
| `cli/ask_model.py` | one leased endpoint, every question, over plain `urllib` |
| `cli/compare_answers.py` | the gather manifest reduced to coverage and agreement |
| `catalog.yaml` | `smol-135` / `smol-360` on vLLM, on a GPU |
| `catalog-mock.yaml` | the same two aliases, simulated, no GPU |
