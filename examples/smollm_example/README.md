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
infer-stack run --endpoint smol-135 --ttl 1h ${SLURM_JOB_GPUS:+...} -- \
    docker run --rm --network host -v /repo:/repo -e OPENAI_BASE_URL ... <image> \
        python -m smollm_example.cli.ask_model --endpoint=smol-135 ...
```

Cache guard outermost, so a node whose output already exists neither leases nor
starts a container. The `${SLURM_JOB_GPUS...}` word is unexpanded on purpose:
the allocation it names does not exist yet on the host that rendered the string.

The example uses kwdagger's serial backend, so `ask` deliberately does **not**
pass infer-stack's `--queue`: its cells cannot contend with each other. If old
leases occupy the GPUs, the example fails on placement instead of waiting for
capacity. The one-hour lease TTL is only a backstop for a hard-killed process;
`infer-stack run` releases normally in its `finally` path.

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

The matrix contains **endpoint aliases**, not Hugging Face model IDs. The two
checked-in catalogs are read-only example fixtures; do not point infer-stack's
catalog editor at them. The editor rewrites YAML structurally, which discards
comments even when the resulting catalog is semantically equivalent.

For a custom real model, make an ignored local copy first, then let the
infer-stack CLI ensure the model and endpoint entries exist. From this directory:

```bash
LOCAL_CATALOG="$PWD/catalog.local.yaml"
test -e "$LOCAL_CATALOG" || cp catalog.yaml "$LOCAL_CATALOG"

infer-stack catalog model add qwen05 \
    --catalog="$LOCAL_CATALOG" \
    --source=hf://Qwen/Qwen2.5-0.5B-Instruct

infer-stack catalog endpoint add qwen-05 \
    --catalog="$LOCAL_CATALOG" \
    --model=qwen05 \
    --max-model-len=2048 \
    --gpu-mem=0.2 \
    --extra-args='--enforce-eager --dtype=half' \
    --reclaim=stop
```

`catalog.local*.yaml` is gitignored. Those `catalog ... add` commands are safe
to rerun: an identical existing definition is reported as already up to date,
while a conflicting definition fails and shows the differing fields rather than
overwriting it.

If an earlier experiment already rewrote the tracked `catalog.yaml`, preserve
its custom entries in the ignored local file and restore the documented fixture:

```bash
cp catalog.yaml catalog.local.yaml
git restore -- catalog.yaml
```

The restored file gets its comments back from Git; continue editing only
`catalog.local.yaml` through the infer-stack CLI.

Point `run.sh` at that local catalog and override the kwdagger matrix:

```bash
SMOLLM_CATALOG="$LOCAL_CATALOG" \
    ./run.sh --params='matrix: {ask.endpoint: [qwen-05]}'
```

or compare models from different families in one sweep:

```bash
SMOLLM_CATALOG="$LOCAL_CATALOG" \
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
on the host instead.

The build context is **only this example directory**, and the Dockerfile copies
only `__init__.py` and `cli/`. The card, pipeline, catalogs, README, `run.sh`, the
rest of the MAGNET checkout, and run outputs never enter the image. In
particular, adding an infer-stack endpoint to `catalog.yaml` or overriding the
kwdagger matrix does not invalidate the node image.

MAGNET itself is installed before those files are copied, from a GitHub archive
pinned to commit `cf9cc968d7a88470657e7938addfb3a1a6d0f986`. This keeps the
pre-release example reproducible without baking the current checkout into the
image. Once `aiq-magnet` is released, the intended change is just the Dockerfile
argument default, for example:

```dockerfile
ARG MAGNET_INSTALL="aiq-magnet==0.1.0"
```

The image remains a slim Python base plus MAGNET's core, with no GPU or model
weights. It also does not install infer-stack. `infer-stack run` executes on
the host, outside the container; all that reaches inside is `OPENAI_BASE_URL`
and `OPENAI_API_KEY`, and the node talks to that endpoint over plain HTTP.

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

### Developer smoke test

`test.sh` runs all four combinations of real/mock endpoints and
container/host node execution:

```bash
./test.sh
```

This is deliberately a developer-machine test, not a CI entry point. It assumes
a usable NVIDIA GPU, Docker, MAGNET with the `leasing` extra, and an initialized
infer-stack backend. Every variant gets a fresh output root so existing kwdagger
artifacts cannot turn the smoke test into a cache-only run. Artifacts are kept
under `runs/dev-smoke-<timestamp>/` by default; set `SMOLLM_TEST_RUNS` to choose
another root.

Before starting, `test.sh` refuses to run if infer-stack reports any active
leases. This default is non-destructive because a developer may have unrelated
work using the same daemon. On a dedicated developer machine, opt into a clean
start with:

```bash
./test.sh --release
```

`--release` runs `infer-stack release --all --yes --evict` before the smoke
variants, clearing active leases and idle deployments. If a variant itself
leaks a lease, later variants are skipped rather than entering a capacity wait.
A failed variant that leaves the lease pool clean does not prevent the remaining
variants from running.

At exit, `test.sh` always enumerates all four variants in order, with their
real/mock endpoint mode, container/host node mode, status, and artifact path.
Variants not reached because of a lease leak are reported as skipped.

The two containerized variants run first. The first may build the node image;
the second should reuse it. Switching from the real catalog to the mock catalog
does not change the image build context, so `pip install` should remain cached.

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
  class: magnet.execution.MagnetYamlProcessNode
  endpoint_params: [endpoint]
  executable: "python -m smollm_example.cli.ask_model"
```

Nothing in `cli/` changes when it does. `tests/test_yaml_container_nodes.py`
pins that destination and skips itself on a kwdagger that cannot do it yet.

All three Python DAG nodes already use `MagnetYamlProcessNode`. Containerization
and endpoint leasing are independent capabilities on that node: the invocation
may enable either, both, or neither. Only `Ask.endpoint_params` names an
endpoint, so enabling leasing leaves `Items` and `Compare` unwrapped while the
same container setting can still apply to all three.

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
| `test.sh` | developer-only smoke test for all four real/mock × container/host modes |
| `Dockerfile` | the node image `run.sh` builds by default: slim Python plus MAGNET core |
| `smollm_kwdagger.yaml` | the card: claim, evidence scope, sweep |
| `pipeline.py` | the DAG: three nodes, one gather edge, which one leases |
| `cli/make_items.py` | the dummy dataset, generated from a seed |
| `cli/ask_model.py` | one leased endpoint, every question, over plain `urllib` |
| `cli/compare_answers.py` | the gather manifest reduced to coverage and agreement |
| `catalog.yaml` | checked-in real-model fixture: `smol-135` / `smol-360` on vLLM |
| `catalog-mock.yaml` | checked-in mock fixture with the same two aliases |
| `catalog.local*.yaml` | ignored local copies for custom endpoint experiments |
