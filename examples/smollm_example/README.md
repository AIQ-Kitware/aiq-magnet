# SmolLM leasing and container example

This example runs a three-node MAGNET evaluation against two SmolLM endpoints.
By default, infer-stack serves the real SmolLM2 135M and 360M instruct models
with vLLM on an NVIDIA GPU, and MAGNET runs each node command in a container.

```bash
./run.sh
```

Available modes:

```bash
./run.sh --mock                    # simulator endpoints; no GPU
./run.sh --no-container            # real models; node commands on the host
./run.sh --mock --no-container     # simulator endpoints; node commands on the host
./run.sh --dry_run=1               # compile without running
./run.sh --help
```

`--mock` and `--no-container` are independent. Other arguments are passed to
`magnet evaluate_new`.

## Requirements

Install MAGNET with leasing support:

```bash
pip install 'aiq-magnet[leasing]'
```

Initialize an infer-stack backend once per machine:

```bash
infer-stack config init --yes --backend compose
```

The default mode also requires Docker and an NVIDIA GPU visible to
`nvidia-smi`. `--mock` removes the GPU requirement. `--no-container` removes
the Docker requirement.

## Pipeline

The DAG is defined in `pipeline.py`:

```text
items -> ask (one cell per endpoint) -> compare
```

- `items` writes eight generated addition questions.
- `ask` sends every question to one endpoint. Its `endpoint` parameter is also
  the infer-stack catalog alias leased by that cell.
- `compare` gathers every endpoint's answers and reports coverage and agreement.

The card claims only that every endpoint answered every question. Arithmetic
accuracy and cross-model agreement are reported but are not part of the claim.
This keeps the same coverage claim meaningful for real and simulated endpoints.

## Per-node leasing

Only `ask` declares `endpoint_params`, so only `ask` acquires model leases.
`items` and `compare` do not hold GPUs.

The lease wraps the container command:

```text
infer-stack run --endpoint smol-135 ... -- \
    docker run ... \
        python -m smollm_example.cli.ask_model --endpoint=smol-135 ...
```

This keeps infer-stack on the host while forwarding its OpenAI-compatible
endpoint into the node container through `OPENAI_BASE_URL` and
`OPENAI_API_KEY`.

The example uses kwdagger's serial backend, so `Ask.lease_queue` is disabled.
A busy GPU therefore produces a placement failure instead of waiting behind an
unrelated lease. The one-hour TTL bounds stale leases left by a hard-killed
process; normal `infer-stack run` execution releases its lease when the command
ends.

Under Slurm, MAGNET defers the GPU allow-list expansion until the job starts so
infer-stack sees the GPUs allocated to that job rather than every GPU visible
on the host.

## Endpoint matrix

The card's default matrix contains infer-stack endpoint aliases:

```yaml
matrix:
  ask.endpoint:
    - smol-135
    - smol-360
```

Override the matrix through `--params`:

```bash
./run.sh --params='matrix: {ask.endpoint: [smol-135]}'
```

### Custom endpoints

The checked-in catalogs are example fixtures. For local changes, copy the real
catalog to a gitignored `catalog.local*.yaml` file:

```bash
LOCAL_CATALOG="$PWD/catalog.local.yaml"
test -e "$LOCAL_CATALOG" || cp catalog.yaml "$LOCAL_CATALOG"
```

Add a model and endpoint with infer-stack:

```bash
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

Run against the local catalog:

```bash
SMOLLM_CATALOG="$LOCAL_CATALOG" \
    ./run.sh --params='matrix: {ask.endpoint: [qwen-05]}'
```

or compare aliases from the same catalog:

```bash
SMOLLM_CATALOG="$LOCAL_CATALOG" \
    ./run.sh --params='matrix: {ask.endpoint: [smol-135, qwen-05]}'
```

The example client uses the OpenAI chat-completions API, so custom endpoints
must provide a compatible chat interface.

## Containers

`run.sh` builds the adjacent `Dockerfile` unless `--no-container` is given.
The build context is this example directory, and the image copies only
`__init__.py` and `cli/`. Cards, catalogs, documentation, run outputs, and the
rest of the repository do not enter the image.

The Dockerfile installs MAGNET from the source artifact pinned by
`MAGNET_INSTALL`. The image contains neither model weights nor infer-stack;
infer-stack serves endpoints on the host.

Node working directories are bind-mounted at the same absolute paths they have
on the host so kwdagger's absolute artifact paths remain valid inside the
container.

## Real and simulated catalogs

`catalog.yaml` and `catalog-mock.yaml` define the same endpoint aliases.
`run.sh` uses `catalog.yaml`; `--mock` selects `catalog-mock.yaml`.

The mock catalog uses infer-stack's simulator and requires no GPU or model
weights. Simulator outputs are synthetic, so model-quality metrics such as
`exact_rate` and `agreement` should not be interpreted as model results.

## Smoke tests

### Developer smoke test

`test.sh` runs all four real/mock x container/host combinations:

```bash
./test.sh
```

It requires an NVIDIA GPU, Docker, MAGNET with the `leasing` extra, and an
initialized infer-stack backend. Each variant uses a fresh output root under
`runs/dev-smoke-<timestamp>/` by default; override it with `SMOLLM_TEST_RUNS`.

The test refuses to start while active infer-stack leases exist. On a dedicated
machine, clear active leases and idle deployments before the run with:

```bash
./test.sh --release
```

A lease leak stops later variants. Other failures are recorded and the remaining
variants continue. The final summary reports all four variants and their
artifact paths.

### CI smoke test

GitHub Actions runs:

```bash
./run.sh --mock
```

on `ubuntu-latest`. This exercises the containerized node path, kwdagger,
per-node leasing, infer-stack's simulator, the OpenAI-compatible gateway, and
the final claim without requiring a GPU.

## Files

| File | Purpose |
|---|---|
| `run.sh` | Run the example |
| `test.sh` | Developer smoke test for all four execution modes |
| `Dockerfile` | Container image for node commands |
| `smollm_kwdagger.yaml` | Evaluation card and endpoint matrix |
| `pipeline.py` | Three-node DAG and leasing declaration |
| `cli/make_items.py` | Generate the input questions |
| `cli/ask_model.py` | Query one OpenAI-compatible endpoint |
| `cli/compare_answers.py` | Gather answers and compute metrics |
| `catalog.yaml` | Real SmolLM endpoint definitions |
| `catalog-mock.yaml` | Simulator definitions for the same aliases |
| `catalog.local*.yaml` | Ignored local catalogs for custom endpoints |
