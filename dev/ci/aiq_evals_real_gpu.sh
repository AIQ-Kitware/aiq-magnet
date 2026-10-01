#!/usr/bin/env bash
# Full real-hardware MAGNET <-> aiq-magnet-evals acceptance.
#
# This opt-in gate runs three evaluation modes through real infer-stack
# Compose/vLLM/GPU serving:
#   1. Inspect one-shot generation
#   2. Inspect forced tool-use / agentic evaluation
#   3. OLMo Eval OpenAI-Agents tool-use evaluation
#
# Usage from the aiq-magnet repository root:
#
#   AIQ_MAGNET_EVALS_DIR=~/code/aiq-magnet-evals \
#   INFER_STACK_DIR=~/code/infer_stack \
#   MAGNET_REAL_GPU_ALLOWED_GPUS=0 \
#   dev/ci/aiq_evals_real_gpu.sh [WORK_DIR]
#
# Optional:
#   MAGNET_REAL_GPU_ENDPOINT=qwen2.5-7b
#   MAGNET_REAL_GPU_MODEL_REVISION=Qwen/Qwen2.5-7B-Instruct
#   MAGNET_REAL_GPU_AGENTIC_ENDPOINT=qwen2.5-7b-agentic-e2e
#   MAGNET_REAL_GPU_TOOL_CALL_PARSER=hermes
#   MAGNET_REAL_GPU_ALLOWED_GPUS=0
#   MAGNET_TEST_CONTAINER_IMAGE=ubuntu:24.04
#
# The runner temporarily adds the agentic endpoint alias to the selected real
# infer-stack catalog when it is absent.  The alias copies the base endpoint,
# appends vLLM auto-tool-choice flags, uses reclaim=stop, and is removed by
# restoring the original catalog byte-for-byte on exit.  To avoid mutating a
# live controller configuration underneath unrelated work, this automatic
# addition is refused while another infer-stack lease is active.
#
# The script is a child process: failures exit this script, not the caller's
# interactive shell.  Deliberately no `set -o pipefail` is used.
set -eu

WORK=${1:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/magnet-aiq-evals-real-gpu}
EVALS_REPO=${AIQ_MAGNET_EVALS_REPO:-https://github.com/Erotemic/aiq-magnet-evals.git}
OLMO_REPO=${OLMO_REPO:-https://github.com/allenai/olmo-eval.git}
OLMO_REVISION=73ade80e24f796af55caeb8fd7b75a7f3fd607fd
ENDPOINT=${MAGNET_REAL_GPU_ENDPOINT:-qwen2.5-7b}
MODEL_REVISION=${MAGNET_REAL_GPU_MODEL_REVISION:-Qwen/Qwen2.5-7B-Instruct}
AGENTIC_ENDPOINT=${MAGNET_REAL_GPU_AGENTIC_ENDPOINT:-${ENDPOINT}-agentic-e2e}
TOOL_CALL_PARSER=${MAGNET_REAL_GPU_TOOL_CALL_PARSER:-hermes}
ALLOWED_GPUS=${MAGNET_REAL_GPU_ALLOWED_GPUS:-${SLURM_JOB_GPUS:-0}}
CONTAINER_IMAGE=${MAGNET_TEST_CONTAINER_IMAGE:-ubuntu:24.04}

mkdir -p "$WORK"

for command in uv docker nvidia-smi git; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "error: required command is not on PATH: $command" >&2
        exit 2
    fi
done
if [ -z "${INFER_STACK_DIR:-}" ] && ! command -v infer-stack >/dev/null 2>&1; then
    echo 'error: infer-stack is not on PATH and INFER_STACK_DIR is not set' >&2
    exit 2
fi
if ! docker info >/dev/null 2>&1; then
    echo 'error: Docker daemon is not reachable' >&2
    exit 2
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
    echo 'error: NVIDIA GPU is not visible to nvidia-smi' >&2
    exit 2
fi

retry() {
    for attempt in 1 2 3; do
        if "$@"; then
            return 0
        fi
        if [ "$attempt" = 3 ]; then
            echo "external setup failed after 3 attempts: $*" >&2
            return 1
        fi
        sleep $((attempt * 10))
    done
}

EVALS=${AIQ_MAGNET_EVALS_DIR:-$WORK/aiq-magnet-evals}
if [ -z "${AIQ_MAGNET_EVALS_DIR:-}" ]; then
    REV=${AIQ_MAGNET_EVALS_REV:?set AIQ_MAGNET_EVALS_REV, or set AIQ_MAGNET_EVALS_DIR to a local checkout}
    if [ ! -d "$EVALS/.git" ]; then
        retry git clone -q "$EVALS_REPO" "$EVALS"
    fi
    git -C "$EVALS" checkout -q "$REV"
fi

if [ ! -f "$EVALS/pyproject.toml" ]; then
    echo "error: not an aiq-magnet-evals checkout: $EVALS" >&2
    exit 2
fi
if [ ! -f pyproject.toml ] || [ ! -d magnet ]; then
    echo 'error: run this script from the aiq-magnet repository root' >&2
    exit 2
fi

MAGNET_ENV=$WORK/magnet-py313
INSPECT_ENV=$WORK/inspect-openai-py313
CONTAINER_ENV=$WORK/magnet-container-py313
OLMO=$WORK/olmo-eval-py313

uv venv -q --python 3.13 "$MAGNET_ENV"
MAGNET_INSTALL=(-e "$EVALS" -e ./packages/aiq-magnet-theory -e '.[tests,leasing]')
if [ -n "${INFER_STACK_DIR:-}" ]; then
    MAGNET_INSTALL+=(-e "$INFER_STACK_DIR")
fi
retry uv pip install -q --python "$MAGNET_ENV/bin/python" "${MAGNET_INSTALL[@]}"

uv venv -q --python 3.13 "$INSPECT_ENV"
retry uv pip install -q --python "$INSPECT_ENV/bin/python" -e "$EVALS[inspect]" openai

uv venv -q --python 3.13 --python-preference only-managed "$CONTAINER_ENV"
retry uv pip install -q --python "$CONTAINER_ENV/bin/python" \
    -e "$EVALS" -e ./packages/aiq-magnet-theory -e '.[tests]'

if [ ! -d "$OLMO/.git" ]; then
    retry git clone -q "$OLMO_REPO" "$OLMO"
fi
git -C "$OLMO" checkout -q "$OLMO_REVISION"
if [ -n "$(git -C "$OLMO" status --porcelain --untracked-files=no)" ]; then
    echo "error: cached OLMo checkout has tracked modifications: $OLMO" >&2
    exit 2
fi
olmo_sync() {
    (cd "$OLMO" && uv sync -q --frozen --no-default-groups \
        --extra litellm --extra agents --python 3.13)
}
retry olmo_sync
retry uv pip install -q --python "$OLMO/.venv/bin/python" kwconf

retry docker pull -q "$CONTAINER_IMAGE"
# Editable OLMo installs need git in the executing container, not just on the
# host. Build the evaluator-owned provenance image from the selected base.
retry docker build -q --build-arg "BASE_IMAGE=$CONTAINER_IMAGE" \
    --iidfile "$WORK/worker-container.iid" \
    -f "$EVALS/dev/environments/worker-container.Dockerfile" "$EVALS/dev/environments"
CONTAINER_IMAGE=$(cat "$WORK/worker-container.iid")

INFER_STACK_BIN=$MAGNET_ENV/bin/infer-stack
if [ ! -x "$INFER_STACK_BIN" ]; then
    INFER_STACK_BIN=$(command -v infer-stack || true)
fi
if [ -z "$INFER_STACK_BIN" ] || [ ! -x "$INFER_STACK_BIN" ]; then
    echo 'error: could not resolve the infer-stack executable under test' >&2
    exit 2
fi

STATUS=$($INFER_STACK_BIN status 2>&1) || {
    printf '%s\n' "$STATUS" >&2
    echo 'error: infer-stack status failed' >&2
    exit 2
}
printf '%s\n' "$STATUS"

BACKEND=$(printf '%s\n' "$STATUS" | sed -n 's/^[[:space:]]*backend:[[:space:]]*//p' | head -n 1)
if [ "$BACKEND" != compose ]; then
    echo "error: real-GPU acceptance requires infer-stack backend=compose; got: ${BACKEND:-<unknown>}" >&2
    exit 2
fi
CATALOG_PATH=$(printf '%s\n' "$STATUS" | sed -n 's/^[[:space:]]*catalog:[[:space:]]*\([^[:space:]]*\).*/\1/p' | head -n 1)
if [ -z "$CATALOG_PATH" ] || [ ! -f "$CATALOG_PATH" ]; then
    echo "error: could not resolve writable catalog path from infer-stack status: ${CATALOG_PATH:-<unknown>}" >&2
    exit 2
fi

CATALOG=$($INFER_STACK_BIN catalog show 2>&1) || {
    printf '%s\n' "$CATALOG" >&2
    echo 'error: infer-stack catalog show failed' >&2
    exit 2
}
if ! printf '%s\n' "$CATALOG" | grep -Eq "^[[:space:]]*${ENDPOINT//./\\.}:[[:space:]]*$"; then
    cat >&2 <<EOF
error: infer-stack catalog does not contain base endpoint '$ENDPOINT'.

Either choose an existing real vLLM endpoint:
    MAGNET_REAL_GPU_ENDPOINT=<alias> dev/ci/aiq_evals_real_gpu.sh "$WORK"

or seed/add one first, for example:
    infer-stack catalog suggest --apply
EOF
    exit 2
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$WORK/runs/$STAMP
mkdir -p "$OUT"

# The agentic path needs vLLM automatic tool parsing.  If the caller has not
# supplied a preconfigured agentic endpoint, create a temporary alias by copying
# the real base endpoint and adding the two vLLM flags.  extra_args are used so
# infer-stack treats this as a distinct deployment identity rather than
# coalescing it with a non-tool-enabled resident process.
CATALOG_BACKUP=$OUT/catalog.before.yaml
CATALOG_MUTATED=0
restore_catalog() {
    if [ "$CATALOG_MUTATED" = 1 ] && [ -f "$CATALOG_BACKUP" ]; then
        cp -p "$CATALOG_BACKUP" "$CATALOG_PATH"
        echo "restored infer-stack catalog: $CATALOG_PATH"
    fi
}
trap restore_catalog EXIT

if ! printf '%s\n' "$CATALOG" | grep -Eq "^[[:space:]]*${AGENTIC_ENDPOINT//./\\.}:[[:space:]]*$"; then
    ACTIVE_LEASES=$(printf '%s\n' "$STATUS" | sed -n 's/^leasing:[[:space:]]*\([0-9][0-9]*\)[[:space:]]*active.*/\1/p' | head -n 1)
    if [ -z "$ACTIVE_LEASES" ] || [ "$ACTIVE_LEASES" != 0 ]; then
        echo "error: refusing temporary catalog mutation with ${ACTIVE_LEASES:-unknown} active infer-stack lease(s)" >&2
        echo "pre-create '$AGENTIC_ENDPOINT' yourself or rerun when the controller is quiescent" >&2
        exit 2
    fi
    cp -p "$CATALOG_PATH" "$CATALOG_BACKUP"
    CATALOG_MUTATED=1
    CATALOG_PATH="$CATALOG_PATH" \
    BASE_ENDPOINT="$ENDPOINT" \
    AGENTIC_ENDPOINT="$AGENTIC_ENDPOINT" \
    TOOL_CALL_PARSER="$TOOL_CALL_PARSER" \
        "$MAGNET_ENV/bin/python" - <<'PY'
import copy
import os
from pathlib import Path

import yaml

path = Path(os.environ['CATALOG_PATH'])
data = yaml.safe_load(path.read_text()) or {}
endpoints = data.setdefault('endpoints', {})
base_name = os.environ['BASE_ENDPOINT']
agent_name = os.environ['AGENTIC_ENDPOINT']
parser = os.environ['TOOL_CALL_PARSER']
if base_name not in endpoints:
    raise SystemExit(f'base endpoint not found while preparing agent alias: {base_name}')
if agent_name in endpoints:
    raise SystemExit(f'agent endpoint appeared concurrently: {agent_name}')
agent = copy.deepcopy(endpoints[base_name])
agent['served_name'] = agent_name
runtime = agent.setdefault('runtime', {})
extra = [str(arg) for arg in (runtime.get('extra_args') or [])]
for flag in ('--enable-auto-tool-choice', f'--tool-call-parser={parser}'):
    if flag not in extra:
        extra.append(flag)
runtime['extra_args'] = extra
agent['reclaim'] = {'policy': 'stop'}
endpoints[agent_name] = agent

tmp = path.with_name(path.name + '.aiq-real-gpu.tmp')
tmp.write_text(yaml.safe_dump(data, sort_keys=False))
tmp.replace(path)
print(f'created temporary agentic endpoint: {agent_name}')
print(f'  base: {base_name}')
print(f'  tool parser: {parser}')
print(f'  extra_args: {extra}')
PY
else
    echo "using existing agentic endpoint: $AGENTIC_ENDPOINT"
fi

# Validate that an existing or newly created agentic endpoint really has the
# tool-serving flags.  This is intentionally checked from the catalog bytes,
# not inferred from the endpoint's name.
CATALOG_PATH="$CATALOG_PATH" \
AGENTIC_ENDPOINT="$AGENTIC_ENDPOINT" \
TOOL_CALL_PARSER="$TOOL_CALL_PARSER" \
    "$MAGNET_ENV/bin/python" - <<'PY'
import os
from pathlib import Path

import yaml

path = Path(os.environ['CATALOG_PATH'])
data = yaml.safe_load(path.read_text()) or {}
name = os.environ['AGENTIC_ENDPOINT']
parser = os.environ['TOOL_CALL_PARSER']
endpoint = (data.get('endpoints') or {}).get(name)
if endpoint is None:
    raise SystemExit(f'agentic endpoint is absent after preparation: {name}')
extra = [str(arg) for arg in ((endpoint.get('runtime') or {}).get('extra_args') or [])]
required = {'--enable-auto-tool-choice', f'--tool-call-parser={parser}'}
missing = sorted(required.difference(extra))
if missing:
    raise SystemExit(
        f'agentic endpoint {name!r} is missing vLLM tool flags {missing}; '
        f'runtime.extra_args={extra}'
    )
PY

CATALOG=$($INFER_STACK_BIN catalog show 2>&1) || {
    printf '%s\n' "$CATALOG" >&2
    echo 'error: infer-stack rejected the prepared catalog' >&2
    exit 2
}
if ! printf '%s\n' "$CATALOG" | grep -Eq "^[[:space:]]*${AGENTIC_ENDPOINT//./\\.}:[[:space:]]*$"; then
    echo "error: infer-stack did not expose prepared agentic endpoint '$AGENTIC_ENDPOINT'" >&2
    exit 2
fi

# Remove stale lease-derived process state from the child environment.  Keep
# INFER_STACK_CONFIG_DIR / INFER_STACK_DATA_DIR if the caller intentionally
# selected a control plane; the test reports exactly which one it sees.
unset INFER_STACK_BACKEND || true
unset INFER_STACK_CATALOG || true
unset INFER_STACK_LEASE_ID || true
unset OPENAI_BASE_URL || true
for name in $(env | sed -n 's/^\(INFER_STACK_ENDPOINT_[A-Za-z0-9_]*\)=.*/\1/p'); do
    unset "$name"
done

OLD_PYTHONPATH=${PYTHONPATH:-}
if [ -n "$OLD_PYTHONPATH" ]; then
    TEST_PYTHONPATH="$PWD/tests:$PWD:$EVALS:$OLD_PYTHONPATH"
else
    TEST_PYTHONPATH="$PWD/tests:$PWD:$EVALS"
fi

# A dummy key is enough for the local OpenAI-compatible vLLM endpoint; the
# important part is that the secret is present for each evaluator's declared
# requirement.  Do not export it back into the caller's shell.
TEST_OPENAI_API_KEY=${OPENAI_API_KEY:-aiq-real-gpu-local-key}

echo
cat <<EOF
=== real-GPU aiq-magnet-evals acceptance ===
work:             $WORK
artifacts:        $OUT
base endpoint:    $ENDPOINT
agentic endpoint: $AGENTIC_ENDPOINT
model revision:   $MODEL_REVISION
tool parser:      $TOOL_CALL_PARSER
OLMo revision:    $OLMO_REVISION
allowed GPUs:     $ALLOWED_GPUS
python:           3.13
container:        $CONTAINER_IMAGE
EOF

set +e
MAGNET_TEST_REAL_GPU=1 \
MAGNET_REQUIRE_AIQ_EVALS=1 \
MAGNET_REAL_GPU_ENDPOINT="$ENDPOINT" \
MAGNET_REAL_GPU_AGENTIC_ENDPOINT="$AGENTIC_ENDPOINT" \
MAGNET_REAL_GPU_MODEL_REVISION="$MODEL_REVISION" \
MAGNET_REAL_GPU_OUTPUT_DIR="$OUT" \
MAGNET_TEST_CONTAINER_VENV="$CONTAINER_ENV" \
MAGNET_TEST_CONTAINER_IMAGE="$CONTAINER_IMAGE" \
AIQ_EVALS_REPO="$EVALS" \
AIQ_EVALS_INSPECT_OPENAI_PYTHON="$INSPECT_ENV/bin/python" \
AIQ_EVALS_OLMO_PYTHON="$OLMO/.venv/bin/python" \
OPENAI_API_KEY="$TEST_OPENAI_API_KEY" \
SLURM_JOB_GPUS="$ALLOWED_GPUS" \
PATH="$MAGNET_ENV/bin:$PATH" \
PYTHONPATH="$TEST_PYTHONPATH" \
    "$MAGNET_ENV/bin/python" -m pytest -q -s -p no:cacheprovider \
        tests/test_aiq_evals_real_gpu.py \
        2>&1 | tee "$OUT/pytest.log"
status=${PIPESTATUS[0]}
set -e

echo "real-GPU acceptance exit status: $status"
echo "artifacts: $OUT"
if [ "$status" = 0 ]; then
    echo "=== real-GPU aiq-magnet-evals acceptance: VERIFIED ==="
fi
exit "$status"
