#!/usr/bin/env bash
# Full real-hardware MAGNET <-> aiq-magnet-evals acceptance.
#
# This is intentionally opt-in.  Unlike aiq_evals_integration.sh, this runner
# acquires a real infer-stack Compose endpoint and performs actual vLLM/GPU
# inference from a Dockerized MAGNET EvaluationNode.
#
# Usage from the aiq-magnet repository root:
#
#   AIQ_MAGNET_EVALS_DIR=~/code/aiq-magnet-evals \
#   INFER_STACK_DIR=~/code/infer_stack \
#   dev/ci/aiq_evals_real_gpu.sh [WORK_DIR]
#
# Optional:
#   MAGNET_REAL_GPU_ENDPOINT=qwen2.5-7b
#   MAGNET_REAL_GPU_MODEL_REVISION=Qwen/Qwen2.5-7B-Instruct
#   MAGNET_REAL_GPU_ALLOWED_GPUS=0
#   MAGNET_TEST_CONTAINER_IMAGE=ubuntu:24.04
#
# The current infer-stack control plane must use backend=compose and its
# catalog must already contain MAGNET_REAL_GPU_ENDPOINT.  The runner does not
# rewrite the user's catalog or move the host to a separate controller.
#
# The script is a child process: failures exit this script, not the caller's
# interactive shell.  Deliberately no `set -o pipefail` is used.
set -eu

WORK=${1:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/magnet-aiq-evals-real-gpu}
EVALS_REPO=${AIQ_MAGNET_EVALS_REPO:-https://github.com/Erotemic/aiq-magnet-evals.git}
ENDPOINT=${MAGNET_REAL_GPU_ENDPOINT:-qwen2.5-7b}
MODEL_REVISION=${MAGNET_REAL_GPU_MODEL_REVISION:-Qwen/Qwen2.5-7B-Instruct}
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

retry docker pull -q "$CONTAINER_IMAGE"

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

CATALOG=$($INFER_STACK_BIN catalog show 2>&1) || {
    printf '%s\n' "$CATALOG" >&2
    echo 'error: infer-stack catalog show failed' >&2
    exit 2
}
if ! printf '%s\n' "$CATALOG" | grep -Eq "^[[:space:]]*${ENDPOINT//./\\.}:[[:space:]]*$"; then
    cat >&2 <<EOF
error: infer-stack catalog does not contain endpoint '$ENDPOINT'.

Either choose an existing real vLLM endpoint:
    MAGNET_REAL_GPU_ENDPOINT=<alias> dev/ci/aiq_evals_real_gpu.sh "$WORK"

or seed/add one first, for example:
    infer-stack catalog suggest --apply
EOF
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

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$WORK/runs/$STAMP
mkdir -p "$OUT"

OLD_PYTHONPATH=${PYTHONPATH:-}
if [ -n "$OLD_PYTHONPATH" ]; then
    TEST_PYTHONPATH="$PWD/tests:$PWD:$EVALS:$OLD_PYTHONPATH"
else
    TEST_PYTHONPATH="$PWD/tests:$PWD:$EVALS"
fi

# A dummy key is enough for the local OpenAI-compatible vLLM endpoint; the
# important part is that the secret is present for the evaluator's declared
# requirement.  Do not export it back into the caller's shell.
TEST_OPENAI_API_KEY=${OPENAI_API_KEY:-aiq-real-gpu-local-key}

echo
cat <<EOF
=== real-GPU aiq-magnet-evals acceptance ===
work:          $WORK
artifacts:     $OUT
endpoint:      $ENDPOINT
model revision:$MODEL_REVISION
allowed GPUs:  $ALLOWED_GPUS
python:        3.13
container:     $CONTAINER_IMAGE
EOF

set +e
MAGNET_TEST_REAL_GPU=1 \
MAGNET_REQUIRE_AIQ_EVALS=1 \
MAGNET_REAL_GPU_ENDPOINT="$ENDPOINT" \
MAGNET_REAL_GPU_MODEL_REVISION="$MODEL_REVISION" \
MAGNET_REAL_GPU_OUTPUT_DIR="$OUT" \
MAGNET_TEST_CONTAINER_VENV="$CONTAINER_ENV" \
MAGNET_TEST_CONTAINER_IMAGE="$CONTAINER_IMAGE" \
AIQ_EVALS_REPO="$EVALS" \
AIQ_EVALS_INSPECT_OPENAI_PYTHON="$INSPECT_ENV/bin/python" \
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
exit "$status"
