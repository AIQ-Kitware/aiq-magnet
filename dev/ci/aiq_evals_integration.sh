#!/usr/bin/env bash
# MAGNET <-> aiq-magnet-evals integration (aiq-magnet-evals integration plan M10).
#
# Installs MAGNET (with HELM) and aiq-magnet-evals from a pinned checkout,
# builds Inspect and OLMo Eval workers at their verified pins, an Inspect worker
# with `openai`, and a portable MAGNET environment for the Docker test, and runs
# the integration tests with MAGNET_REQUIRE_AIQ_EVALS=1: a missing engine,
# worker, tmux, Docker, or infer-stack fails the job instead of skipping it.
# Real infer-stack leases use its null serving backend (no GPU).
#
# Usage (repository root): dev/ci/aiq_evals_integration.sh [WORK_DIR]
#   AIQ_MAGNET_EVALS_REPO / AIQ_MAGNET_EVALS_REV select the evaluator revision;
#   AIQ_MAGNET_EVALS_DIR uses an existing checkout instead of cloning.
set -euo pipefail
WORK=${1:-${RUNNER_TEMP:-/tmp}/magnet-aiq-evals}
EVALS_REPO=${AIQ_MAGNET_EVALS_REPO:-https://github.com/Erotemic/aiq-magnet-evals.git}
OLMO_REPO=${OLMO_REPO:-https://github.com/allenai/olmo-eval.git}
OLMO_REV=73ade80e24f796af55caeb8fd7b75a7f3fd607fd
mkdir -p "$WORK"

retry() {  # bounded retries: installation is the external step
  for attempt in 1 2 3; do
    "$@" && return 0
    [ "$attempt" = 3 ] && { echo "external: failed 3 times: $*" >&2; return 1; }
    sleep $((attempt * 10))
  done
}

EVALS=${AIQ_MAGNET_EVALS_DIR:-$WORK/aiq-magnet-evals}
if [ -z "${AIQ_MAGNET_EVALS_DIR:-}" ]; then
  REV=${AIQ_MAGNET_EVALS_REV:?set AIQ_MAGNET_EVALS_REV to the aiq-magnet-evals revision under test}
  [ -d "$EVALS/.git" ] || retry git clone -q "$EVALS_REPO" "$EVALS"
  git -C "$EVALS" checkout -q "$REV"
fi

# MAGNET, its HELM extra, and aiq-magnet-evals; this is also the HELM worker.
uv venv -q --python 3.12 "$WORK/magnet"
retry uv pip install -q --python "$WORK/magnet/bin/python" \
  -c "$EVALS/dev/environments/phase1/helm-py312-constraints.txt" \
  -e "$EVALS" -e ./packages/aiq-magnet-theory -e '.[tests,helm,leasing]' 'crfm-helm==0.5.14'

# Inspect worker at the verified pin.
uv venv -q --python 3.11 "$WORK/inspect"
retry uv pip install -q --python "$WORK/inspect/bin/python" \
  -c "$EVALS/dev/environments/phase1/inspect-py311-constraints.txt" -e "$EVALS[inspect]"

# Inspect with `openai` (outside the verified pin set): the real-lease test
# drives Inspect's openai provider for a primary and a separately leased grader.
uv venv -q --python 3.11 "$WORK/inspect-openai"
retry uv pip install -q --python "$WORK/inspect-openai/bin/python" \
  -c "$EVALS/dev/environments/phase1/inspect-py311-constraints.txt" -e "$EVALS[inspect]" openai

# OLMo Eval worker: isolated checkout synced from its upstream lock.
OLMO=$WORK/olmo-eval
[ -d "$OLMO/.git" ] || retry git clone -q "$OLMO_REPO" "$OLMO"
git -C "$OLMO" checkout -q "$OLMO_REV"
(cd "$OLMO" && retry uv sync -q --frozen --no-default-groups --extra litellm --extra agents --python 3.12)

# A MAGNET environment on a uv-managed Python, usable inside ubuntu:24.04.
uv venv -q --python 3.13 --python-preference only-managed "$WORK/magnet-container"
retry uv pip install -q --python "$WORK/magnet-container/bin/python" \
  -e "$EVALS" -e ./packages/aiq-magnet-theory -e '.[tests]'
retry docker pull -q ubuntu:24.04
retry docker build -q --build-arg BASE_IMAGE=ubuntu:24.04 \
  --iidfile "$WORK/worker-container.iid" \
  -f "$EVALS/dev/environments/worker-container.Dockerfile" "$EVALS/dev/environments"

MAGNET_REQUIRE_AIQ_EVALS=1 \
MAGNET_TEST_DOCKER=1 \
MAGNET_TEST_CONTAINER_VENV="$WORK/magnet-container" \
MAGNET_TEST_CONTAINER_IMAGE="$(cat "$WORK/worker-container.iid")" \
AIQ_EVALS_REPO="$EVALS" \
AIQ_EVALS_HELM_PYTHON="$WORK/magnet/bin/python" \
AIQ_EVALS_INSPECT_PYTHON="$WORK/inspect/bin/python" \
AIQ_EVALS_INSPECT_OPENAI_PYTHON="$WORK/inspect-openai/bin/python" \
AIQ_EVALS_OLMO_PYTHON="$OLMO/.venv/bin/python" \
PATH="$WORK/magnet/bin:$PATH" \
  "$WORK/magnet/bin/python" -m pytest -q -p no:cacheprovider \
    tests/test_aiq_evals_integration.py tests/test_aiq_evals_examples.py \
    tests/test_aiq_evals_container.py tests/test_aiq_evals_lease.py
