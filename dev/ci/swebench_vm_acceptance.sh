#!/usr/bin/env bash
# Real Harbor EvaluationNodes with deterministic HTTP and null serving leases.
set -euo pipefail
cd "$(dirname "$0")/../.."
MAGNET_ROOT=$PWD
: "${AIQ_MAGNET_EVALS_DIR:?Set AIQ_MAGNET_EVALS_DIR to the evaluator checkout}"
: "${INFER_STACK_DIR:?Set INFER_STACK_DIR to the serving checkout}"
WORK=${AIQ_HARBOR_WORK:-/tmp/aiq-harbor-roadmap}
mkdir -p "$WORK"
RUN=$(mktemp -d "$WORK/magnet-vm-XXXXXXXX")
MAGNET_ENV=${MAGNET_ENV:-$RUN/magnet}
HARBOR_WORKER=${MAGNET_HARBOR_WORKER:-$RUN/harbor}
uv venv -q --python 3.13 --python-preference only-managed "$MAGNET_ENV"
uv pip install -q --python "$MAGNET_ENV/bin/python" \
    -e "$MAGNET_ROOT/packages/aiq-magnet-theory" -e "$MAGNET_ROOT[tests]" \
    -e "$AIQ_MAGNET_EVALS_DIR" -e "$INFER_STACK_DIR"
uv venv -q --python 3.12 --python-preference only-managed "$HARBOR_WORKER"
uv pip install -q --python "$HARBOR_WORKER/bin/python" \
    -c "$AIQ_MAGNET_EVALS_DIR/dev/environments/harbor-py312-constraints.txt" \
    -e "$AIQ_MAGNET_EVALS_DIR[harbor,tests]"
docker image inspect ubuntu:24.04 >/dev/null 2>&1 || docker pull ubuntu:24.04
export PATH="$MAGNET_ENV/bin:$PATH"
export MAGNET_REQUIRE_AIQ_EVALS=1 MAGNET_TEST_DOCKER=1 HARBOR_TELEMETRY=0
export AIQ_EVALS_REPO="$AIQ_MAGNET_EVALS_DIR" AIQ_EVALS_HARBOR_PYTHON="$HARBOR_WORKER/bin/python"
"$MAGNET_ENV/bin/python" -m pytest -q tests/test_aiq_evals_swebench.py \
    tests/test_aiq_evals_examples.py -k 'not test_helm_generation_recipe and not test_inspect_recipes and not test_olmo_generation_recipe and not test_olmo_agent_recipe and not test_mixed_engine_comparison_recipe'
if [ -n "${AIQ_VM_MAGNET_CHECKS:-}" ]; then
    "$MAGNET_ENV/bin/python" - "$AIQ_VM_MAGNET_CHECKS" <<'PY'
import json
import sys
from pathlib import Path
module = 'tests.test_aiq_evals_swebench.'
Path(sys.argv[1]).write_text(json.dumps({
    'magnet-fake-lease-integration': [module + name for name in (
        'test_harbor_scheduler_lease_reuse_and_evidence[False]',
        'test_harbor_scheduler_lease_reuse_and_evidence[True]',
        'test_stale_serving_facts_abort_before_inference[catalog]',
        'test_stale_serving_facts_abort_before_inference[lease]',
        'test_concurrent_selectors_share_one_native_execution',
        'test_infrastructure_failure_produces_no_false_score[True]',
        'test_native_pro_replay_failure_is_ineligible',
        'test_cancelled_native_lease_releases_and_cleans_sandbox')],
    'quantization-measurement-invalidation': [module + name for name in (
        'test_quantization_invalidates_same_alias_measurement',
        'test_uncertain_serving_inputs_cannot_become_reusable')],
}, indent=2) + '\n')
PY
fi
