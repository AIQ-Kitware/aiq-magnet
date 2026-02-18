#!/usr/bin/env bash

# Copy/paste friendly: set SCRIPT_DIR to this folder (edit if you run elsewhere).
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="./llama_consistency"
fi
cd "$SCRIPT_DIR"
echo "$SCRIPT_DIR"
# Set PYTHONPATH to ensure Python can see the example directory.
export PYTHONPATH=.

EVAL_DPATH=${EVAL_DPATH:-$PWD/results}
echo "EVAL_DPATH = $EVAL_DPATH"
kwdagger schedule \
    --params="
        pipeline: 'llama_consistency.pipelines.llama_pipeline()'
        matrix:
            llama_predict.base_model:
                - meta/llama-2-13b
                - meta/llama-2-70b
                - meta/llama-2-7b
                - meta/llama-3-70b
                - meta/llama-3-8b
                - meta/llama-65b
            llama_predict.comp_model:
                - meta/llama-2-13b
                - meta/llama-2-70b
                - meta/llama-2-7b
                - meta/llama-3-70b
                - meta/llama-3-8b
                - meta/llama-65b
    " \
    --root_dpath="${EVAL_DPATH}" \
    --tmux_workers=2 \
    --backend=serial --skip_existing=1 \
    --run=1