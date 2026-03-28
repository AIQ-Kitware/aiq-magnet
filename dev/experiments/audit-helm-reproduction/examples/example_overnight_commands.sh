#!/usr/bin/env bash
set -euo pipefail

# Example overnight control on a richer generation-style task.

dev/experiments/audit-helm-reproduction/scripts/make_repeat_pair_manifests.sh \
  'narrative_qa:model=eleutherai/pythia-6.9b,data_augmentation=canonical' \
  audit-narrative-pythia \
  dev/experiments/audit-helm-reproduction/configs/generated \
  --max-eval-instances 1000 \
  --devices 0 \
  --tmux-workers 1

dev/experiments/audit-helm-reproduction/scripts/run_from_manifest.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/audit-narrative-pythia_r1.yaml

dev/experiments/audit-helm-reproduction/scripts/run_from_manifest.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/audit-narrative-pythia_r2.yaml

dev/experiments/audit-helm-reproduction/scripts/compare_batch.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/audit-narrative-pythia_r1.yaml

dev/experiments/audit-helm-reproduction/scripts/compare_batch.sh \
  dev/experiments/audit-helm-reproduction/configs/generated/audit-narrative-pythia_r2.yaml
