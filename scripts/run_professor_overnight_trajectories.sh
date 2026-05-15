#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p scenario_outputs/overnight_logs

echo "Starting professor overnight trajectory runs at $(date)"
echo "Repository: $ROOT"
echo

echo "=== Scenario 4: 202004, 4 seeds, 20000 steps ==="
python scripts/scenario4.py \
  --date 202004 \
  --n-seeds 4 \
  --n-steps 20000 \
  --base-seed 20260424 \
  --reg-mode l2 \
  --constraint-mode box_barrier \
  --contrast-function sc_beats_ww \
  --l2reg 0.3 \
  --eta 0.001 \
  --beta 10 \
  --random-start-scale 0.005 \
  --save-trajectory-tensors \
  --trajectory-format both \
  --trajectory-burn-in-frac 0.5 \
  --trajectory-thin 1

echo
echo "=== E2E diversification: 202004, 4 seeds, 20000 steps ==="
python scripts/scenario5.py \
  --date 202004 \
  --probe decision_fragility \
  --model locked_e2e \
  --objective entropy_max \
  --reg-mode l2 \
  --constraint-mode box_barrier \
  --l2reg 0.3 \
  --eta 0.09 \
  --beta 10 \
  --random-start-scale 0.0 \
  --n-seeds 4 \
  --n-steps 20000 \
  --base-seed 20260424 \
  --save-trajectory-tensors \
  --trajectory-format both \
  --trajectory-burn-in-frac 0.5 \
  --trajectory-thin 1

echo
echo "Finished professor overnight trajectory runs at $(date)"
