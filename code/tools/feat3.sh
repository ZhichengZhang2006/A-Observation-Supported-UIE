#!/usr/bin/env bash
# H2 v3 feature pipeline.
# Uses ONLY splits the frozen checkpoints never trained on (the test splits) plus
# the 7-method ensemble spread as the ordinary epistemic-uncertainty comparator.
# No new inference is run: all seven methods' test predictions already exist.
set -u
cd /home/user/obsui || exit 1
PY=/usr/local/iCompute/envs/Mamba/bin/python
LOG=logs/features_v3c.log
: > "$LOG"

for D in LSUI UIEB; do
  for B in phaseformer aquaclarity fdcenet; do
    echo "=== features $D test $B ===" >> "$LOG"
    "$PY" code/tools/extract_features_v3.py --dataset "$D" --split test \
          --backbone "$B" --workers 24 >> "$LOG" 2>&1
  done
  echo "=== disagreement $D test ===" >> "$LOG"
  "$PY" code/tools/ensemble_disagreement_v3.py --dataset "$D" --split test >> "$LOG" 2>&1
done
echo "=== FEATURES DONE ===" >> "$LOG"
