#!/usr/bin/env bash
# H2 v3 data pipeline: unseen-split predictions -> per-image features -> ensemble disagreement.
# Every split used here is one the frozen checkpoints were NOT trained on.
# UIEB train was deliberately skipped: the UIEB checkpoints score ~42 dB on it
# (memorised), so it cannot train an error model.
set -u
cd /home/user/obsui || exit 1
PY=/usr/local/iCompute/envs/Mamba/bin/python
LOG=logs/pipeline_v3.log
: > "$LOG"

for D in UIEB LSUI; do
  for M in phaseformer aquaclarity; do
    echo "=== baseline $M $D val ===" >> "$LOG"
    "$PY" code/tools/run_baseline.py --method "$M" --dataset "$D" --split val >> "$LOG" 2>&1
  done
done

for D in UIEB LSUI; do
  for S in val test; do
    for B in phaseformer aquaclarity; do
      echo "=== features $D $S $B ===" >> "$LOG"
      "$PY" code/tools/extract_features_v3.py --dataset "$D" --split "$S" \
            --backbone "$B" --workers 24 >> "$LOG" 2>&1
    done
    echo "=== disagreement $D $S ===" >> "$LOG"
    "$PY" code/tools/ensemble_disagreement_v3.py --dataset "$D" --split "$S" >> "$LOG" 2>&1
  done
done
echo "=== PIPELINE DONE ===" >> "$LOG"
