#!/usr/bin/env bash
# Re-run every baseline whose official weights are present, on every dataset,
# under the frozen protocol.  Run detached because the 428-image LSUI sweep alone
# takes well over an hour and an ssh channel must not be the thing keeping it alive.
#
# Invariant this script protects: the protocol is not allowed to vary by dataset.
# A per-dataset resize rule is exactly the bug that made the third-party prediction
# folders unquotable (UIEB low by 2.3 dB while UFO was high by 0.6 dB -- wrong in
# both directions, which no single rescaling explains).  So every dataset goes
# through the same whole-image -> 256x256 bilinear path here.
set -u
cd /home/user/obsui || exit 1
PY=/usr/local/iCompute/envs/Mamba/bin/python
mkdir -p logs

for DS in LSUI UFO; do
  for M in phaseformer aquaclarity fdcenet cdfuie ssuie cclnet sfgnet; do
    OUT="logs/base_${M}_${DS}.log"
    if [ -s "artifacts/baseline_${M}_${DS}.json" ]; then
      echo "skip ${M}/${DS} (already scored)"
      continue
    fi
    echo "=== ${M} / ${DS} ==="
    $PY -u code/tools/run_baseline.py \
        --method "$M" --dataset "$DS" --device cuda:0 \
        > "$OUT" 2>&1
    echo "  exit=$? $(tail -2 "$OUT" | tr '\n' ' ')"
  done
done

echo "ALL BASELINE RUNS DONE"
