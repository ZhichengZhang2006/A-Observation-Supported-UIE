#!/usr/bin/env bash
# Phase-1 architecture ablation sweep: E0-E3 x {UIEB, LSUI}, one seed.
#
#   bash code/tools/run_ablation_phase1.sh            # the 8 screening runs (10k)
#   bash code/tools/run_ablation_phase1.sh extend e0 e2   # extend named arms to 20k
#
# Deliberately serial: the GPU is a 20 GB MIG slice, two training processes do not
# fit, and serial runs keep the per-update timing comparable across arms.
#
# Every run appends to runs/ablation_phase1/<tag>/train.log and writes
# result.json; the sweep log records the exact command line, so a run can be
# reproduced or resumed without guesswork.
set -u

PY=${PY:-/usr/local/iCompute/envs/Mamba/bin/python}
ROOT=${ROOT:-/home/user/obsui}
OUT=${OUT:-$ROOT/runs/ablation_phase1}
DATASETS=${DATASETS:-"UIEB LSUI"}
ARMS=${ARMS:-"e0 e1 e2 e3"}
UPDATES=${UPDATES:-10000}
EVAL_EVERY=${EVAL_EVERY:-500}
SEED=${SEED:-0}
EXTEND_TO=${EXTEND_TO:-20000}

export PYTHONPATH=$ROOT/code
export OBS_THREADS=${OBS_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT"

run_one() {
  local arm=$1 ds=$2 updates=$3 tag=$4
  local dir="$OUT/$tag"
  mkdir -p "$dir"
  if [ -f "$dir/result.json" ]; then
    echo "[skip] $tag already has result.json"
    return 0
  fi
  echo "=== $(date -Is) $tag ==="
  "$PY" "$ROOT/code/tools/train_ablation.py" \
      --arm "$arm" --dataset "$ds" --updates "$updates" \
      --eval-every "$EVAL_EVERY" --seed "$SEED" \
      --out-dir "$OUT" --tag "$tag" 2>&1 | tee "$dir/train.log"
  local rc=${PIPESTATUS[0]}
  echo "=== $(date -Is) $tag exit=$rc ==="
  return $rc
}

case "${1:-sweep}" in
  sweep)
    for ds in $DATASETS; do
      for arm in $ARMS; do
        run_one "$arm" "$ds" "$UPDATES" "${arm}_${ds}_${UPDATES}"
      done
    done
    ;;
  extend)
    shift
    if [ $# -lt 1 ]; then
      echo "usage: $0 extend <arm> [<arm> ...]   (extends to $EXTEND_TO, screened on UIEB+LSUI)"
      exit 2
    fi
    for arm in "$@"; do
      for ds in $DATASETS; do
        run_one "$arm" "$ds" "$EXTEND_TO" "${arm}_${ds}_${EXTEND_TO}"
      done
    done
    ;;
  *)
    echo "usage: $0 [sweep|extend <arm> ...]"
    exit 2
    ;;
esac

echo "=== $(date -Is) done ==="
"$PY" "$ROOT/code/tools/report_ablation.py" --runs "$OUT" \
    --out "$ROOT/artifacts/ablation_phase1.json" || true
