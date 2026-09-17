#!/usr/bin/env bash
# run_fpr_dataset.sh <dataset> —— 通用四方法坐标置换 FPR（顺序，各 20 次自动 collect）
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/envs/spatial/bin/python"
DATASET="${1:?用法: run_fpr_dataset.sh <dataset>}"
LOG="$ROOT/results/local_results/$DATASET/logs"
mkdir -p "$LOG"

run_one() {
  local m="$1"
  echo "======================================================"
  echo "  FPR $m ($DATASET) 开始  $(date '+%F %T')"
  echo "======================================================"
  if "$PY" src/utils/permutation_fpr.py \
       --dataset "$DATASET" --method "$m" --n-perms 20 --device cpu \
       2>&1 | tee "$LOG/fpr_${m}.log"; then
    echo "  FPR $m 完成  $(date '+%F %T')"
  else
    echo "  FPR $m 失败  $(date '+%F %T')  —— 详见 $LOG/fpr_${m}.log"
  fi
}

run_one spark
run_one nnsvg
run_one spagcn
run_one spaseg

echo "======================================================"
echo "  全部 FPR ($DATASET) 结束  $(date '+%F %T')"
echo "======================================================"
