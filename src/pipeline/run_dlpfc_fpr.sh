#!/usr/bin/env bash
# =============================================================================
# run_dlpfc_fpr.sh —— DLPFC_151507 四方法坐标置换 FPR（本地 CPU，顺序执行）
# 每个方法 --n-perms 20，末了 permutation_fpr.py 自动 collect。
# nnSVG 放最前（新增的 R 置换分支，最早暴露问题）；单方法失败不中断其余。
# =============================================================================
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/envs/spatial/bin/python"
LOG="$ROOT/results/local_results/DLPFC_151507/logs"
mkdir -p "$LOG"

run_one() {
  local m="$1"
  echo "======================================================"
  echo "  FPR $m 开始   $(date '+%F %T')"
  echo "======================================================"
  if "$PY" src/utils/permutation_fpr.py \
       --dataset DLPFC_151507 --method "$m" --n-perms 20 --device cpu \
       2>&1 | tee "$LOG/fpr_${m}.log"; then
    echo "  FPR $m 完成   $(date '+%F %T')"
  else
    echo "  FPR $m 失败   $(date '+%F %T')  —— 详见 $LOG/fpr_${m}.log"
  fi
}

run_one nnsvg
run_one spark
run_one spagcn
run_one spaseg

echo "======================================================"
echo "  全部 FPR 结束  $(date '+%F %T')"
echo "======================================================"
