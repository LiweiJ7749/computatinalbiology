#!/usr/bin/env bash
# =============================================================================
# submit_drosophila_3d.sh —— 一条命令提交 drosophila 全量 3D 完整流程
# =============================================================================
# 用法（登录节点，只负责提交作业，不做计算）：
#   cd ~/svg_methods && bash src/pipeline/sbatch/submit_drosophila_3d.sh
#
# 自动按依赖顺序提交：
#   1) 全量 3D 流水线（SPARK-X + SpaGCN/SpaSEG 逐切片 + 合并 + 评估 + 空间图）
#   2) 三个方法的坐标置换 FPR（spark/spagcn/spaseg，各 20 次置换，array 并行）
#   3) 各方法 FPR 的 collect 汇总（依赖对应 array 完成）
# =============================================================================
set -euo pipefail
cd ~/svg_methods

DS=Stereo_seq_drosophila
N_PERMS="${N_PERMS:-20}"
DEVICE="${DEVICE:-cpu}"

echo "[submit] 全量 3D 流水线"
J0=$(sbatch --parsable src/pipeline/sbatch/Stereo_seq_drosophila_cpu.sh)

for M in spark spagcn spaseg; do
  F=$(sbatch --parsable --dependency=afterok:${J0} \
    --export=ALL,DATASET=${DS},METHOD=${M},N_PERMS=${N_PERMS},DEVICE=${DEVICE} \
    src/pipeline/sbatch/permutation_fpr_cpu.sh)
  sbatch --dependency=afterok:${F} \
    --export=ALL,DATASET=${DS},METHOD=${M} \
    src/pipeline/sbatch/permutation_fpr_collect_cpu.sh >/dev/null
  echo "[submit] ${M} FPR array=${F}（collect 已挂依赖）"
done

echo "[submit] 完成：full_pipeline=${J0}"
