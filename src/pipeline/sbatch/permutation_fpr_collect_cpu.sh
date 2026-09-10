#!/bin/bash
#SBATCH --job-name=fpr_collect
#SBATCH --partition=6240-36C-192G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH -o fpr_collect.%j.out
#SBATCH -e fpr_collect.%j.err
#
# 汇总坐标置换 FPR 的 perm_*/result.json（在 array 完成后提交）。
#   sbatch --dependency=afterok:<array_job_id> --export=ALL,DATASET=Stereo_seq_drosophila,METHOD=spagcn \
#          src/pipeline/sbatch/permutation_fpr_collect_cpu.sh
set -euo pipefail

DATASET="${DATASET:-Stereo_seq_drosophila}"
METHOD="${METHOD:-spark}"

cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python

"$SVG_PYTHON" src/utils/permutation_fpr.py \
  --dataset "$DATASET" \
  --method "$METHOD" \
  --collect
