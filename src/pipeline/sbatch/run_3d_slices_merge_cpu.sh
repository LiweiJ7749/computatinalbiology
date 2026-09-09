#!/bin/bash
#SBATCH --job-name=svg3d_merge
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=6:00:00
#SBATCH -o svg3d_merge.%j.out
#SBATCH -e svg3d_merge.%j.err
#
# 合并逐切片结果（merge_slices.py）+ 3D 评估（SPARK-X 原生 + 合并后的 SpaGCN/SpaSEG）
# + 空间可视化。建议在 array 完成后加依赖提交：
#   sbatch --export=ALL,DATASET=Stereo_seq_drosophila src/pipeline/sbatch/run_3d_slices_merge_cpu.sh
#   （或：sbatch --dependency=afterok:<array_job_id> --export=ALL,DATASET=<D> 本脚本）
set -euo pipefail

DATASET="${DATASET:-Stereo_seq_drosophila}"

cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/run_3d_slices_benchmark.sh \
  --dataset "$DATASET" \
  --methods spagcn,spaseg \
  --skip-export --skip-slices
