#!/bin/bash
#SBATCH --job-name=stereo_cpu
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=6:00:00
#SBATCH -o stereo_cpu.%j.out
#SBATCH -e stereo_cpu.%j.err

# Stereo_seq_drosophila（3D，15295 spot × 13668 基因 dense，1.7GB）：前处理 + SPARK-X 原生 3D
# 说明：3D 数据仅 SPARK-X 原生支持；SpaGCN/SpaSEG 走逐切片合并（见 run_3d_slices_array_cpu.sh）。
# 此处 --skip-eval：3D 评估需等逐切片合并完成后统一进行（run_3d_slices_merge_cpu.sh），
# 避免在此重复计算全基因 Moran's I。dense 矩阵需大内存 → 6126-24C-768G。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset Stereo_seq_drosophila \
  --methods spark \
  --skip-eval
