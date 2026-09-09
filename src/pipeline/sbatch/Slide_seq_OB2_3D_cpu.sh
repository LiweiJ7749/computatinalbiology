#!/bin/bash
#SBATCH --job-name=slide_cpu
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH -o slide_cpu.%j.out
#SBATCH -e slide_cpu.%j.err

# Slide_seq_OB2_3D（3D，20 切片 × 30041 spot，float64 约 4.5GB）：前处理 + SPARK-X 原生 3D
# 说明：3D 数据仅 SPARK-X 原生支持；SpaGCN/SpaSEG 走逐切片合并（见 run_3d_slices_array_cpu.sh）。
# 此处 --skip-eval：3D 评估需等逐切片合并完成后统一进行（run_3d_slices_merge_cpu.sh）。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset Slide_seq_OB2_3D \
  --methods spark \
  --skip-eval
