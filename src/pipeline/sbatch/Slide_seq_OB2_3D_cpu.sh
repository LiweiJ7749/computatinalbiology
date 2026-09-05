#!/bin/bash
#SBATCH --job-name=slide_cpu
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH -o slide_cpu.%j.out
#SBATCH -e slide_cpu.%j.err

# Slide_seq_OB2_3D（3D，20 切片 × 30041 spot，float64 约 4.5GB）：前处理 + SPARK-X
# 说明：3D 数据仅 SPARK-X 支持；多切片 2D→3D 重建尚待实现，运行前需确认前处理分支。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset Slide_seq_OB2_3D \
  --methods spark \
  --skip-eval
