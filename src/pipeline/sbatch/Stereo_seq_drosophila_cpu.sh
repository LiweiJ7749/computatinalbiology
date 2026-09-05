#!/bin/bash
#SBATCH --job-name=stereo_cpu
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=6:00:00
#SBATCH -o stereo_cpu.%j.out
#SBATCH -e stereo_cpu.%j.err

# Stereo_seq_drosophila（3D，15295 spot × 13668 基因 dense，1.7GB）：前处理 + SPARK-X
# 说明：3D 数据仅 SPARK-X 支持（dim=3 自动过滤）；dense 矩阵需大内存 → 6126-24C-768G。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset Stereo_seq_drosophila \
  --methods spark \
  --skip-eval
