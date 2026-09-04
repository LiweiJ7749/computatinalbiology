#!/bin/bash
#SBATCH --job-name=hd_kidney_cpu
#SBATCH --partition=7542-64C-512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=72:00:00
#SBATCH -o hd_kidney_cpu.%j.out
#SBATCH -e hd_kidney_cpu.%j.err

# Visium_HD_Mouse_Kidney（502009 spot × 19336 基因）：SPARK-X + nnSVG（前处理已由 preprocess 脚本单独完成）
# 注意：nnSVG 逐基因 BRISC 在此规模极慢（可能数小时~数十小时），日志里会分别打印各步耗时。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset Visium_HD_Mouse_Kidney \
  --methods spark,nnsvg \
  --cores 64 \
  --skip-preprocess --skip-eval
