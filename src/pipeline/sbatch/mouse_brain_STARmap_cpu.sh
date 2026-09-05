#!/bin/bash
#SBATCH --job-name=star_cpu
#SBATCH --partition=6240-36C-192G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=2:00:00
#SBATCH -o star_cpu.%j.out
#SBATCH -e star_cpu.%j.err

# mouse_brain_STARmap（930 spot × 996 基因，~MB）：前处理 + SPARK-X + nnSVG（CPU）
# 资源预估：SPARK-X<1s、nnSVG 分钟级 → 6240-36C-192G（36核/192G 免费）足够。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset mouse_brain_STARmap \
  --methods spark,nnsvg \
  --cores 32 \
  --skip-eval
