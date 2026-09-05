#!/bin/bash
#SBATCH --job-name=olf_cpu
#SBATCH --partition=7542-64C-512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=8:00:00
#SBATCH -o olf_cpu.%j.out
#SBATCH -e olf_cpu.%j.err

# Visium_Mouse_Olfactory_Bulb（1185 spot × 32285 基因，47MB）：前处理 + SPARK-X + nnSVG
# 资源预估：nnSVG 32285 基因逐基因，64 核并行约几分钟~十几分钟 → 7542-64C-512G（免费）够。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset Visium_Mouse_Olfactory_Bulb \
  --methods spark,nnsvg \
  --cores 64 \
  --skip-eval
