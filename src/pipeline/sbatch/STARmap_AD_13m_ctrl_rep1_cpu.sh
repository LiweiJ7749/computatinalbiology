#!/bin/bash
#SBATCH --job-name=starad_cpu
#SBATCH --partition=7542-64C-512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=12:00:00
#SBATCH -o starad_cpu.%j.out
#SBATCH -e starad_cpu.%j.err

# STARmap_AD_13m_ctrl_rep1（8034 spot × 2766 基因，35MB）：前处理 + SPARK-X + nnSVG
# 资源预估：nnSVG 2766 基因逐基因，64 核约数十分钟~数小时 → 7542-64C-512G（免费）。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset STARmap_AD_13m_ctrl_rep1 \
  --methods spark,nnsvg \
  --cores 64 \
  --skip-eval
