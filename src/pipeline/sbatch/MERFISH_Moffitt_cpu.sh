#!/bin/bash
#SBATCH --job-name=merfish_cpu
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH -o merfish_cpu.%j.out
#SBATCH -e merfish_cpu.%j.err

# MERFISH_Moffitt（103万 spot × 161 基因）：SPARK-X(投影核) + nnSVG + SpaGCN
# （前处理已按 bin_factor=8 聚合到 ~2万 meta-spot；SPARK-X 用投影核 single 避免 O(n^2)）。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset MERFISH_Moffitt \
  --methods spark,nnsvg,spagcn \
  --cores 8 \
  --skip-preprocess --skip-eval
