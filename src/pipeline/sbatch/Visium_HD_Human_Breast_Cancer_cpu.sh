#!/bin/bash
#SBATCH --job-name=bc_cpu
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH -o bc_cpu.%j.out
#SBATCH -e bc_cpu.%j.err

# Human_Breast_Cancer：SPARK-X（超大 h5ad，仅 SPARK-X 线性可扩展；nnSVG/SpaGCN 不适用）
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset Visium_HD_Human_Breast_Cancer \
  --methods spark \
  --skip-preprocess --skip-eval
