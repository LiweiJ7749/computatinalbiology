#!/bin/bash
#SBATCH --job-name=dlpfc508_cpu
#SBATCH --partition=7542-64C-512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=12:00:00
#SBATCH -o dlpfc508_cpu.%j.out
#SBATCH -e dlpfc508_cpu.%j.err

# DLPFC_151508：前处理 + SPARK-X + nnSVG（CPU，33538 基因逐基因）
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset DLPFC_151508 \
  --methods spark,nnsvg \
  --cores 64 \
  --skip-eval
