#!/bin/bash
#SBATCH --job-name=dlpfc510_cpu
#SBATCH --partition=7542-64C-512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=24:00:00
#SBATCH -o dlpfc510_cpu.%j.out
#SBATCH -e dlpfc510_cpu.%j.err

# DLPFC_151510（4634 spot × 33538 基因）：全流程 CPU 单作业（四方法）
# - nnSVG 33538 基因逐基因，--cores 64 并行
# - SpaGCN/SpaSEG 放 CPU（小数据 GPU 利用率低，省 GPU 卡）
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset DLPFC_151510 \
  --methods spark,nnsvg,spagcn,spaseg \
  --cores 64 \
  --device cpu
