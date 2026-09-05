#!/bin/bash
#SBATCH --job-name=merfish_pre
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=4:00:00
#SBATCH -o merfish_pre.%j.out
#SBATCH -e merfish_pre.%j.err

# MERFISH_Moffitt 前处理：103万 spot × 161 基因 dense（1.4GB），读入+导出需大内存。
# 独立成作业，使 cpu（SPARK/nnSVG）与 gpu（SpaSEG）在前处理后并行。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python

$SVG_PYTHON src/preprocess/h5ad_preprocess.py \
  --dataset MERFISH_Moffitt \
  --methods spark nnsvg spagcn spaseg
