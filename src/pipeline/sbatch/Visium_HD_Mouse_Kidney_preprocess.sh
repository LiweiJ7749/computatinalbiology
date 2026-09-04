#!/bin/bash
#SBATCH --job-name=hd_kidney_pre
#SBATCH --partition=7542-64C-512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=4:00:00
#SBATCH -o hd_kidney_pre.%j.out
#SBATCH -e hd_kidney_pre.%j.err

# Visium_HD_Mouse_Kidney 前处理：读 1.7GB h5ad，一次生成四方法的输入。
# 独立成作业，使 cpu（SPARK/nnSVG）与 gpu（SpaGCN/SpaSEG）在前处理完成后并行，不必互相等待。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python

$SVG_PYTHON src/preprocess/h5ad_preprocess.py \
  --dataset Visium_HD_Mouse_Kidney \
  --methods spark nnsvg spagcn spaseg
