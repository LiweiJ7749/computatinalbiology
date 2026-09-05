#!/bin/bash
#SBATCH --job-name=bc_convert
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH -o bc_convert.%j.out
#SBATCH -e bc_convert.%j.err

# Human_Breast_Cancer：feature_slice.h5(1.6GB) -> h5ad（bin_size=16 控制规模）
# 说明：11mm 大组织，8um bin 会产生超大 bin 数，16um 为 1/4 规模。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python

$SVG_PYTHON src/preprocess/10xVisium_pretreat_h5toh5ad.py \
  data/10xVisium/Human_Breast_Cancer/Visium_HD_11mm_Human_Breast_Cancer_feature_slice.h5 \
  data/10xVisium/Human_Breast_Cancer/Visium_HD_11mm_Human_Breast_Cancer_spatial.tar.gz \
  data/10xVisium/Human_Breast_Cancer/Visium_HD_Human_Breast_Cancer.h5ad \
  --bin-size 16
