#!/bin/bash
#SBATCH --job-name=merfish_cpu
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH -o merfish_cpu.%j.out
#SBATCH -e merfish_cpu.%j.err

# MERFISH_Moffitt（103万 spot × 161 基因）：全流程 CPU 单作业
# - 方法 spark,spagcn,spaseg（nnSVG 由 datasets.json 的 exclude_methods 排除）
# - 前处理按 run_params bin_factor=8 聚合；SPARK-X 用投影核 single 避免 O(n^2)
# - SpaSEG 放 CPU（训练 CNN 极小、detect_svg 单核，GPU 利用率低，省 GPU 卡）
# - eval 在末尾自动执行；161 基因 → X.toarray 仅 ~1.3GB，内存安全
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset MERFISH_Moffitt \
  --methods spark,spagcn,spaseg \
  --device cpu
