#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --partition=normal_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:25:00
#SBATCH -o smoke_test.%j.out
#SBATCH -e smoke_test.%j.err

# 简单冒烟测试（normal_test 队列，30 分钟内强杀）：
#   1) 验证 Python 环境  2) 验证 R 环境  3) 前处理 + SPARK-X（独立 outdir 不污染正式结果）
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$HOME/svg_methods/envs/spatial_R/bin:$PATH
export LD_LIBRARY_PATH=$HOME/svg_methods/envs/spatial_R/lib:${LD_LIBRARY_PATH:-}
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

echo "=== [1/3] Python 环境 ==="
$SVG_PYTHON -c "import scanpy, torch, numpy, pandas; print('python OK | torch', torch.__version__, '| scanpy', scanpy.__version__)"

echo "=== [2/3] R 环境 ==="
$SVG_RSCRIPT -e 'suppressPackageStartupMessages({library(SPARK); library(nnSVG); library(SpatialExperiment)}); cat("R OK\n")'

echo "=== [3/3] 前处理 + SPARK-X（独立 outdir，不污染正式结果）==="
bash src/pipeline/models_benchmark.sh \
  --dataset mouse_brain_STARmap \
  --methods spark \
  --outdir results/test_smoke \
  --skip-eval

echo "=== SMOKE TEST DONE ==="
