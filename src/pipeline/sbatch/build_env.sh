#!/bin/bash
#SBATCH --job-name=build_env
#SBATCH --partition=7542-64C-512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=4:00:00
#SBATCH -o build_env.%j.out
#SBATCH -e build_env.%j.err

# 在 CPU 节点构建 envs/spatial 与 envs/spatial_R（不占登录节点）。
# --device cuda 让 envs/spatial 装 cu124 torch（pip 装 torch 不需本机有 GPU），
# 这样 SpaGCN/SpaSEG 在 GPU 节点可直接复用同一环境。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
CONDA=$HOME/miniforge3/bin/conda DEVICE=cuda bash setup_linux_env.sh
