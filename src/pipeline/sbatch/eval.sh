#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=7542-64C-512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=1:00:00
#SBATCH -o eval.%j.out
#SBATCH -e eval.%j.err

# 评估汇总（在 cpu + gpu 作业都完成后提交）：sbatch eval.sh <dataset>
set -euo pipefail
cd ~/svg_methods
DATASET=${1:?用法: sbatch eval.sh <dataset>}
$HOME/svg_methods/envs/spatial/bin/python src/utils/evaluation.py --dataset "$DATASET"
