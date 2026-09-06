#!/usr/bin/env bash
# =============================================================================
# run_3d_slices_benchmark.sh —— 3D 数据集上 2D 方法（nnSVG/SpaGCN/SpaSEG）的
# “逐切片检测 + 跨切片合并”主控脚本（Linux / WSL / HPC）
# =============================================================================
# 针对 Slide-seq / Stereo-seq 这类 dim=3 数据：SPARK-X 走原生 3D（见
# models_benchmark.sh --methods spark），本脚本负责其余三个 2D 方法：
#
#   1) export_3d_slices.py  逐切片导出 2D 输入
#   2) 逐切片运行现有 2D 脚本（nnSVG / SpaGCN / SpaSEG）
#   3) merge_slices.py      用 Fisher 组合 p 值把逐切片 *_rank.csv 合并为单一排名
#
# 用法（项目根）：
#   bash src/pipeline/run_3d_slices_benchmark.sh --dataset Slide_seq_OB2_3D
#   bash src/pipeline/run_3d_slices_benchmark.sh --dataset Stereo_seq_drosophila --methods nnsvg spagcn
# =============================================================================
set -euo pipefail

DATASET=""
H5AD=""
OUTDIR_ARG=""
SAMPLE_ARG=""
METHODS_ARG=""
CORES=""
DEVICE="auto"
SKIP_EXPORT=0

usage() {
  cat <<'EOF' >&2
用法: run_3d_slices_benchmark.sh [选项]
  --dataset KEY        dim=3 数据集 key（Slide_seq_OB2_3D / Stereo_seq_drosophila）
  --h5ad PATH          显式 h5ad（仅 Stereo-seq 单文件形式；Slide-seq 用注册表 slices）
  --outdir PATH        输出根目录（默认 results/local_results/<dataset>）
  --sample NAME        样本标签
  --methods LIST       逗号分隔 2D 方法子集（默认 nnsvg,spagcn,spaseg）
  --cores N            nnSVG 线程数
  --device DEV         python 模型设备（auto/cuda/cpu）
  --skip-export        跳过逐切片导出（要求已生成）
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --h5ad)    H5AD="${2:-}"; shift 2 ;;
    --outdir)  OUTDIR_ARG="${2:-}"; shift 2 ;;
    --sample)  SAMPLE_ARG="${2:-}"; shift 2 ;;
    --methods) METHODS_ARG="${2:-}"; shift 2 ;;
    --cores)   CORES="${2:-}"; shift 2 ;;
    --device)  DEVICE="${2:-}"; shift 2 ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[错误] 未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

now() { date '+%H:%M:%S'; }
log_header() { echo ""; echo "============================================================"; echo "  [$(now)] $1"; echo "============================================================"; }
log_msg() { echo "  [$(now)] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export SVG_REPO_ROOT="$ROOT"

PYTHON="${SVG_PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for c in "$ROOT/envs/spatial/bin/python" "$ROOT/envs/spatial/bin/python3"; do
    [ -x "$c" ] && { PYTHON="$c"; break; }
  done
  [ -z "$PYTHON" ] && PYTHON="$(command -v python3 || command -v python || true)"
fi
RSCRIPT="${SVG_RSCRIPT:-}"
if [ -z "$RSCRIPT" ]; then
  [ -x "$ROOT/envs/spatial_R/bin/Rscript" ] && RSCRIPT="$ROOT/envs/spatial_R/bin/Rscript" || RSCRIPT="$(command -v Rscript || true)"
fi
[ -z "$PYTHON" ] && { echo "[错误] 找不到 python，请设置 SVG_PYTHON" >&2; exit 2; }
[ -z "$RSCRIPT" ] && { echo "[错误] 找不到 Rscript，请设置 SVG_RSCRIPT" >&2; exit 2; }

# 2D 方法集合
TWO_D="nnsvg spagcn spaseg"
if [ -n "$METHODS_ARG" ]; then
  METHODS=""
  for m in ${METHODS_ARG//,/ }; do
    case " $TWO_D " in
      *" $m "*) METHODS="$METHODS $m" ;;
      *) echo "[警告] 忽略未知 2D 方法: $m（可选: $TWO_D）" >&2 ;;
    esac
  done
  METHODS="${METHODS# }"
else
  METHODS="$TWO_D"
fi
[ -z "$METHODS" ] && { echo "[错误] 没有可运行的 2D 方法" >&2; exit 1; }

# 解析 run（用 methods=["spark"] 规避 resolve_run 的 3D 方法过滤）
resolved="$("$PYTHON" - "$DATASET" "$OUTDIR_ARG" "$SAMPLE_ARG" "$H5AD" 2>&1 <<'PY' || true
import os, sys
sys.path.insert(0, os.environ["SVG_REPO_ROOT"])
import src
ds, out, sample, h5ad = (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
run = src.resolve_run(dataset=(ds or None), outdir=(out or None),
                      sample=(sample or None), h5ad=(h5ad or None),
                      methods=["spark"])
print("<<<RES>>>outdir=" + str(run["outdir"]))
print("<<<RES>>>sample=" + str(run["sample"]))
print("<<<RES>>>tech=" + str(run.get("tech") or ""))
print("<<<RES>>>dim=" + str(run.get("dim")))
PY
)"
OUTDIR="$(printf '%s\n' "$resolved" | sed -n 's/^<<<RES>>>outdir=//p')"
SAMPLE="$(printf '%s\n' "$resolved" | sed -n 's/^<<<RES>>>sample=//p')"
TECH="$(printf '%s\n' "$resolved" | sed -n 's/^<<<RES>>>tech=//p')"
DIM="$(printf '%s\n' "$resolved" | sed -n 's/^<<<RES>>>dim=//p')"
[ "$DIM" = "3" ] || { echo "[错误] 数据集 dim=$DIM，非 3D" >&2; printf '%s\n' "$resolved" >&2; exit 2; }
[ -z "$OUTDIR" ] || [ -z "$SAMPLE" ] && { echo "[错误] 无法解析 run 配置" >&2; printf '%s\n' "$resolved" >&2; exit 2; }

mkdir -p "$OUTDIR/logs"
log_header "3D 逐切片 2D 方法: $DATASET"
log_msg "outdir = $OUTDIR | sample = $SAMPLE | tech = $TECH | methods = $METHODS"

run_R() {
  ( cd "$ROOT" && \
    PATH="$ROOT/envs/spatial_R/bin:$PATH" \
    LD_LIBRARY_PATH="$ROOT/envs/spatial_R/lib:${LD_LIBRARY_PATH:-}" \
    "$RSCRIPT" "$@" )
}

subdir_of() {
  case "$1" in
    nnsvg)  echo "nnSVG" ;;
    spagcn) echo "spaGCN" ;;
    spaseg) echo "spaSEG" ;;
  esac
}

export_nnsvg_params() {
  local line
  line="$("$PYTHON" -c "
import json
try:
    p = json.load(open('$ROOT/configs/run_params.json'))
    n = p.get('$DATASET', {}).get('nnsvg', {})
    print(n.get('pcspots', 0.01), n.get('ncounts', 3))
except Exception:
    print(0.01, 3)
" 2>/dev/null)"
  export NNSVG_PCSPOTS="$(echo "$line" | awk '{print $1}')"
  export NNSVG_NCOUNTS="$(echo "$line" | awk '{print $2}')"
}

# ---------------- 1) 逐切片导出 2D 输入 ----------------
if [ "$SKIP_EXPORT" -eq 0 ]; then
  log_header "步骤 1/3: 逐切片导出 2D 输入 (export_3d_slices.py)"
  EXTRA=()
  [ -n "$H5AD" ] && EXTRA+=(--h5ad "$H5AD")
  "$PYTHON" "$ROOT/src/preprocess/export_3d_slices.py" \
    --dataset "$DATASET" "${EXTRA[@]}" --outdir "$OUTDIR" --sample "$SAMPLE" \
    --methods $METHODS 2>&1 | tee -a "$OUTDIR/logs/export_3d_slices.log"
fi

# 用第一个方法的 slices 目录确定切片集合（所有方法切片结构一致）
REF_METHOD="$(echo "$METHODS" | awk '{print $1}')"
REF_SUBDIR="$(subdir_of "$REF_METHOD")"
shopt -s nullglob
slice_dirs=("$OUTDIR"/"$REF_SUBDIR"/slices/S*)
shopt -u nullglob
[ "${#slice_dirs[@]}" -gt 0 ] || { echo "[错误] 未找到切片目录: $OUTDIR/$REF_SUBDIR/slices/S*" >&2; exit 1; }
log_msg "切片数 = ${#slice_dirs[@]}"

# ---------------- 2) 逐切片运行 + 合并 ----------------
export_nnsvg_params
for m in $METHODS; do
  subdir="$(subdir_of "$m")"
  log_header "步骤 2/3: 逐切片运行 $m"
  for d in "${slice_dirs[@]}"; do
    i="$(basename "$d" | sed 's/^S//')"
    sample_slice="${SAMPLE}_S${i}"
    case "$m" in
      nnsvg)
        if [ -n "$CORES" ]; then
          run_R "$ROOT/src/r_models/run_nnSVG.r" "$d" "$sample_slice" "$CORES" 2>&1 | tee -a "$OUTDIR/logs/nnsvg_S${i}.log"
        else
          run_R "$ROOT/src/r_models/run_nnSVG.r" "$d" "$sample_slice" 2>&1 | tee -a "$OUTDIR/logs/nnsvg_S${i}.log"
        fi
        ;;
      spagcn)
        "$PYTHON" "$ROOT/src/py_models/run_spaGCN.py" \
          --h5ad "$d/spaGCN/${sample_slice}_spaGCN.h5ad" --outdir "$d" \
          --sample "$sample_slice" --device "$DEVICE" 2>&1 | tee -a "$OUTDIR/logs/spagcn_S${i}.log"
        ;;
      spaseg)
        "$PYTHON" "$ROOT/src/py_models/run_spaSEG.py" \
          --h5ad "$d/spaSEG/${sample_slice}_spaSEG.h5ad" --outdir "$d" \
          --sample "$sample_slice" --device "$DEVICE" 2>&1 | tee -a "$OUTDIR/logs/spaseg_S${i}.log"
        ;;
    esac
  done

  log_header "步骤 3/3: 合并 $m 切片排名 (merge_slices.py)"
  "$PYTHON" "$ROOT/src/py_models/merge_slices.py" \
    --method "$m" --method-dir "$OUTDIR/$subdir" --sample "$SAMPLE" \
    2>&1 | tee -a "$OUTDIR/logs/merge_${m}.log"
done

log_header "run_3d_slices_benchmark 完成"
echo "结果目录: $OUTDIR"
echo "合并排名: $OUTDIR/{nnSVG,spaGCN,spaSEG}/SVG_*_${SAMPLE}_rank.csv"
