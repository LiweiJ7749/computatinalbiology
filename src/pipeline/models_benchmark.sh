#!/usr/bin/env bash
# =============================================================================
# models_benchmark.sh —— 四方法(SVG)批量对比主控脚本（Linux / WSL / HPC）
# =============================================================================
# 依次完成：
#   0) 解析 run 配置（dataset/outdir/sample，经 src.resolve_run 归一）
#   1) 共同前处理：src/preprocess/h5ad_preprocess.py（建目录 + 生成
#      SPARK_X|nnSVG 的 counts.mtx 等 + spaGCN/spaSEG 的 <sample>_*.h5ad）
#   2) 依次调用四个方法（同一份 h5ad）：
#        SPARK-X : Rscript src/r_models/run_spark.r
#        nnSVG   : Rscript src/r_models/run_nnSVG.r
#        SpaGCN  : python src/py_models/run_spaGCN.py
#        SpaSEG  : python src/py_models/run_spaSEG.py
#   3) 每步输出日志到 <outdir>/logs/<step>.log 与控制台
#   4) 最后调用 src/utils/evaluation.py
#
# 说明：
#   - R 依赖 renv 项目库（项目根 .Rprofile 自动挂载），故所有子命令都在
#     $ROOT 目录下执行。
#   - R 解释器：优先 SVG_RSCRIPT 环境变量，然后项目内 envs/spatial_R/bin/Rscript
#     （conda R 4.3.1 + renv 统一管理包），最后回退系统 Rscript。
#   - Python 解释器：优先 SVG_PYTHON 环境变量，然后项目内 envs/spatial/bin/python。
#
# 用法示例（在项目根）：
#   bash src/pipeline/models_benchmark.sh --dataset mouse_brain_STARmap
#   bash src/pipeline/models_benchmark.sh --dataset mouse_brain_STARmap --methods spagcn,spaseg
#   bash src/pipeline/models_benchmark.sh --h5ad ./data/.../x.h5ad --outdir ./results/local_results/my --sample my
# =============================================================================
set -euo pipefail

# ---------------- 参数默认值 ----------------
DATASET="mouse_brain_STARmap"
H5AD=""
SPATIAL=""
OUTDIR_ARG=""
SAMPLE_ARG=""
METHODS_ARG=""
CORES=""
DEVICE="auto"
SKIP_PRE=0
SKIP_EVAL=0
ALL_METHODS="spark nnsvg spagcn spaseg"

usage() {
  cat <<'EOF' >&2
用法: models_benchmark.sh [选项]
  --dataset KEY        数据集 key（默认 mouse_brain_STARmap；见 src.DATASETS）
  --h5ad PATH          输入 h5ad（绝对或相对项目根，覆盖 dataset）
  --spatial PATH       坐标文件（tissue_positions.csv / spatial.tar.gz）
  --outdir PATH        输出根目录（默认 results/local_results/<dataset>）
  --sample NAME        样本标签（默认取 dataset 注册值）
  --methods LIST       逗号分隔方法子集（默认全部: spark,nnsvg,spagcn,spaseg）
  --cores N            nnSVG 线程数（Linux fork 并行生效）
  --device DEV         python 模型设备（auto/cuda/cpu，默认 auto）
  --skip-preprocess    跳过共同前处理（要求数据已生成）
  --skip-eval          跳过最后的 evaluation.py 调用
  -h, --help
环境变量: SVG_PYTHON / SVG_RSCRIPT（解释器路径覆盖）
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --h5ad)    H5AD="${2:-}"; shift 2 ;;
    --spatial) SPATIAL="${2:-}"; shift 2 ;;
    --outdir)  OUTDIR_ARG="${2:-}"; shift 2 ;;
    --sample)  SAMPLE_ARG="${2:-}"; shift 2 ;;
    --methods) METHODS_ARG="${2:-}"; shift 2 ;;
    --cores)   CORES="${2:-}"; shift 2 ;;
    --device)  DEVICE="${2:-}"; shift 2 ;;
    --skip-preprocess) SKIP_PRE=1; shift ;;
    --skip-eval)       SKIP_EVAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[错误] 未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

# ---------------- 日志工具（结构化、带时间戳） ----------------
now() { date '+%H:%M:%S'; }
log_header() {
  local title="$1"
  echo ""
  echo "============================================================"
  echo "  [$(now)] $title"
  echo "============================================================"
}
log_msg()  { echo "  [$(now)] $*"; }

# ---------------- 定位项目根与解释器 ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export SVG_REPO_ROOT="$ROOT"

PYTHON="${SVG_PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for c in "$ROOT/envs/spatial/bin/python" "$ROOT/envs/spatial/bin/python3"; do
    if [ -n "$c" ] && [ -x "$c" ]; then PYTHON="$c"; break; fi
  done
  [ -z "$PYTHON" ] && PYTHON="$(command -v python3 || command -v python || true)"
fi
RSCRIPT="${SVG_RSCRIPT:-}"
if [ -z "$RSCRIPT" ]; then
  for c in "$ROOT/envs/spatial_R/bin/Rscript"; do
    if [ -n "$c" ] && [ -x "$c" ]; then RSCRIPT="$c"; break; fi
  done
  [ -z "$RSCRIPT" ] && RSCRIPT="$(command -v Rscript || true)"
fi
[ -z "$PYTHON" ] && { echo "[错误] 找不到 python，请设置环境变量 SVG_PYTHON" >&2; exit 2; }
[ -z "$RSCRIPT" ] && { echo "[错误] 找不到 Rscript，请设置环境变量 SVG_RSCRIPT" >&2; exit 2; }

log_header "models_benchmark: 批量 SVG 对比"
log_msg "ROOT    = $ROOT"
log_msg "PYTHON  = $PYTHON"
log_msg "RSCRIPT = $RSCRIPT"
log_msg "DATASET = $DATASET"

# ---------------- 方法选择（逗号/空格分隔，规范顺序） ----------------
if [ -n "$METHODS_ARG" ]; then
  METHODS=""
  for m in ${METHODS_ARG//,/ }; do
    case " $ALL_METHODS " in
      *" $m "*) METHODS="$METHODS $m" ;;
      *) echo "[警告] 未知方法: $m（忽略，可选: $ALL_METHODS）" >&2 ;;
    esac
  done
  METHODS="${METHODS# }"
else
  METHODS="$ALL_METHODS"
fi
[ -z "$METHODS" ] && { echo "[错误] 没有可运行的方法" >&2; exit 1; }
log_msg "methods = $METHODS"

# ---------------- 解析 run（outdir/sample/方法过滤以 python src 为准） ----------------
# 用固定前缀标记输出行，避免 src.resolve_run 的日志污染解析（其 3D 跳过提示会打印到 stdout）。
resolved="$("$PYTHON" - "$DATASET" "$OUTDIR_ARG" "$SAMPLE_ARG" "$H5AD" "$SPATIAL" "$METHODS" 2>&1 <<'PY' || true
import os, sys
sys.path.insert(0, os.environ["SVG_REPO_ROOT"])
import src
ds, out, sample, h5ad, spatial, meth = (sys.argv[1], sys.argv[2], sys.argv[3],
                                        sys.argv[4], sys.argv[5], sys.argv[6])
meth_list = [m for m in meth.replace(',', ' ').split() if m]
run = src.resolve_run(dataset=(ds or None), outdir=(out or None),
                      sample=(sample or None), h5ad=(h5ad or None),
                      spatial=(spatial or None), methods=(meth_list or None))
print("<<<RES>>>outdir=" + str(run["outdir"]))
print("<<<RES>>>sample=" + str(run["sample"]))
print("<<<RES>>>methods=" + " ".join(run["methods"]))
PY
)"
OUTDIR="$(printf '%s\n' "$resolved" | sed -n 's/^<<<RES>>>outdir=//p')"
SAMPLE="$(printf '%s\n' "$resolved" | sed -n 's/^<<<RES>>>sample=//p')"
# 3D 数据集会在 resolve_run 内把方法过滤为仅 spark（4 方法中仅 SPARK-X 支持 3D）
METHODS="$(printf '%s\n' "$resolved" | sed -n 's/^<<<RES>>>methods=//p')"
[ -z "$METHODS" ] && { echo "[错误] 请求方法与该数据集维度无交集（dim=3 仅支持 spark）" >&2; exit 1; }
if [ -z "$OUTDIR" ] || [ -z "$SAMPLE" ]; then
  echo "[错误] 无法解析 run 配置（dataset=$DATASET）。" >&2
  echo "  - 请检查 src/__init__.py 的 DATASETS 是否含该 key，" >&2
  echo "    或改用 --h5ad/--outdir/--sample 显式指定。" >&2
  printf '  python 输出:\n%s\n' "$resolved" >&2
  exit 2
fi

mkdir -p "$OUTDIR/logs"

log_msg "outdir = $OUTDIR"
log_msg "sample = $SAMPLE"
log_msg "device = $DEVICE"

# 传递 python 方法的公共参数（h5ad 仅当用户显式给出才传）
PY_EXTRA=()
[ -n "$H5AD" ]    && PY_EXTRA+=(--h5ad "$H5AD")
[ -n "$SPATIAL" ] && PY_EXTRA+=(--spatial "$SPATIAL")

# 在项目根执行 R（项目 .Rprofile 已挂载 renv 库）+ 带上 conda R 的 bin/lib 路径
# （lib 需显式加入，R 包编译期 RPATH 指向各自 conda 前缀，跨前缀运行时靠
#  LD_LIBRARY_PATH 兜底）。
run_R() {
  ( cd "$ROOT" && \
    PATH="$ROOT/envs/spatial_R/bin:$PATH" \
    LD_LIBRARY_PATH="$ROOT/envs/spatial_R/lib:${LD_LIBRARY_PATH:-}" \
    "$RSCRIPT" "$@" )
}

# 读取数据集级 nnSVG 过滤参数（configs/run_params.json）并 export 为环境变量，
# 供 run_nnSVG.r 通过 NNSVG_PCSPOTS / NNSVG_NCOUNTS 读取。
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
  log_msg "nnSVG 过滤参数: pcspots=${NNSVG_PCSPOTS}% ncounts=${NNSVG_NCOUNTS}"
}

# ---------------- 1) 共同前处理 ----------------
if [ "$SKIP_PRE" -eq 0 ]; then
  log_header "步骤 1/2: 共同前处理 (h5ad_preprocess.py)"
  PREP_SCRIPT="$ROOT/src/preprocess/h5ad_preprocess.py"
  # shellcheck disable=SC2086
  "$PYTHON" "$PREP_SCRIPT" --dataset "$DATASET" \
    "${PY_EXTRA[@]}" \
    --outdir "$OUTDIR" --sample-name "$SAMPLE" --methods $METHODS \
    2>&1 | tee -a "$OUTDIR/logs/preprocess.log"
  log_msg "前处理完成"
fi

# ---------------- 2) 逐方法运行 ----------------
log_header "步骤 2/2: 运行各方法"
for m in $METHODS; do
  case "$m" in
    spark)
      log_header "SPARK-X: $SAMPLE 运行开始"
      t0="$(date +%s)"
      run_R "$ROOT/src/r_models/run_spark.r" \
          "$OUTDIR/SPARK_X" "$SAMPLE" 2>&1 | tee -a "$OUTDIR/logs/spark.log"
      log_msg "SPARK-X 完成（$(( $(date +%s) - t0 ))s）"
      ;;
    nnsvg)
      log_header "nnSVG: $SAMPLE 运行开始"
      t0="$(date +%s)"
      if [ -n "$CORES" ]; then
        run_R "$ROOT/src/r_models/run_nnSVG.r" \
            "$OUTDIR/nnSVG" "$SAMPLE" "$CORES" 2>&1 | tee -a "$OUTDIR/logs/nnsvg.log"
      else
        run_R "$ROOT/src/r_models/run_nnSVG.r" \
            "$OUTDIR/nnSVG" "$SAMPLE" 2>&1 | tee -a "$OUTDIR/logs/nnsvg.log"
      fi
      log_msg "nnSVG 完成（$(( $(date +%s) - t0 ))s）"
      ;;
    spagcn)
      log_header "SpaGCN: $SAMPLE 运行开始"
      t0="$(date +%s)"
      "$PYTHON" "$ROOT/src/py_models/run_spaGCN.py" --dataset "$DATASET" \
        "${PY_EXTRA[@]}" --outdir "$OUTDIR" --sample "$SAMPLE" \
        --device "$DEVICE" 2>&1 | tee -a "$OUTDIR/logs/spagcn.log"
      log_msg "SpaGCN 完成（$(( $(date +%s) - t0 ))s）"
      ;;
    spaseg)
      log_header "SpaSEG: $SAMPLE 运行开始"
      t0="$(date +%s)"
      "$PYTHON" "$ROOT/src/py_models/run_spaSEG.py" --dataset "$DATASET" \
        "${PY_EXTRA[@]}" --outdir "$OUTDIR" --sample "$SAMPLE" \
        --device "$DEVICE" 2>&1 | tee -a "$OUTDIR/logs/spaseg.log"
      log_msg "SpaSEG 完成（$(( $(date +%s) - t0 ))s）"
      ;;
  esac
done

# ---------------- 3) 评估 ----------------
if [ "$SKIP_EVAL" -eq 0 ]; then
  log_header "步骤 3: 调用 evaluation.py"
  EVAL_SCRIPT="$ROOT/src/utils/evaluation.py"
  if [ -f "$EVAL_SCRIPT" ]; then
    "$PYTHON" "$EVAL_SCRIPT" --dataset "$DATASET" --outdir "$OUTDIR" \
      --sample "$SAMPLE" --methods "$(echo ${METHODS} | tr ' ' ',')" \
      2>&1 | tee -a "$OUTDIR/logs/evaluation.log" || \
      log_msg "[警告] evaluation.py 返回非零"
  else
    log_msg "[跳过] src/utils/evaluation.py 不存在"
  fi
fi

log_header "models_benchmark 全部完成"
echo "结果目录: $OUTDIR"
echo "日志目录: $OUTDIR/logs"
echo "日志目录: $OUTDIR/logs"
