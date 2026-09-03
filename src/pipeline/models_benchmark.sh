#!/usr/bin/env bash
# =============================================================================
# models_benchmark.sh —— 四方法(SVG)批量对比主控脚本
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
#   - R 依赖 renv 项目锁（在项目根 .Rprofile 自动激活），故所有子命令都在
#     $ROOT 目录下执行。
#   - R 解释器：优先 SVG_RSCRIPT 环境变量，然后环境的 env_R/lib/R/bin/Rscript.exe
#     （conda R 4.4.3，renv 统一管理包），最后回退系统 Rscript。
#   - Python 解释器：优先 SVG_PYTHON 环境变量，然后 env_spatial/python.exe。
#   - 跨解释器传参统一用 Windows 路径（cygpath -w），兼容 Cygwin/MSYS。
#
# 用法示例（在项目根，env_spatial python + renv R 已就绪）：
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
  --cores N            nnSVG 线程数（仅 Linux/macOS fork 并行生效）
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
# 跨原生解释器传参统一转 Windows 路径（Cygwin 不会自动转换）
if command -v cygpath >/dev/null 2>&1; then
  WIN_ROOT="$(cygpath -w "$ROOT")"
else
  WIN_ROOT="$ROOT"
fi
export SVG_REPO_ROOT="$WIN_ROOT"

PYTHON="${SVG_PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for c in "$ROOT/env_spatial/python.exe" "$ROOT/env_spatial/bin/python"; do
    if [ -n "$c" ] && [ -x "$c" ]; then PYTHON="$c"; break; fi
  done
  [ -z "$PYTHON" ] && PYTHON="$(command -v python3 || command -v python || true)"
fi
RSCRIPT="${SVG_RSCRIPT:-}"
if [ -z "$RSCRIPT" ]; then
  for c in "$ROOT/env_R/lib/R/bin/Rscript.exe" \
           "$ROOT/env_R/bin/Rscript" \
           "D:/R-4.4.3/bin/Rscript.exe"; do
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

# ---------------- 解析 run（outdir/sample 以 python src 为准） ----------------
resolved="$("$PYTHON" - "$DATASET" "$OUTDIR_ARG" "$SAMPLE_ARG" "$H5AD" "$SPATIAL" 2>&1 <<'PY' || true
import os, sys
sys.path.insert(0, os.environ["SVG_REPO_ROOT"])
import src
ds, out, sample, h5ad, spatial = (sys.argv[1], sys.argv[2], sys.argv[3],
                                   sys.argv[4], sys.argv[5])
run = src.resolve_run(dataset=(ds or None), outdir=(out or None),
                      sample=(sample or None), h5ad=(h5ad or None),
                      spatial=(spatial or None))
print(run["outdir"])
print(run["sample"])
PY
)"
OUTDIR_WIN="$(printf '%s\n' "$resolved" | sed -n '1p')"
SAMPLE="$(printf '%s\n' "$resolved" | sed -n '2p')"
if [ -z "$OUTDIR_WIN" ] || [ -z "$SAMPLE" ]; then
  echo "[错误] 无法解析 run 配置（dataset=$DATASET）。" >&2
  echo "  - 请检查 src/__init__.py 的 DATASETS 是否含该 key，" >&2
  echo "    或改用 --h5ad/--outdir/--sample 显式指定。" >&2
  printf '  python 输出:\n%s\n' "$resolved" >&2
  exit 2
fi

# shell 工具(mkdir/tee/date...)需 POSIX 路径；原生解释器(R/Python)需 Windows 路径
if command -v cygpath >/dev/null 2>&1; then
  OUTDIR="$(cygpath -u "$OUTDIR_WIN")"
else
  OUTDIR="$OUTDIR_WIN"
fi
to_win() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else echo "$1"; fi
}

mkdir -p "$OUTDIR/logs"
LOG_SUMMARY="$OUTDIR/logs/pipeline_summary.log"

log_msg "outdir = $OUTDIR"
log_msg "sample = $SAMPLE"
log_msg "device = $DEVICE"

# 传递 python 方法的公共参数（h5ad 仅当用户显式给出才传）
PY_EXTRA=()
[ -n "$H5AD" ]    && PY_EXTRA+=(--h5ad "$H5AD")
[ -n "$SPATIAL" ] && PY_EXTRA+=(--spatial "$SPATIAL")

# 组装 R 运行库路径（按优先级：env_R -> D:/R-4.4.3，供 Rscript 加载 DLL）
r_env_path() {
  local p="$PATH"
  [ -d "$ROOT/env_R/Library/bin" ]      && p="$ROOT/env_R/Library/bin:$p"
  [ -d "$ROOT/env_R/bin" ]              && p="$ROOT/env_R/bin:$p"
  [ -d "$ROOT/env_R/lib/R/bin" ]        && p="$ROOT/env_R/lib/R/bin:$p"
  [ -d "D:/R-4.4.3/bin" ]              && p="D:/R-4.4.3/bin:$p"
  printf '%s' "$p"
}
# 在项目根执行 R（触发 renv 自动激活）+ 带上 conda-R 库路径
run_R() {
  ( cd "$ROOT" && PATH="$(r_env_path)" "$RSCRIPT" "$@" )
}

# ---------------- 1) 共同前处理 ----------------
if [ "$SKIP_PRE" -eq 0 ]; then
  log_header "步骤 1/2: 共同前处理 (h5ad_preprocess.py)"
  PREP_SCRIPT="$WIN_ROOT/src/preprocess/h5ad_preprocess.py"
  # shellcheck disable=SC2086
  "$PYTHON" "$PREP_SCRIPT" --dataset "$DATASET" \
    "${PY_EXTRA[@]}" \
    --outdir "$OUTDIR_WIN" --sample-name "$SAMPLE" --methods $METHODS \
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
      run_R "$WIN_ROOT/src/r_models/run_spark.r" \
          "$OUTDIR_WIN/SPARK_X" "$SAMPLE" 2>&1 | tee -a "$OUTDIR/logs/spark.log"
      log_msg "SPARK-X 完成（$(( $(date +%s) - t0 ))s）"
      ;;
    nnsvg)
      log_header "nnSVG: $SAMPLE 运行开始"
      t0="$(date +%s)"
      if [ -n "$CORES" ]; then
        run_R "$WIN_ROOT/src/r_models/run_nnSVG.r" \
            "$OUTDIR_WIN/nnSVG" "$SAMPLE" "$CORES" 2>&1 | tee -a "$OUTDIR/logs/nnsvg.log"
      else
        run_R "$WIN_ROOT/src/r_models/run_nnSVG.r" \
            "$OUTDIR_WIN/nnSVG" "$SAMPLE" 2>&1 | tee -a "$OUTDIR/logs/nnsvg.log"
      fi
      log_msg "nnSVG 完成（$(( $(date +%s) - t0 ))s）"
      ;;
    spagcn)
      log_header "SpaGCN: $SAMPLE 运行开始"
      t0="$(date +%s)"
      "$PYTHON" "$WIN_ROOT/src/py_models/run_spaGCN.py" --dataset "$DATASET" \
        "${PY_EXTRA[@]}" --outdir "$OUTDIR_WIN" --sample "$SAMPLE" \
        --device "$DEVICE" 2>&1 | tee -a "$OUTDIR/logs/spagcn.log"
      log_msg "SpaGCN 完成（$(( $(date +%s) - t0 ))s）"
      ;;
    spaseg)
      log_header "SpaSEG: $SAMPLE 运行开始"
      t0="$(date +%s)"
      "$PYTHON" "$WIN_ROOT/src/py_models/run_spaSEG.py" --dataset "$DATASET" \
        "${PY_EXTRA[@]}" --outdir "$OUTDIR_WIN" --sample "$SAMPLE" \
        --device "$DEVICE" 2>&1 | tee -a "$OUTDIR/logs/spaseg.log"
      log_msg "SpaSEG 完成（$(( $(date +%s) - t0 ))s）"
      ;;
  esac
done

# ---------------- 3) 评估 ----------------
if [ "$SKIP_EVAL" -eq 0 ]; then
  log_header "步骤 3: 调用 evaluation.py"
  EVAL_SCRIPT="$WIN_ROOT/src/utils/evaluation.py"
  if [ -f "$EVAL_SCRIPT" ]; then
    "$PYTHON" "$EVAL_SCRIPT" --dataset "$DATASET" --outdir "$OUTDIR_WIN" \
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
