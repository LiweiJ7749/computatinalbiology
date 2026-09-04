# -*- coding: utf-8 -*-
"""src —— 批量 SVG 方法对比：运行配置解析 / 目录管理 / 共同前处理

把四种方法（SPARK-X、nnSVG、SpaGCN、SpaSEG）从"单一样本、手改路径"的脚本
重构成"一个 run 配置驱动、被 pipeline 批量调用"的形式，所有共同逻辑集中于此：

  1) 初始化 / 常量
       ROOT / DATA_DIR / RESULTS_DIR / METHOD_SUBDIRS / DATASETS ...
  2) 运行配置（run 字典）
       resolve_run()   把 dataset / h5ad / outdir / sample 归一成唯一配置
       ensure_run_dirs()  目录检查与创建（含每个方法的子目录）
  3) 四方法共同前处理（原 src/preprocess/h5ad_preprocess.py 的逻辑上移到这里，
     该脚本只保留一个调用本模块的薄 CLI）
       locate_spatial / load_coords / export_r_format / prepare_spagcn /
       prepare_spaseg / preprocess_run()
  4) 环境探测
       find_python() / find_rscript()

约定（方法名 -> 结果子目录 / 运行脚本 / 输出 CSV 前缀）::

    spark  -> SPARK_X/ , src/r_models/run_spark.r   , SVG_SPARK_<sample>.csv
    nnsvg  -> nnSVG/   , src/r_models/run_nnSVG.r   , SVG_nnSVG_<sample>.csv
    spagcn -> spaGCN/  , src/py_models/run_spaGCN.py, SVG_spaGCN_<sample>.csv
    spaseg -> spaSEG/  , src/py_models/run_spaSEG.py, SVG_spaSEG_<sample>.csv

其中 <sample> 为 run["sample"]（默认取自数据集注册表 DATASETS 的 sample 字段）。

设计要点：模块顶层只 import 标准库与纯 pathlib，anndata/numpy/pandas/scipy 等
重依赖均在函数内延迟导入 —— 模型脚本执行 ``import src`` 时开销极低。
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) 常量路径
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # 项目根
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "local_results"
SRC_DIR = ROOT / "src"

# 解释器候选（Windows .exe 与 Linux/POSIX 均覆盖）：find_python/find_rscript 按序探测。
PYTHON_CANDIDATES = (
    ROOT / "env_spatial" / "bin" / "python",               # Linux conda env
    ROOT / "env_spatial" / "bin" / "python3",
    ROOT / "env_spatial_linux" / "bin" / "python",
    ROOT / "env_spatial" / "python.exe",                   # Windows conda env
)
RSCRIPT_CANDIDATES = (
    ROOT / "env_R" / "bin" / "Rscript",                    # Linux conda R
    ROOT / "env_R_linux" / "bin" / "Rscript",
    ROOT / "env_R" / "lib" / "R" / "bin" / "Rscript.exe",  # Windows conda R
)

# ---------------------------------------------------------------------------
# 2) 方法注册（子目录 / 脚本 / 标准顺序）
# ---------------------------------------------------------------------------
METHOD_SUBDIRS = {"spark": "SPARK_X", "nnsvg": "nnSVG",
                  "spagcn": "spaGCN", "spaseg": "spaSEG"}
ALL_METHODS = ["spark", "nnsvg", "spagcn", "spaseg"]   # 标准执行顺序

# 3D 支持矩阵：目前仅 SPARK-X（locus 为 n x d）支持 3D SVG 检测；
# nnSVG(受限于 BRISC ncol==2)、SpaGCN(2D 邻接)、SpaSEG(2D 网格 CNN) 均为 2D。
METHODS_3D = ["spark"]


def supported_methods(dim) -> list:
    """按空间维度返回可用方法子集（dim=3 时仅 SPARK-X）。"""
    if int(dim or 2) == 3:
        return list(METHODS_3D)
    return list(ALL_METHODS)

MODEL_SCRIPTS = {
    "spark":  SRC_DIR / "r_models" / "run_spark.r",
    "nnsvg":  SRC_DIR / "r_models" / "run_nnSVG.r",
    "spagcn": SRC_DIR / "py_models" / "run_spaGCN.py",
    "spaseg": SRC_DIR / "py_models" / "run_spaSEG.py",
}

# 统一方法颜色（ColorBrewer Set1 前四色，Python 绘图 / evaluation 使用；
# R 脚本中无法 import 本模块，在脚本内硬编码同一组 hex 以保证跨语言一致）。
METHOD_COLORS = {
    "spark":  "#E41A1C",   # 红
    "nnsvg":  "#377EB8",   # 蓝
    "spagcn": "#4DAF4A",   # 绿
    "spaseg": "#984EA3",   # 紫
}
METHOD_LABELS = {
    "spark":  "SPARK-X",
    "nnsvg":  "nnSVG",
    "spagcn": "SpaGCN",
    "spaseg": "SpaSEG",
}

# ---------------------------------------------------------------------------
# 3) 数据集注册表（可被命令行显式参数覆盖）
#   默认从 configs/datasets.json 加载；若文件不存在则回退硬编码。
# ---------------------------------------------------------------------------
CONFIG_DIR = ROOT / "configs"
MODEL_PARAMS_DIR = CONFIG_DIR / "model_params"


def _load_datasets_from_config() -> dict:
    """从 datasets.json 加载数据集注册表，并补齐绝对路径。"""
    import json

    path = CONFIG_DIR / "datasets.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for key, entry in raw.items():
        resolved = {}
        for k, v in entry.items():
            if k in ("h5ad", "spatial") and v and not Path(v).is_absolute():
                resolved[k] = ROOT / v
            else:
                resolved[k] = v
        out[key] = resolved
    return out


# 尝试从 config 加载数据集，无则用硬编码 fallback
_config_datasets = _load_datasets_from_config()
DATASETS = _config_datasets if _config_datasets else {
    "mouse_brain_STARmap": {
        "h5ad": (DATA_DIR / "STARmap" / "mouse_brain_cortex"
                 / "mouse_brain_STARmap_processed.h5ad"),
        "sample": "STARmap_Mouse_Brain",
        "tech": "STARmap",
    },
    "Visium_Mouse_Olfactory_Bulb": {
        "h5ad": DATA_DIR / "Visium" / "Mouse_Olf_Bulb"
                / "Visium_Mouse_Olfactory_Bulb.h5ad",
        "spatial": (DATA_DIR / "Visium" / "Mouse_Olf_Bulb"
                    / "Visium_Mouse_Olfactory_Bulb_spatial.tar.gz"),
        "sample": "Visium_Mouse_Olfactory_Bulb",
        "tech": "Visium",
    },
}


# ---------------------------------------------------------------------------
# 3b) 技术类型（tech type）注册：不同平台 h5ad 数据结构差异的独立参数
# ---------------------------------------------------------------------------
TECH_TYPES = ("STARmap", "Visium", "Visium_HD", "DLPFC",
              "MERFISH", "Slide_seq", "Stereo_seq")


def _normalize_tech(tech) -> str:
    """把 tech 归一成规范名（大小写/连字符/空格/下划线不敏感）。"""
    if not tech:
        return ""
    t = str(tech).strip().lower().replace("-", "_").replace(" ", "_")
    if t in ("star_map", "starmap", "seqfish", "seq_fish"):
        return "STARmap"
    if t in ("visium_hd", "visiumhd", "visium_hd"):
        return "Visium_HD"
    if t in ("merfish", "mer_fish"):
        return "MERFISH"
    if t in ("slide_seq", "slideseq", "slide_seq_v2", "slide_seqv2",
             "slideseqv2", "slideseq_v2"):
        return "Slide_seq"
    if t in ("stereo_seq", "stereoseq", "st_seq"):
        return "Stereo_seq"
    if t == "dlpfc":
        return "DLPFC"
    if "visium" in t:
        return "Visium"
    return str(tech)


def tech_profile(tech: str) -> dict:
    """返回该技术类型的预处理 profile（坐标来源 / counts 来源 / 是否有组织学图像）。

    ``counts`` 字段取值：
      - "X"            : X 即真实 counts（无独立 raw 层，或平台无 raw 层概念）
      - 任意 layer 名   : 如 "raw_count" / "raw_counts"，从 h5ad.layers 取，缺失时回退 X
      - "auto"         : 自动探测（raw_count -> raw_counts -> counts -> X）

    h5ad 结构差异点（见 docs/preprocessing_tech_type.md）：
      - STARmap  : obsm['spatial']，counts 优先 layers['raw_count']（mouse_brain），
                   Zeng2023 AD 复本无 raw 层、X 即整数 counts（回退 X）。
      - Visium   : 坐标在 spatial.tar.gz / tissue_positions.csv，X 即真实 counts，
                   有 H&E 组织学图像。
      - Visium_HD/ DLPFC / Slide_seq : obsm['spatial']，counts 取 X。
      - MERFISH  : obsm['spatial']，counts 取 layers['raw_count']。
      - Stereo_seq : obsm['spatial']（3 列，前两列 x/y），counts 取 layers['raw_counts']。

    未知/未指定 tech 时返回 ``coords="auto", counts="auto"``，回到自动探测逻辑。
    """
    t = _normalize_tech(tech)
    profiles = {
        "Visium":     {"coords": "tissue_positions", "counts": "X", "histology": True},
        "Visium_HD":  {"coords": "obsm_spatial", "counts": "X", "histology": False},
        "STARmap":    {"coords": "obsm_spatial", "counts": "raw_count", "histology": False},
        "DLPFC":      {"coords": "obsm_spatial", "counts": "X", "histology": False},
        "MERFISH":    {"coords": "obsm_spatial", "counts": "raw_count", "histology": False},
        "Slide_seq":  {"coords": "obsm_spatial", "counts": "X", "histology": False},
        "Stereo_seq": {"coords": "obsm_spatial", "counts": "raw_counts", "histology": False},
    }
    return profiles.get(t, {"coords": "auto", "counts": "auto", "histology": False})


def load_model_params(method: str) -> dict:
    """从 configs/model_params/<METHOD>.json 加载模型超参数。

    查找优先级：<METHOD_SUBDIRS[method]>.json > <method>.json > 空字典。
    """
    import json

    candidates = [
        MODEL_PARAMS_DIR / f"{METHOD_SUBDIRS.get(method, method)}.json",
        MODEL_PARAMS_DIR / f"{method}.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# 3a) 日志工具（结构化、带时间戳、可读性好）
# ---------------------------------------------------------------------------
def log_message(msg: str = "", section: str = None, flush: bool = True) -> None:
    """结构化日志输出：时间戳前缀 + 可选节标题。

    Parameters
    ----------
    msg : str
        日志消息文本。空字符串时不输出（常用于只输出 section 标题）。
    section : str, optional
        节标题，非 None 时先输出带分隔线的标题。
    flush : bool
        是否立即刷新输出（默认 True，保证实时可见）。
    """
    from datetime import datetime

    ts = datetime.now().strftime("%H:%M:%S")
    if section is not None:
        print(f"\n{'='*60}", flush=flush)
        print(f"  [{ts}] {section}", flush=flush)
        print(f"{'='*60}", flush=flush)
    if msg:
        print(f"  [{ts}] {msg}", flush=flush)


def log_header(title: str) -> None:
    """输出大节标题（带两侧空行）。"""
    log_message(section=title)


def log_step(step: str, total: int, msg: str = "") -> None:
    """输出步骤标题（如 [1/6] 读取数据）。"""
    label = f"[{step}/{total}] {msg}" if msg else f"[{step}/{total}]"
    log_message(section=label)


# ---------------------------------------------------------------------------
# 4) run 配置解析 + 目录管理
# ---------------------------------------------------------------------------
def _resolve_data_path(p):
    """把数据集注册的相对路径归一为绝对路径（相对项目根）。"""
    p = Path(p)
    if not p.is_absolute():
        cand = ROOT / p
        return cand if cand.exists() else p
    return p


def resolve_run(dataset=None, h5ad=None, spatial=None, outdir=None,
                sample=None, tech=None, methods=None) -> dict:
    """把一个 run 的意图归一成唯一的配置字典。

    参数可混用：命中 ``dataset`` 注册表后，显式传入的 h5ad/spatial/outdir/sample/
    tech 会覆盖注册表默认值。返回字典字段：
      dataset / h5ad / spatial / outdir / sample / tech / dim / methods /
      method_dirs（3D 数据集额外含 slices / n_slices / reconstruction / z_source）
    """
    methods = list(methods or ALL_METHODS)
    bad = [m for m in methods if m not in METHOD_SUBDIRS]
    if bad:
        raise ValueError(f"未知方法 {bad}，可选: {list(METHOD_SUBDIRS)}")
    methods = [m for m in ALL_METHODS if m in methods]   # 规范顺序

    reg = DATASETS.get(dataset, {}) if dataset else {}

    # --- 维度 / 3D 重建信息 ---
    dim = int(reg.get("dim") or 2)
    slices = (reg.get("slices") or []) if not h5ad else []
    if isinstance(slices, (list, tuple)):
        slices = [_resolve_data_path(p) for p in slices]

    # --- h5ad（Slide-seq 多切片无单一 h5ad，允许为空） ---
    h5ad_path = Path(h5ad) if h5ad else reg.get("h5ad")
    if h5ad_path is not None:
        h5ad_path = _resolve_data_path(h5ad_path)
    if h5ad_path is None and not slices:
        raise ValueError("无法确定 h5ad：请提供 --dataset 或 --h5ad，"
                         f"已知数据集: {list(DATASETS)}")
    if h5ad_path is not None:
        h5ad_path = Path(h5ad_path)

    # --- 3D 方法过滤（仅 SPARK-X 支持 3D） ---
    if dim == 3:
        kept = [m for m in methods if m in supported_methods(3)]
        skipped = [m for m in methods if m not in supported_methods(3)]
        if skipped:
            log_message(f"dim=3：仅支持 3D 方法 {kept}，跳过 2D 方法 {skipped} "
                        f"(nnSVG/SpaGCN/SpaSEG 仅支持 2D)")
        methods = kept

    # --- spatial / outdir / sample ---
    spatial_path = Path(spatial) if spatial else reg.get("spatial")
    out_root = Path(outdir) if outdir else reg.get(
        "outdir", RESULTS_DIR / (dataset or (h5ad_path.stem if h5ad_path else "run")))
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    sample_name = (sample or reg.get("sample")
                   or (h5ad_path.stem if h5ad_path else dataset) or "sample")
    tech_name = _normalize_tech(tech if tech is not None else reg.get("tech"))

    run = {
        "dataset": dataset,
        "h5ad": h5ad_path,
        "spatial": spatial_path,
        "outdir": out_root,
        "sample": sample_name,
        "tech": tech_name,
        "dim": dim,
        "methods": methods,
        "method_dirs": {m: out_root / METHOD_SUBDIRS[m] for m in methods},
    }
    if dim == 3:
        run["slices"] = slices
        run["n_slices"] = reg.get("n_slices")
        run["reconstruction"] = reg.get("reconstruction")
        run["z_source"] = reg.get("z_source")
    return run


def ensure_run_dirs(run: dict, methods=None) -> None:
    """检查并创建 run 输出目录与每个方法的子目录。"""
    methods = list(methods) if methods is not None else list(run["methods"])
    run["outdir"].mkdir(parents=True, exist_ok=True)
    for m in methods:
        run["method_dirs"][m].mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 5) 环境探测（供 pipeline .sh 或交互式调用）
# ---------------------------------------------------------------------------
def find_python():
    """返回可用的 python 解释器路径（优先本项目 env_spatial，其次 PATH）。"""
    for c in PYTHON_CANDIDATES:
        if c.exists():
            return str(c)
    for name in ("python3", "python"):
        p = shutil.which(name)
        if p:
            return p
    return None


def find_rscript():
    """返回可用的 Rscript 路径（优先本项目 env_R，其次 PATH）。"""
    for c in RSCRIPT_CANDIDATES:
        if c.exists():
            return str(c)
    return shutil.which("Rscript")


def external_src_dir(name):
    """返回外部方法源码目录（如 ``SpaGCN_src`` / ``SpaSEG_src``）。

    这些源码为纯 Python、平台无关，Windows 与 Linux 共用同一份；默认优先
    ``env_spatial``、其次 ``env_spatial_linux``，也可用环境变量 ``SVG_EXT_DIR``
    覆盖（HPC/Linux 上无需把解释器路径写死进模型脚本）。
    """
    base = os.environ.get("SVG_EXT_DIR")
    if base:
        return Path(base) / name
    for cand in (ROOT / "env_spatial", ROOT / "env_spatial_linux"):
        p = cand / name
        if p.exists():
            return p
    return ROOT / "env_spatial" / name


# ---------------------------------------------------------------------------
# 6) 共同前处理（延迟导入 anndata/scipy 等重依赖）
# ---------------------------------------------------------------------------
def _read_tissue_positions(source: Path):
    """读取 tissue_positions.csv（或其所在 spatial.tar.gz），返回以 barcode 为索引的 DataFrame。"""
    import io

    import pandas as pd

    if source.suffix in (".gz", ".tgz") or ".tar" in source.name:
        import tarfile

        with tarfile.open(source, "r:*") as tar:
            member = next((m for m in tar.getmembers()
                           if m.name.endswith("tissue_positions.csv")), None)
            if member is None:
                raise FileNotFoundError(f"{source} 内未找到 tissue_positions.csv")
            raw = tar.extractfile(member).read()
        tp = pd.read_csv(io.BytesIO(raw))
    else:
        tp = pd.read_csv(source)
    if "barcode" not in tp.columns:
        raise ValueError("tissue_positions.csv 缺少 barcode 列")
    return tp.set_index("barcode")


def _read_hires_image(source: Path):
    """从 spatial.tar.gz 提取 tissue_hires_image.png（BGR 数组），无则 None。"""
    try:
        import tarfile

        import numpy as np

        with tarfile.open(source, "r:*") as tar:
            member = next((m for m in tar.getmembers()
                           if m.name.endswith("tissue_hires_image.png")), None)
            if member is None:
                return None
            data = tar.extractfile(member).read()
        import cv2

        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def locate_spatial(h5ad_path: Path, spatial_arg) -> tuple:
    """按优先级返回 (tissue_positions.csv 路径 | None, spatial.tar.gz 路径 | None)。"""
    if spatial_arg is not None:
        s = Path(spatial_arg)
        if s.exists():
            return (s, None) if s.suffix.lower() == ".csv" else (None, s)
    # 已解压的全局坐标表
    extracted = RESULTS_DIR / "spatial" / "tissue_positions.csv"
    if extracted.exists():
        return extracted, None
    # h5ad 同目录下的 *_spatial.tar.gz（找不到路径时跳过）
    if h5ad_path is not None:
        tgz = h5ad_path.with_name(h5ad_path.name.replace(".h5ad", "_spatial.tar.gz"))
        if tgz.exists():
            return None, tgz
    return None, None


def load_coords(h5ad, spatial_arg, tech=None, h5ad_path=None, dim=2) -> "pd.DataFrame":
    """读入坐标并与 h5ad 的 obs.index 对齐，返回 barcode 为索引的 DataFrame。

    按技术类型选择坐标来源（tech_profile 的 ``coords`` 字段）：
      - Visium   : tissue_positions.csv（来自 spatial.tar.gz），x/y 为全分辨率像素坐标
      - STARmap/Visium_HD : obsm['spatial']（前两列 -> x/y）
      - Stereo_seq(dim=3): obsm['spatial'] 三列 -> x/y/z
    未指定 tech 时回到自动探测（先尝试 tissue_positions，再回退 obsm['spatial']）。

    ``dim`` 控制 obsm 坐标取几列（3 维数据取 x/y/z）。

    ``h5ad_path`` 用于在未显式给 spatial 时自动定位同目录 *_spatial.tar.gz；
    优先级：显式 h5ad_path > anndata.filename（部分版本 read_h5ad 不写 filename）。
    """
    import numpy as np
    import pandas as pd

    profile = tech_profile(tech)
    coords_src = profile["coords"]

    if h5ad_path is not None:
        h5ad_path = Path(h5ad_path)
    elif getattr(h5ad, "filename", None):
        h5ad_path = Path(h5ad.filename)
    else:
        h5ad_path = None
    tp = None
    if coords_src in ("auto", "tissue_positions"):
        csv_path, tgz_path = locate_spatial(h5ad_path, spatial_arg)
        if csv_path is not None:
            tp = _read_tissue_positions(csv_path)
        elif tgz_path is not None:
            tp = _read_tissue_positions(tgz_path)

    if tp is not None and len(tp) > 0:
        if "in_tissue" in tp.columns:
            tp = tp[tp["in_tissue"] == 1]
        tp = tp.loc[[b for b in h5ad.obs.index if b in tp.index]]
        if "pxl_col_in_fullres" in tp.columns and "pxl_row_in_fullres" in tp.columns:
            tp = tp.rename(columns={"pxl_col_in_fullres": "x",
                                    "pxl_row_in_fullres": "y"})
        if len(tp) > 0:
            return tp
        if coords_src == "tissue_positions":
            raise FileNotFoundError("技术类型 Visium 未在 tissue_positions 中找到有效坐标")

    if coords_src in ("auto", "obsm_spatial"):
        if "spatial" in h5ad.obsm and h5ad.obsm["spatial"] is not None:
            coords = np.asarray(h5ad.obsm["spatial"], dtype=float)
            if int(dim or 2) == 3:
                if coords.shape[1] < 3:
                    raise ValueError("dim=3 要求 obsm['spatial'] 至少 3 列 (x/y/z)")
                return pd.DataFrame(coords[:, :3], index=h5ad.obs.index,
                                    columns=["x", "y", "z"])
            return pd.DataFrame(coords[:, :2], index=h5ad.obs.index,
                                columns=["x", "y"])
        if coords_src == "obsm_spatial":
            raise FileNotFoundError("技术类型要求 obsm['spatial']，但 h5ad 中缺失")

    raise FileNotFoundError("找不到坐标来源"
                            "(tissue_positions.csv / spatial.tar.gz / obsm['spatial'])")


def _raw_counts_matrix(h5ad, tech=None):
    """返回喂给 R 方法的"真实 counts"矩阵 (spots x genes)。

    按技术类型决定 counts 来源（tech_profile 的 ``counts`` 字段）：
      - "X"          : X 即原始 counts（Visium / Visium_HD / DLPFC / Slide_seq）
      - 具体 layer 名  : 如 "raw_count"(STARmap/MERFISH)、"raw_counts"(Stereo_seq)，
                        从 h5ad.layers 取，缺失时回退 X 并告警。
      - "auto"       : 自动探测（raw_count -> raw_counts -> counts -> X）。
    """
    import numpy as np
    from scipy import sparse

    counts_src = tech_profile(tech)["counts"]
    if counts_src == "X":
        X = h5ad.X
        log_message("表达矩阵: X 即真实 counts（技术类型无独立 raw 层）")
    elif counts_src == "auto":
        # 按常见命名顺序探测 raw 层
        X = None
        for name in ("raw_count", "raw_counts", "counts"):
            if name in h5ad.layers and h5ad.layers[name] is not None:
                X = h5ad.layers[name]
                log_message(f"表达矩阵: 使用 layers['{name}']（真实 counts）")
                break
        if X is None:
            X = h5ad.X
            log_message("表达矩阵: 无 raw 层，回退 X（注意：可能不是原始 counts）")
    else:
        # 指定了某个 layer 名
        if counts_src in h5ad.layers and h5ad.layers[counts_src] is not None:
            X = h5ad.layers[counts_src]
            log_message(f"表达矩阵: 使用 layers['{counts_src}']（真实 counts）")
        else:
            X = h5ad.X
            log_message(f"表达矩阵: 无 layers['{counts_src}']，回退 X（注意：可能不是原始 counts）")
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    return X.astype(np.float64)


def _write_r_format(counts_genes_spots, gene_names, barcodes,
                    coords_df, outdir: Path, field=None) -> None:
    """把已对齐的数据写为 R 方法可读的 counts.mtx / genes.csv / barcodes.csv / location.csv。

    ``counts_genes_spots`` 为 genes x spots 稀疏矩阵；``coords_df`` 以 barcode 为索引、
    列名为坐标（2D: x,y；3D: x,y,z）。``field`` 取值 'integer'/'real'（mmwrite 精度），
    默认 None 时整数数据用 'integer'、浮点归一化数据用 'real'。
    """
    import pandas as pd
    from scipy import io as sio

    outdir.mkdir(parents=True, exist_ok=True)
    if field is None:
        field = "integer" if counts_genes_spots.dtype.kind in "iu" else "real"
    sio.mmwrite(str(outdir / "counts.mtx"), counts_genes_spots, field=field)
    pd.Series(gene_names).to_csv(outdir / "genes.csv", index=False, header=False)
    pd.Series(barcodes).to_csv(outdir / "barcodes.csv", index=False, header=False)
    coords_df.to_csv(outdir / "location.csv", index_label="barcode")
    log_message(f"R 格式导出: {counts_genes_spots.shape[0]} genes "
                f"x {counts_genes_spots.shape[1]} spots "
                f"[{','.join(coords_df.columns)}] -> {outdir}")


def export_r_format(h5ad, tp, outdir: Path, tech=None, dim=2) -> None:
    """导出 SPARK-X / nnSVG 可直接读取的 counts.mtx / genes.csv / barcodes.csv / location.csv。

    counts.mtx 为 genes x spots（行为基因）。counts 来源按技术类型由
    _raw_counts_matrix 决定（STARmap 取 raw_count 层；Visium 取 X）。
    ``dim=3`` 时 location.csv 写三列 (x, y, z)。
    """
    import pandas as pd

    X = _raw_counts_matrix(h5ad, tech=tech)
    gene_names = list(h5ad.var.index.astype(str))

    barcodes = [b for b in h5ad.obs.index if b in tp.index]
    keep_idx = [i for i, b in enumerate(h5ad.obs.index) if b in tp.index]
    X = X[keep_idx, :]                     # spots x genes（与坐标对齐）
    tp_aligned = tp.loc[barcodes]
    X = X.T.tocsr()                        # genes x spots（R 方法约定）

    coord_cols = [c for c in ("x", "y", "z") if c in tp_aligned.columns]
    loc = tp_aligned[coord_cols]
    _write_r_format(X, gene_names, barcodes, loc, outdir)


def prepare_spagcn(h5ad, tp, img, outdir: Path, sample_name: str) -> None:
    """为 SpaGCN 准备输入：写 <sample>_spaGCN.h5ad + 坐标列 + 组织学图像。

    SpaGCN 官方要求的 obs 坐标列（对应 positions.txt 的 6 列）：
      x_array = array_row,  y_array = array_col
      x_pixel = pxl_row_in_fullres, y_pixel = pxl_col_in_fullres
    组织学图像可选（histology=False 时 SpaGCN 仅用空间坐标建邻接矩阵）。
    """
    import numpy as np
    import anndata as ad

    outdir.mkdir(parents=True, exist_ok=True)
    adata = _copy_h5ad(h5ad)
    barcodes = list(adata.obs.index)
    tp_aligned = tp.loc[[b for b in barcodes if b in tp.index]]

    x = tp_aligned["x"].astype(float).values
    y = tp_aligned["y"].astype(float).values
    adata.obs["x_pixel"] = x
    adata.obs["y_pixel"] = y
    adata.obs["x"] = x
    adata.obs["y"] = y
    # obsm['spatial'] 供 run_spaGCN 在 obs 无 x/y 时回退读取（Visium 原 h5ad 无此键）
    adata.obsm["spatial"] = np.column_stack([x, y]).astype(np.float32)

    if "array_row" in tp_aligned.columns and "array_col" in tp_aligned.columns:
        adata.obs["x_array"] = tp_aligned["array_row"].astype(int).values
        adata.obs["y_array"] = tp_aligned["array_col"].astype(int).values
    else:
        adata.obs["x_array"] = x.astype(int)
        adata.obs["y_array"] = y.astype(int)

    h5ad_out = outdir / f"{sample_name}_spaGCN.h5ad"
    adata.write(h5ad_out)
    if img is not None:
        import cv2

        cv2.imwrite(str(outdir / "histology.png"), img)
        log_message(f"SpaGCN 准备: 坐标列 + 组织学图像 -> {h5ad_out} / histology.png")
    else:
        log_message(f"SpaGCN 准备: 坐标列（无组织学图像，可用 histology=False）-> {h5ad_out}")
    del adata


def prepare_spaseg(h5ad, tp, outdir: Path, sample_name: str) -> None:
    """为 SpaSEG 准备输入：写 <sample>_spaSEG.h5ad + obsm['spatial'] 坐标。"""
    import numpy as np

    outdir.mkdir(parents=True, exist_ok=True)
    adata = _copy_h5ad(h5ad)
    barcodes = list(adata.obs.index)
    tp_aligned = tp.loc[[b for b in barcodes if b in tp.index]]

    adata.obsm["spatial"] = np.column_stack([
        tp_aligned["x"].astype(float).values,
        tp_aligned["y"].astype(float).values,
    ]).astype(np.float32)

    h5ad_out = outdir / f"{sample_name}_spaSEG.h5ad"
    adata.write(h5ad_out)
    log_message(f"SpaSEG 准备: obsm['spatial'] -> {h5ad_out}")
    del adata


def _copy_h5ad(h5ad):
    """复制一个轻量 AnnData（避免污染读入的原始对象），保留 layers（如 raw_count）。"""
    import anndata as ad

    return ad.AnnData(
        X=h5ad.X.copy(),
        obs=h5ad.obs.copy(),
        var=h5ad.var.copy(),
        layers={k: v.copy() for k, v in h5ad.layers.items()},
        obsm={k: v.copy() for k, v in h5ad.obsm.items()},
        uns={k: v for k, v in h5ad.uns.items()},
    )


def histology_source(run: dict):
    """返回可提取组织学图像的 spatial.tar.gz 路径（无则 None）。"""
    sp = run.get("spatial")
    if sp is not None:
        s = Path(sp)
        if s.exists() and s.suffix.lower() != ".csv":
            return s
    h5ad_path = run["h5ad"]
    tgz = h5ad_path.with_name(h5ad_path.name.replace(".h5ad", "_spatial.tar.gz"))
    return tgz if tgz.exists() else None


def _load_slideseq_3d(slice_paths, z_spacing=1.0):
    """把 Slide-seq 的连续 2D 切片堆叠为 3D（2D→3D 重建）。

    返回 (counts: spots x genes csr, gene_names, barcodes, coords_df)。
    - z 轴 = 切片序号 * z_spacing（注册表 reconstruction="stack_2d_slices"）。
    - 基因取各切片并集（首见顺序），缺失切片补 0。
    - barcode 加切片前缀保证跨切片唯一。
    """
    import numpy as np
    import pandas as pd
    import anndata as ad
    from scipy import sparse

    slices_coords, slices_X, slices_genes, all_barcodes = [], [], [], []
    seen_genes = set()
    union_genes = []

    for i, p in enumerate(slice_paths):
        p = Path(p)
        a = ad.read_h5ad(p)
        coords2 = np.asarray(a.obsm["spatial"], dtype=float)[:, :2]
        z = np.full((coords2.shape[0], 1), float(i) * z_spacing)
        slices_coords.append(np.hstack([coords2, z]))

        Xi = a.X if sparse.issparse(a.X) else sparse.csr_matrix(a.X)
        slices_X.append(Xi)

        gs = [str(g) for g in a.var.index]
        slices_genes.append(gs)
        for g in gs:
            if g not in seen_genes:
                seen_genes.add(g)
                union_genes.append(g)
        all_barcodes.extend([f"S{i}_{b}" for b in a.obs.index])
        log_message(f"切片 {p.name}: {Xi.shape[0]} spots x {Xi.shape[1]} genes")
        del a

    # 按基因并集对齐各切片列，再纵向堆叠
    gene_to_idx = {g: k for k, g in enumerate(union_genes)}
    aligned = []
    for Xi, gs in zip(slices_X, slices_genes):
        mapper = np.array([gene_to_idx[g] for g in gs], dtype=np.int32)
        new_idx = mapper[Xi.indices]
        aligned.append(sparse.csr_matrix(
            (Xi.data, new_idx, Xi.indptr),
            shape=(Xi.shape[0], len(union_genes))))
    counts = sparse.vstack(aligned, format="csr")            # spots x genes

    coords = np.vstack(slices_coords)
    coords_df = pd.DataFrame(coords, columns=["x", "y", "z"], index=all_barcodes)
    return counts, union_genes, all_barcodes, coords_df


def _preprocess_stereo3d(run):
    """Stereo-seq 单文件 3D（obsm['spatial'] 三列 x/y/z）。"""
    import anndata as ad

    adata = ad.read_h5ad(run["h5ad"])
    log_message(f"shape = {adata.shape} (spots x genes)")
    tp = load_coords(adata, run.get("spatial"), tech=run.get("tech"),
                     h5ad_path=run["h5ad"], dim=3)
    log_message(f"对齐坐标后 spots = {len(tp)}")
    export_r_format(adata, tp, run["method_dirs"]["spark"],
                    tech=run.get("tech"), dim=3)
    del adata


def _preprocess_run_3d(run, methods):
    """3D 预处理：仅生成 SPARK-X（唯一支持 3D 的方法）所需输入。"""
    methods = [m for m in methods if m in supported_methods(3)]
    if not methods:
        raise ValueError("dim=3 无可用方法（仅 SPARK-X 支持 3D）")
    ensure_run_dirs(run, methods=methods)
    tech = run.get("tech")

    if tech == "Stereo_seq":
        _preprocess_stereo3d(run)
    elif tech == "Slide_seq":
        counts, genes, barcodes, coords_df = _load_slideseq_3d(run["slices"])
        outdir = run["method_dirs"]["spark"]
        _write_r_format(counts.T.tocsr(), genes, barcodes, coords_df, outdir)
    else:
        raise ValueError(f"dim=3 但技术类型 {tech!r} 尚无 3D 预处理实现")
    log_message("3D 前处理完成", section="完成")
    return run


def preprocess_run(run: dict, methods=None) -> dict:
    """对一次 run 生成方法所需的全部输入，并建立目录。

    - 2D：R 方法（spark/nnsvg）写 counts.mtx / genes.csv / barcodes.csv / location.csv；
      SpaGCN / SpaSEG 写 <sample>_spaGCN.h5ad / <sample>_spaSEG.h5ad（含坐标）。
    - 3D：仅 SPARK-X（Slide-seq 堆叠重建 / Stereo-seq 三列坐标）。
    """
    import anndata as ad

    methods = [m for m in (methods or run["methods"])]

    if int(run.get("dim") or 2) == 3:
        return _preprocess_run_3d(run, methods)

    ensure_run_dirs(run, methods=methods)

    log_header(f"共同前处理: {run.get('dataset', '')}")

    h5ad_path = run["h5ad"]
    if not h5ad_path.exists():
        raise FileNotFoundError(f"h5ad 不存在: {h5ad_path}")
    log_message(f"读取 h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    log_message(f"shape = {adata.shape} (spots x genes)")

    tech = run.get("tech")
    profile = tech_profile(tech)
    log_message(f"技术类型 = {tech or 'auto'} "
                f"(coords={profile['coords']}, counts={profile['counts']}, "
                f"histology={profile['histology']})")

    tp = load_coords(adata, run.get("spatial"), tech=tech, h5ad_path=h5ad_path)
    log_message(f"对齐坐标后 spots = {len(tp)}")

    img = None
    if "spagcn" in methods:
        src = histology_source(run)
        if src is not None:
            img = _read_hires_image(src)
            if img is not None:
                log_message("已提取组织学图像 (tissue_hires_image.png)")

    for m in methods:
        outdir = run["method_dirs"][m]
        if m in ("spark", "nnsvg"):
            export_r_format(adata, tp, outdir, tech=tech)
        elif m == "spagcn":
            prepare_spagcn(adata, tp, img, outdir, run["sample"])
        elif m == "spaseg":
            prepare_spaseg(adata, tp, outdir, run["sample"])
    del adata
    log_message("前处理完成", section="完成")
    return run


# 兼容：一些旧脚本/文档把它当作 dict 使用
SUBDIRS = METHOD_SUBDIRS


def __getattr__(name):
    """对旧引用给出更友好的错误（避免 import 时即失败）。"""
    raise AttributeError(
        f"src 模块没有属性 {name!r}。批量重构后请使用 "
        "src.resolve_run / src.ensure_run_dirs / src.preprocess_run。")


if __name__ == "__main__":      # python -m src 无意义时的提示（防止静默）
    print(__doc__)
