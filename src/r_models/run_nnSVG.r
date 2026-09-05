# ============================================================
# nnSVG 方法：SVG 检测（批量化）
# 输入：<data_dir> 下的中间文件（由 src/preprocess/h5ad_preprocess.py 生成）
#       counts.mtx (genes x spots), genes.csv, barcodes.csv, location.csv
# 输出：SVG_nnSVG_<sample>.csv + ggplot2 展示图
#       nnSVG_spe.rds (SpatialExperiment 中间对象，便于调试)
#
# 并行策略（仅用 nnSVG 公开的 BPPARAM 参数，不改包源码，保证 HPC 可用）：
#   - Windows：诊断确认 PSOCK/SnowParam 在此环境不稳定（BRISC 的 order/neighbor
#     对象经 socket 导出到 worker 时崩溃或丢失），故强制 SerialParam（串行）。
#   - Linux/macOS (HPC)：自动用 MulticoreParam(n_threads)（fork 并行），逐基因
#     并行跑 BRISC，可获得近线性加速。
# 用法: Rscript run_nnSVG.r <data_dir> [sample] [n_threads]
#       sample 默认取 data_dir 上级目录名（如 mouse_brain_STARmap）
# ============================================================
suppressPackageStartupMessages({
  library(nnSVG)
  library(SpatialExperiment)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(scran)
  library(Matrix)
  library(ggplot2)
  library(BiocParallel)
})

# 日志工具（与 run_spark.r 一致）
log_msg <- function(msg) {
  ts <- format(Sys.time(), "%H:%M:%S")
  cat(sprintf("  [%s] %s\n", ts, msg))
}
log_header <- function(title) {
  cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
  ts <- format(Sys.time(), "%H:%M:%S")
  cat(sprintf("  [%s] %s\n", ts, title))
  cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")
}
log_step <- function(i, total, msg) {
  log_header(sprintf("[%d/%d] %s", i, total, msg))
}

# ---------- 路径 ----------
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else
  "F:/computatinalbiology/results/local_results/mouse_brain_STARmap/nnSVG"
sample <- if (length(args) >= 2) args[2] else
  basename(dirname(normalizePath(data_dir, winslash = "/")))
n_threads <- if (length(args) >= 3) as.integer(args[3]) else {
  # 默认核数按"核数 ∩ 可用内存"取较小值，避免在小内存机器(如 WSL)上因
  # 每个 fork worker 约 1~1.5GB 而 OOM 崩溃；HPC 上可显式传 n_threads 覆盖。
  cores <- if (!is.na(parallel::detectCores())) parallel::detectCores() else 4
  mem_avail_gb <- NA_real_
  m <- tryCatch(readLines("/proc/meminfo", warn = FALSE), error = function(e) character())
  ia <- grep("^MemAvailable:", m)
  if (length(ia) == 1) {
    mem_avail_gb <- as.numeric(gsub("[^0-9]", "", m[ia])) / 1048576
  }
  mem_cap <- if (is.na(mem_avail_gb)) cores else floor(mem_avail_gb / 1.5)
  as.integer(max(1L, min(cores, mem_cap)))
}
out_dir <- data_dir

log_header(sprintf("nnSVG: %s", sample))
log_msg(sprintf("数据目录: %s", data_dir))
log_msg(sprintf("样本标签: %s", sample))
log_msg(sprintf("平台: %s | 请求线程数: %d", .Platform$OS.type, n_threads))

counts <- readMM(file.path(data_dir, "counts.mtx"))
genes  <- read.csv(file.path(data_dir, "genes.csv"),  header = FALSE, stringsAsFactors = FALSE)[, 1]
barcodes <- read.csv(file.path(data_dir, "barcodes.csv"), header = FALSE, stringsAsFactors = FALSE)[, 1]
loc_df <- read.csv(file.path(data_dir, "location.csv"), stringsAsFactors = FALSE, row.names = 1)

rownames(counts) <- genes
colnames(counts) <- barcodes

# counts 为 dgCMatrix (genes x spots)
counts <- as(counts, "CsparseMatrix")

# 坐标对齐（用像素坐标，与官方教程一致）
loc <- loc_df[barcodes, c("x", "y")]
colnames(loc) <- c("pxl_col_in_fullres", "pxl_row_in_fullres")
rownames(loc) <- barcodes

cat(sprintf("counts 维度: %d genes x %d spots\n", nrow(counts), ncol(counts)))

# ---------- 构建 SpatialExperiment 对象 ----------
spe <- SpatialExperiment(
  assays = list(counts = counts),
  rowData = DataFrame(gene_id = genes, gene_name = genes),
  spatialCoords = as.matrix(loc)
)
rownames(spe) <- genes

# ---------- 过滤基因 ----------
# filter_genes_pcspots / ncounts 可配置（默认 0.01% / 3）：Visium HD 等稀疏平台
# bin 极小，0.5% 会滤掉 99% 基因；由 models_benchmark.sh 从 configs/run_params.json
# 读取并注入环境变量 NNSVG_PCSPOTS / NNSVG_NCOUNTS。
pcspots <- as.numeric(Sys.getenv("NNSVG_PCSPOTS", "0.01"))
ncounts <- as.numeric(Sys.getenv("NNSVG_NCOUNTS", "3"))
log_msg(sprintf("filter_genes_pcspots = %s%% | ncounts = %s", pcspots, ncounts))
spe <- filter_genes(spe, filter_genes_ncounts = ncounts,
                    filter_genes_pcspots = pcspots, filter_mito = TRUE)
log_msg(sprintf("过滤后基因数: %d", nrow(spe)))

# 剔除 library size = 0 的 spot：稀疏平台（如 Visium HD 的微小 bin）过滤后部分
# spot 在保留基因上总 counts 为 0，会使 logNormCounts 报
# "size factors should be positive"，故先剔除这些 spot。
libsize <- Matrix::colSums(counts(spe))
zero_spots <- sum(libsize == 0)
if (zero_spots > 0) {
  log_msg(sprintf("剔除 library size = 0 的 spot: %d 个", zero_spots))
  spe <- spe[, libsize > 0]
}

# ---------- 计算 logcounts ----------
spe <- computeLibraryFactors(spe)
spe <- logNormCounts(spe)
log_msg(sprintf("assay names: %s", paste(assayNames(spe), collapse = ", ")))

# 保存中间对象便于调试
rds_path <- file.path(out_dir, "nnSVG_spe.rds")
saveRDS(spe, rds_path)
log_msg(sprintf("已保存中间对象: %s", rds_path))

# ---------- 运行 nnSVG ----------
# 平台自适应并行（nnSVG 官方公开 BPPARAM 参数；不改包源码，HPC 可复现）：
#   Windows -> SerialParam（本机诊断: PSOCK 并行不稳定，见头注释）
#   POSIX(HPC) + n_threads>1 -> MulticoreParam(fork)，逐基因并行 BRISC
if (.Platform$OS.type == "windows") {
  log_msg("平台 Windows: 使用串行（PSOCK 并行不稳定，见头注释）")
  bp_param <- BiocParallel::SerialParam()
} else if (n_threads > 1) {
  log_msg(sprintf("平台 POSIX: 使用 MulticoreParam fork 并行, %d workers", n_threads))
  bp_param <- BiocParallel::MulticoreParam(workers = n_threads)
} else {
  log_msg("平台 POSIX: 串行")
  bp_param <- BiocParallel::SerialParam()
}
set.seed(123)
log_step(1, 2, "运行 nnSVG")
t0 <- Sys.time()
spe <- nnSVG(spe, BPPARAM = bp_param, verbose = FALSE)
t1 <- Sys.time()
log_msg(sprintf("nnSVG 运行耗时: %.2f 分钟", as.numeric(difftime(t1, t0, units = "mins"))))

# ---------- 汇总结果 ----------
rd <- as.data.frame(rowData(spe))
res_df <- data.frame(
  gene        = rd$gene_name,
  LR_stat     = rd$LR_stat,
  pval        = rd$pval,
  padj        = rd$padj,
  prop_sv     = rd$prop_sv,
  sigma_sq    = rd$sigma.sq,
  tau_sq      = rd$tau.sq,
  stringsAsFactors = FALSE
)
res_df <- res_df[order(res_df$padj, -res_df$LR_stat, na.last = TRUE), ]
res_df$rank <- seq_len(nrow(res_df))

# ---------- 保存 CSV ----------
csv_path <- file.path(out_dir, sprintf("SVG_nnSVG_%s.csv", sample))
write.csv(res_df, csv_path, row.names = FALSE)
log_msg(sprintf("已保存结果: %s", csv_path))
log_msg(sprintf("显著 SVG (padj < 0.05): %d / %d",
            sum(res_df$padj < 0.05, na.rm = TRUE), nrow(res_df)))

# ---------- 保存跨方法统一的排名 CSV（列固定: gene, stat, pval, padj, rank）----------
# stat 取 LR_stat（越大越显著）；p=0 下溢时封顶避免 Inf 导致跨语言读入不兼容。
eval_df <- data.frame(
  gene = res_df$gene,
  stat = res_df$LR_stat,
  pval = res_df$pval,
  padj = res_df$padj,
  rank = res_df$rank,
  stringsAsFactors = FALSE
)
eval_df <- eval_df[order(eval_df$padj, -eval_df$stat, na.last = TRUE), ]
eval_df$rank <- seq_len(nrow(eval_df))
eval_path <- file.path(out_dir, sprintf("SVG_nnSVG_%s_rank.csv", sample))
write.csv(eval_df, eval_path, row.names = FALSE)
log_msg(sprintf("已保存统一排名 CSV: %s", eval_path))

# ---------- 保存运行时间（JSON，供 evaluation 汇总效率指标）----------
rt_path <- file.path(out_dir, "runtime.json")
writeLines(sprintf('{"method":"nnsvg","sample":"%s","wall_seconds":%.2f}',
                   sample, as.numeric(difftime(t1, t0, units = "secs"))), rt_path)
log_msg(sprintf("已保存运行时间: %s", rt_path))

# ---------- 展示图 ----------
# 1) Top SVG 的 -log10(padj) 柱状图
top_n <- 30
top_df <- head(res_df, top_n)
top_df$gene <- factor(top_df$gene, levels = rev(top_df$gene))

p1 <- ggplot(top_df, aes(x = gene, y = -log10(padj))) +
  geom_col(fill = "darkgreen") +
  coord_flip() +
  labs(title = sprintf("nnSVG: Top SVG genes (%s)", sample),
       x = "Gene", y = "-log10(adjusted P-value)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5))

png1 <- file.path(out_dir, sprintf("SVG_nnSVG_%s_top.png", sample))
ggsave(png1, p1, width = 8, height = 6, dpi = 150)
log_msg(sprintf("已保存展示图: %s", png1))

# 2) Top-1 SVG 的空间表达图（按 res_df 排序取 rank 1，并映射回 spe 行）
ix_name <- as.character(res_df$gene[1])
ix <- which(rowData(spe)$gene_name == ix_name)
if (length(ix) == 0) ix <- 1L
df <- data.frame(
  x = spatialCoords(spe)[, "pxl_col_in_fullres"],
  y = spatialCoords(spe)[, "pxl_row_in_fullres"],
  expr = as.numeric(counts(spe)[ix, ])
)
p2 <- ggplot(df, aes(x = x, y = y, color = expr)) +
  geom_point(size = 0.8) +
  coord_fixed() +
  scale_y_reverse() +
  scale_color_gradient(low = "gray90", high = "blue", trans = "sqrt",
                       breaks = range(df$expr), name = "counts") +
  ggtitle(paste0(ix_name, " (top SVG)")) +
  theme_bw() +
  theme(plot.title = element_text(face = "italic"),
        panel.grid = element_blank(),
        axis.title = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank())

png2 <- file.path(out_dir, sprintf("SVG_nnSVG_%s_top1_spatial.png", sample))
ggsave(png2, p2, width = 7, height = 6, dpi = 150)
log_msg(sprintf("已保存空间表达图: %s", png2))

log_header("完成")
log_msg("nnSVG 完成")
