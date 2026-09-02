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

# ---------- 路径 ----------
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else
  "F:/computatinalbiology/results/local_results/mouse_brain_STARmap/nnSVG"
sample <- if (length(args) >= 2) args[2] else
  basename(dirname(normalizePath(data_dir, winslash = "/")))
n_threads <- if (length(args) >= 3) as.integer(args[3]) else
  if (!is.na(parallel::detectCores())) parallel::detectCores() else 4
out_dir <- data_dir

cat("===== nnSVG: 读取中间文件 =====\n")
cat("数据目录:", data_dir, "\n")
cat("样本标签(sample):", sample, "\n")
cat("平台:", .Platform$OS.type, "| 请求线程数:", n_threads, "\n")

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
spe <- filter_genes(spe, filter_genes_ncounts = 3,
                    filter_genes_pcspots = 0.5, filter_mito = TRUE)
cat(sprintf("过滤后基因数: %d\n", nrow(spe)))

# ---------- 计算 logcounts ----------
spe <- computeLibraryFactors(spe)
spe <- logNormCounts(spe)
cat("assay names:", paste(assayNames(spe), collapse = ", "), "\n")

# 保存中间对象便于调试
rds_path <- file.path(out_dir, "nnSVG_spe.rds")
saveRDS(spe, rds_path)
cat("已保存中间对象:", rds_path, "\n")

# ---------- 运行 nnSVG ----------
# 平台自适应并行（nnSVG 官方公开 BPPARAM 参数；不改包源码，HPC 可复现）：
#   Windows -> SerialParam（本机诊断: PSOCK 并行不稳定，见头注释）
#   POSIX(HPC) + n_threads>1 -> MulticoreParam(fork)，逐基因并行 BRISC
if (.Platform$OS.type == "windows") {
  cat("===== 运行 nnSVG =====\n")
  cat("[平台] Windows: 诊断确认 PSOCK/SnowParam 并行不稳定\n")
  cat("       (BRISC order/neighbor 对象经 socket 导出崩溃), 故使用串行。\n")
  cat("       同一脚本在 Linux/macOS HPC 上将经 nnSVG 的 BPPARAM 自动多核并行。\n")
  bp_param <- BiocParallel::SerialParam()
} else if (n_threads > 1) {
  cat(sprintf("===== 运行 nnSVG (MulticoreParam fork 并行, %d workers) =====\n", n_threads))
  bp_param <- BiocParallel::MulticoreParam(workers = n_threads)
} else {
  cat("===== 运行 nnSVG (串行) =====\n")
  bp_param <- BiocParallel::SerialParam()
}
set.seed(123)
t0 <- Sys.time()
spe <- nnSVG(spe, BPPARAM = bp_param, verbose = FALSE)
t1 <- Sys.time()
cat(sprintf("nnSVG 运行耗时: %.2f 分钟\n", as.numeric(difftime(t1, t0, units = "mins"))))

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
cat("已保存结果:", csv_path, "\n")
cat(sprintf("显著 SVG (padj < 0.05): %d / %d\n",
            sum(res_df$padj < 0.05, na.rm = TRUE), nrow(res_df)))

# ---------- 展示图 ----------
# 1) Top SVG 的 -log10(padj) 柱状图
top_n <- 30
top_df <- head(res_df, top_n)
top_df$gene <- factor(top_df$gene, levels = rev(top_df$gene))

p1 <- ggplot(top_df, aes(x = gene, y = -log10(padj))) +
  geom_col(fill = "darkgreen") +
  coord_flip() +
  labs(title = "nnSVG: Top SVG genes (mouse_brain_STARmap)",
       x = "Gene", y = "-log10(adjusted P-value)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5))

png1 <- file.path(out_dir, sprintf("SVG_nnSVG_%s_top.png", sample))
ggsave(png1, p1, width = 8, height = 6, dpi = 150)
cat("已保存展示图:", png1, "\n")

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
cat("已保存空间表达图:", png2, "\n")

cat("===== nnSVG 完成 ✓ =====\n")
