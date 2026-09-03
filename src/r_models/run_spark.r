# ============================================================
# SPARK-X 方法：SVG 检测（批量化）
# 输入：<data_dir> 下的中间文件（由 src/preprocess/h5ad_preprocess.py 生成）
#       counts.mtx (genes x spots), genes.csv, barcodes.csv, location.csv
# 输出：SVG_SPARK_<sample>.csv + ggplot2 展示图
# 用法: Rscript run_spark.r <data_dir> [sample]
#       sample 默认取 data_dir 上级目录名（如 mouse_brain_STARmap）
# ============================================================
suppressPackageStartupMessages({
  library(SPARK)
  library(Matrix)
  library(ggplot2)
})

# ---------- 路径 ----------
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else
  "F:/computatinalbiology/results/local_results/mouse_brain_STARmap/SPARK_X"
sample <- if (length(args) >= 2) args[2] else
  basename(dirname(normalizePath(data_dir, winslash = "/")))
out_dir <- data_dir

cat("===== SPARK-X: 读取中间文件 =====\n")
cat("数据目录:", data_dir, "\n")
cat("样本标签(sample):", sample, "\n")

counts <- readMM(file.path(data_dir, "counts.mtx"))
genes  <- read.csv(file.path(data_dir, "genes.csv"),  header = FALSE, stringsAsFactors = FALSE)[, 1]
barcodes <- read.csv(file.path(data_dir, "barcodes.csv"), header = FALSE, stringsAsFactors = FALSE)[, 1]
loc_df <- read.csv(file.path(data_dir, "location.csv"), stringsAsFactors = FALSE, row.names = 1)

rownames(counts) <- genes
colnames(counts) <- barcodes

# counts 为 dgCMatrix (genes x spots)，转成 sparseMatrix 供 sparkx
counts <- as(counts, "sparseMatrix")

# 坐标对齐：location.csv 行名即 barcode
loc <- as.matrix(loc_df[barcodes, c("x", "y")])
rownames(loc) <- barcodes

cat(sprintf("counts 维度: %d genes x %d spots\n", nrow(counts), ncol(counts)))
cat(sprintf("location 维度: %d x %d\n", nrow(loc), ncol(loc)))

# ---------- 去除线粒体基因 ----------
mt_idx <- grep("^mt-", rownames(counts), ignore.case = TRUE)
if (length(mt_idx) != 0) {
  counts <- counts[-mt_idx, ]
  cat(sprintf("去除线粒体基因 %d 个\n", length(mt_idx)))
}

# ---------- 运行 SPARK-X ----------
cat("===== 运行 sparkx (mixture) =====\n")
t0 <- Sys.time()
res <- sparkx(counts, loc, numCores = 1, option = "mixture", verbose = FALSE)
t1 <- Sys.time()
wall_seconds <- as.numeric(difftime(t1, t0, units = "secs"))
cat(sprintf("SPARK-X 运行耗时: %.1f 秒\n", wall_seconds))

# ---------- 汇总结果 ----------
res_mtest <- res$res_mtest
res_df <- data.frame(
  gene        = rownames(res_mtest),
  combinedPval = res_mtest$combinedPval,
  adjustedPval = res_mtest$adjustedPval,
  stringsAsFactors = FALSE
)
# 投影核统计量与 p 值（用于参考，sparkx.sk 的 stat 为投影核的统计量）
if ("projection" %in% colnames(res$stats)) {
  res_df$projection_stat <- as.numeric(res$stats[res_df$gene, "projection"])
  res_df$projection_pval <- as.numeric(res$res_stest[res_df$gene, "projection"])
}

# 排名（按 adjustedPval 升序，最小为 1）
res_df <- res_df[order(res_df$adjustedPval, res_df$combinedPval, na.last = TRUE), ]
res_df$rank <- seq_len(nrow(res_df))

# ---------- 保存 CSV ----------
csv_path <- file.path(out_dir, sprintf("SVG_SPARK_%s.csv", sample))
write.csv(res_df, csv_path, row.names = FALSE)
cat("已保存结果:", csv_path, "\n")
cat(sprintf("显著 SVG (adjustedPval < 0.05): %d / %d\n",
            sum(res_df$adjustedPval < 0.05, na.rm = TRUE), nrow(res_df)))

# ---------- 保存跨方法统一的排名 CSV（列固定: gene, stat, pval, padj, rank）----------
# stat 取 -log10(combinedPval)（越大越显著）；p=0 下溢时封顶避免 Inf 导致跨语言读入不兼容。
eval_df <- data.frame(
  gene = res_df$gene,
  stat = -log10(pmax(res_df$combinedPval, 1e-300)),
  pval = res_df$combinedPval,
  padj = res_df$adjustedPval,
  rank = res_df$rank,
  stringsAsFactors = FALSE
)
eval_df <- eval_df[order(eval_df$padj, -eval_df$stat, na.last = TRUE), ]
eval_df$rank <- seq_len(nrow(eval_df))
eval_path <- file.path(out_dir, sprintf("SVG_SPARK_%s_rank.csv", sample))
write.csv(eval_df, eval_path, row.names = FALSE)
cat("已保存统一排名 CSV:", eval_path, "\n")

# ---------- 保存运行时间（JSON，供 evaluation 汇总效率指标）----------
rt_path <- file.path(out_dir, "runtime.json")
writeLines(sprintf('{"method":"spark","sample":"%s","wall_seconds":%.2f}',
                   sample, wall_seconds), rt_path)
cat("已保存运行时间:", rt_path, "\n")

# ---------- 展示图 ----------
top_n <- 30
top_df <- head(res_df, top_n)
top_df$gene <- factor(top_df$gene, levels = rev(top_df$gene))

p <- ggplot(top_df, aes(x = gene, y = -log10(adjustedPval))) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "SPARK-X: Top SVG genes (mouse_brain_STARmap)",
       x = "Gene", y = "-log10(adjusted P-value)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5))

png_path <- file.path(out_dir, sprintf("SVG_SPARK_%s_top.png", sample))
ggsave(png_path, p, width = 8, height = 6, dpi = 150)
cat("已保存展示图:", png_path, "\n")

cat("===== SPARK-X 完成 ✓ =====\n")
