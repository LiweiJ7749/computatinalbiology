# ============================================================
# SPARK-X 方法：SVG 检测（批量化）
# 输入：<data_dir> 下的中间文件（由 src/preprocess/h5ad_preprocess.py 生成）
#       counts.mtx (genes x spots), genes.csv, barcodes.csv, location.csv
#       location.csv 支持 2 列(x,y)或 3 列(x,y,z)；SPARK-X(locus 为 n x d) 天然支持 3D。
# 输出：SVG_SPARK_<sample>.csv + ggplot2 展示图
# 用法: Rscript run_spark.r <data_dir> [sample]
#       sample 默认取 data_dir 上级目录名（如 mouse_brain_STARmap）
# ============================================================
suppressPackageStartupMessages({
  library(SPARK)
  library(Matrix)
  library(ggplot2)
})

# 日志工具
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
  "F:/computatinalbiology/results/local_results/mouse_brain_STARmap/SPARK_X"
sample <- if (length(args) >= 2) args[2] else
  basename(dirname(normalizePath(data_dir, winslash = "/")))
out_dir <- data_dir

log_header(sprintf("SPARK-X: %s", sample))
log_msg(sprintf("数据目录: %s", data_dir))
log_msg(sprintf("样本标签: %s", sample))

counts <- readMM(file.path(data_dir, "counts.mtx"))
genes  <- read.csv(file.path(data_dir, "genes.csv"),  header = FALSE, stringsAsFactors = FALSE)[, 1]
barcodes <- read.csv(file.path(data_dir, "barcodes.csv"), header = FALSE, stringsAsFactors = FALSE)[, 1]
loc_df <- read.csv(file.path(data_dir, "location.csv"), stringsAsFactors = FALSE, row.names = 1)

rownames(counts) <- genes
colnames(counts) <- barcodes

# counts 为 dgCMatrix (genes x spots)，转成 sparseMatrix 供 sparkx
counts <- as(counts, "sparseMatrix")

# 坐标对齐：location.csv 行名即 barcode；列全部取（2D: x,y；3D: x,y,z）
loc <- as.matrix(loc_df[barcodes, , drop = FALSE])
rownames(loc) <- barcodes

cat(sprintf("counts 维度: %d genes x %d spots\n", nrow(counts), ncol(counts)))
cat(sprintf("location 维度: %d x %d\n", nrow(loc), ncol(loc)))

# ---------- 去除线粒体基因 ----------
mt_idx <- grep("^mt-", rownames(counts), ignore.case = TRUE)
if (length(mt_idx) != 0) {
  counts <- counts[-mt_idx, ]
  log_msg(sprintf("去除线粒体基因 %d 个", length(mt_idx)))
}

# ---------- 运行 SPARK-X ----------
# 3D（location 列数 > 2：Slide-seq 切片堆叠 / Stereo-seq 三维）用投影核 option="single"：
# mixture 的高斯/余弦核按 2D 连续坐标设计，对离散 z 轴（整数切片号）会退化
# （余弦核 cos(2*pi*z/l) 别名 -> 常数列 -> crossprod 奇异），投影核 n x d 任意维均成立。
loc_dim <- ncol(loc)
option <- if (loc_dim > 2) "single" else "mixture"
log_step(1, 2, sprintf("运行 sparkx (option=%s, loc=%dx%d)", option, nrow(loc), loc_dim))
t0 <- Sys.time()
res <- sparkx(counts, loc, numCores = 1, option = option, verbose = FALSE)
t1 <- Sys.time()
wall_seconds <- as.numeric(difftime(t1, t0, units = "secs"))
log_msg(sprintf("SPARK-X 运行耗时: %.1f 秒", wall_seconds))

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
log_msg(sprintf("已保存结果: %s", csv_path))
log_msg(sprintf("显著 SVG (adjustedPval < 0.05): %d / %d",
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
log_msg(sprintf("已保存统一排名 CSV: %s", eval_path))

# ---------- 保存运行时间（JSON，供 evaluation 汇总效率指标）----------
rt_path <- file.path(out_dir, "runtime.json")
writeLines(sprintf('{"method":"spark","sample":"%s","wall_seconds":%.2f}',
                   sample, wall_seconds), rt_path)
log_msg(sprintf("已保存运行时间: %s", rt_path))

# ---------- 展示图 ----------
top_n <- 30
top_df <- head(res_df, top_n)
top_df$gene <- factor(top_df$gene, levels = rev(top_df$gene))

p <- ggplot(top_df, aes(x = gene, y = -log10(adjustedPval))) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = sprintf("SPARK-X: Top SVG genes (%s)", sample),
       x = "Gene", y = "-log10(adjusted P-value)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5))

png_path <- file.path(out_dir, sprintf("SVG_SPARK_%s_top.png", sample))
ggsave(png_path, p, width = 8, height = 6, dpi = 150)
log_msg(sprintf("已保存展示图: %s", png_path))

log_header("完成")
log_msg("SPARK-X 完成")
