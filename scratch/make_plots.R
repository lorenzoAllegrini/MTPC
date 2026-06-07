# renders the benchmark plots using the mean acceptance rate (tokens accepted per
# round, out of window_size) as the metric, instead of the percentage scale
r = readRDS("benchmark_results/results_benchmark_w6.rds")
W = 6
order_heads = c("ff", "hmm", "btree", "cp")          # ascending performance
labels = toupper(order_heads)
cols   = c("#F17CB0", "#5DA5DA", "#60BD68", "#FAA43A")
ylab_t = sprintf("mean acceptance rate (tokens/round, out of %d)", W)

# per-sample mean tokens/round (each column is one circuit)
perf_tokens = sapply(order_heads, function(h) rowMeans(r[[h]]$acceptance_matrix, na.rm = TRUE))
# global mean tokens/round (over all rounds) per circuit
global_tok = sapply(order_heads, function(h) { v = r[[h]]$acceptance_matrix; mean(v[!is.na(v)]) })

# boxplot of per-sample mean acceptance rate
png("benchmark_results/plot_boxplot_tokens.png", width = 950, height = 680, res = 120)
boxplot(perf_tokens, names = labels, ylim = c(0, W), col = cols, las = 1,
        main = "Mean acceptance rate by circuit (N = 50 prompts)", ylab = ylab_t)
dev.off()

# global mean acceptance rate as a labelled bar chart
png("benchmark_results/plot_global_tokens.png", width = 950, height = 680, res = 120)
bp = barplot(global_tok, names.arg = labels, col = cols, ylim = c(0, W), las = 1,
             main = "Global mean acceptance rate", ylab = ylab_t)
text(bp, global_tok + 0.25, labels = sprintf("%.2f / %d", global_tok, W), font = 2)
dev.off()

# per-circuit density of the mean acceptance rate
png("benchmark_results/plot_densities_tokens.png", width = 1000, height = 820, res = 120)
par(mfrow = c(2, 2))
for (i in seq_along(order_heads)) {
  plot(density(perf_tokens[, i], na.rm = TRUE), xlim = c(0, W), col = cols[i], lwd = 2,
       main = sprintf("Density: %s", labels[i]), xlab = ylab_t)
}
par(mfrow = c(1, 1))
dev.off()

cat("global mean acceptance rate (tokens/round, out of", W, "):\n")
for (i in seq_along(order_heads)) cat(sprintf("  %-6s %.3f\n", labels[i], global_tok[i]))
cat("saved: benchmark_results/{plot_boxplot_tokens,plot_global_tokens,plot_densities_tokens}.png\n")
