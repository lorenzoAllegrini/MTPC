# statistical analysis of mtpc speculative decoding inference results

library(boot)
source("utils.R")

par(mar=c(4, 4, 2, 1))

# find and load the benchmark results (dynamic detection of w6 or w4)
file_path = "benchmark_results/results_benchmark_w6.rds"
if (!file.exists(file_path)) file_path = "results_benchmark_w6.rds"
if (!file.exists(file_path)) file_path = "benchmark_results/results_benchmark_w4.rds"
if (!file.exists(file_path)) file_path = "results_benchmark_w4.rds"

if (!file.exists(file_path)) {
  stop("Error: No benchmark results file found (results_benchmark_w6.rds or results_benchmark_w4.rds).")
}

window_size = 6L
if (grepl("w4", file_path)) window_size = 4L
if (grepl("w6", file_path)) window_size = 6L
if (grepl("w8", file_path)) window_size = 8L

all_results = readRDS(file_path)
probabilistic_heads = names(all_results)


# helper functions

get_speculative_metrics = function(probabilistic_heads, all_results, window_size) {
  # calculates mean acceptance rate and average tokens generated per draft verification round
  sapply(probabilistic_heads, function(h) {
    acceptance_matrix = all_results[[h]]$acceptance_matrix
    tot = rowSums(acceptance_matrix, na.rm = TRUE)
    v = sum(!is.na(acceptance_matrix))
    c(global_acc = (sum(tot) / (v * window_size)) * 100, 
      mean_tokens = sum(tot) / v)
  })
}

build_perf_matrix = function(probabilistic_heads, all_results, window_size) {
  # constructs a performance matrix of acceptance rates across all models
  sapply(probabilistic_heads, function(h) {
    rowMeans(all_results[[h]]$acceptance_matrix, na.rm = TRUE) / as.numeric(window_size)
  })
}

compute_cis = function(perf_matrix) {
  # computes t-test and bootstrap confidence intervals for model acceptance rates
  lapply(colnames(perf_matrix), function(h) {
    x = perf_matrix[, h]
    t_ci = t.test(x)$conf.int
    b = boot(x, function(d, pos) mean(d[pos]), R = 1000)
    b_ci = boot.ci(b, type = c("basic", "perc"))
    list(mean = mean(x), t_ci = t_ci[1:2], boot_perc = b_ci$percent[4:5], boot_basic = b_ci$basic[4:5])
  })
}

run_shapiro_tests = function(perf_matrix) {
  # runs shapiro-wilk normality test on each model column
  sapply(colnames(perf_matrix), function(h) shapiro.test(perf_matrix[, h])$p.value)
}

build_perf_df = function(perf_matrix) {
  # reshapes the performance matrix into a long format data frame for anova/friedman tests
  data.frame(
    score = as.vector(perf_matrix),
    model = factor(rep(colnames(perf_matrix), each = nrow(perf_matrix))),
    block = factor(rep(1:nrow(perf_matrix), ncol(perf_matrix)))
  )
}


# analysis and execution

cat("========================================================================\n")
cat("      SPECULATIVE DECODING INFERENCE ANALYSIS - MTPC REPORT             \n")
cat("========================================================================\n")
cat(sprintf("Source File : %s\n", file_path))
cat(sprintf("Window Size : %d\n", window_size))
cat(sprintf("Models      : %s\n", paste(toupper(probabilistic_heads), collapse = ", ")))
cat("========================================================================\n\n")

perf_matrix = build_perf_matrix(probabilistic_heads, all_results, window_size)


# section 1: descriptive & exploratory data analysis (eda)
cat("--- 1. DESCRIPTIVE STATISTICS & GLOBAL METRICS -------------------------\n")
spec_metrics = get_speculative_metrics(probabilistic_heads, all_results, window_size)
for (h in colnames(spec_metrics)) {
  cat(sprintf("Model: %-6s | Global Acceptance Rate: %6.2f%% | Mean Tokens/Round: %6.4f\n", 
              toupper(h), spec_metrics["global_acc", h], spec_metrics["mean_tokens", h]))
}
cat("------------------------------------------------------------------------\n\n")

# plotting uses the mean acceptance rate (tokens accepted per round, out of window_size)
perf_tokens = perf_matrix * window_size
axis_lab = sprintf("Mean acceptance rate (tokens/round, out of %d)", window_size)

boxplot(perf_tokens,
        main = "Speculative Decoding Mean Acceptance Rate by Model",
        ylab = axis_lab,
        ylim = c(0, window_size),
        col = c("#5DA5DA", "#FAA43A", "#60BD68", "#F17CB0")[1:ncol(perf_tokens)],
        las = 1)

# density plots layout for individual distributions
par(mfrow = c(2, 2))
for (h in colnames(perf_tokens)) {
  plot(density(perf_tokens[, h], na.rm = TRUE),
       main = sprintf("Density: %s", toupper(h)),
       xlab = axis_lab,
       col = "darkblue",
       lwd = 2)
}
par(mfrow = c(1, 1)) # reset plot layout


# section 2: confidence intervals estimation
cat("--- 2. POINT ESTIMATES & 95% CONFIDENCE INTERVALS ----------------------\n")
ci_results = compute_cis(perf_matrix)
for (i in seq_along(ci_results)) {
  h = colnames(perf_matrix)[i]
  res = ci_results[[i]]
  cat(sprintf("Model: %-6s (Mean Acceptance Rate: %6.2f%%)\n", toupper(h), res$mean * 100))
  cat(sprintf("  - Parametric (t-test)  : [%6.2f%%, %6.2f%%]\n", res$t_ci[1] * 100, res$t_ci[2] * 100))
  cat(sprintf("  - Bootstrap Percentile : [%6.2f%%, %6.2f%%]\n", res$boot_perc[1] * 100, res$boot_perc[2] * 100))
  cat(sprintf("  - Bootstrap Basic      : [%6.2f%%, %6.2f%%]\n", res$boot_basic[1] * 100, res$boot_basic[2] * 100))
  cat("\n")
}
cat("------------------------------------------------------------------------\n\n")


# section 3: verification of statistical assumptions
cat("--- 3. VERIFICATION OF STATISTICAL ASSUMPTIONS -------------------------\n")

# req: normality of datasets (shapiro-wilk)
shapiro_p = run_shapiro_tests(perf_matrix)
cat("Normality check (Shapiro-Wilk Test):\n")
for (h in names(shapiro_p)) {
  cat(sprintf("  - %-6s: p-value = %10.4e (%s)\n", 
              toupper(h), shapiro_p[h], 
              if (shapiro_p[h] >= 0.05) "PASS - normally distributed" else "FAIL - non-normal"))
}
cat("\n")

# req: homogeneity of variances (bartlett test)
perf_df = build_perf_df(perf_matrix)
bartlett_res = bartlett.test(score ~ model, data = perf_df)
cat("Homogeneity of variances (Bartlett Test):\n")
cat(sprintf("  - Chi-Sq = %.4f, df = %d, p-value = %.4e (%s)\n\n", 
            bartlett_res$statistic, bartlett_res$parameter, bartlett_res$p.value,
            if (bartlett_res$p.value >= 0.05) "PASS - homoscedastic" else "FAIL - heteroscedastic"))

# decision rule for selecting parametric vs non-parametric test
use_non_parametric = any(shapiro_p < 0.05, na.rm = TRUE) || (bartlett_res$p.value < 0.05)
cat(sprintf("Decision: Using %s multi-sample test.\n", 
            if (use_non_parametric) "NON-PARAMETRIC (Friedman Test)" else "PARAMETRIC (ANOVA)"))
cat("------------------------------------------------------------------------\n\n")


# section 4: hypothesis testing & post-hoc analysis
cat("--- 4. MULTI-SAMPLE HYPOTHESIS TESTING & POST-HOC COMPARISONS ----------\n")

if (use_non_parametric) {
  # friedman non-parametric test
  f_res = friedman.test(perf_matrix)
  cat(sprintf("Friedman Rank Sum Test:\n"))
  cat(sprintf("  - Chi-Squared = %.4f, df = %d, p-value = %.4e\n", 
              f_res$statistic, f_res$parameter, f_res$p.value))
  
  # run post-hoc Wilcoxon if Friedman test is significant
  if (f_res$p.value < 0.05) {
    cat("\nSignificant differences found. Post-Hoc Pairwise Wilcoxon (Bonferroni correction):\n")
    p_matrix = pairwise.wilcox.test(perf_df$score, perf_df$model, p.adjust.method = "bonferroni", paired = TRUE)$p.value
    print(round(p_matrix, 6))
  } else {
    cat("\nNo statistically significant differences detected between models.\n")
  }
  
} else {
  # anova via linear regression
  fit = lm(score ~ model, data = perf_df)
  anova_res = anova(fit)
  cat("One-way Analysis of Variance (ANOVA):\n")
  print(anova_res)
  
  # run post-hoc Tukey HSD if ANOVA is significant
  p_val = anova_res[["Pr(>F)"]][1]
  if (p_val < 0.05) {
    cat("\nSignificant differences found. Post-Hoc Tukey HSD Test:\n")
    print(TukeyHSD(aov(score ~ model, data = perf_df)))
  } else {
    cat("\nNo statistically significant differences detected between models.\n")
  }
}
cat("========================================================================\n")