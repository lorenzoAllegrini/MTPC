library(boot)
source("utils.R")

file_path = "benchmark_results/results_benchmark_w6.rds"
window_size = 6
if (grepl("w4", file_path)) window_size = 4
if (grepl("w6", file_path)) window_size = 6
if (grepl("w8", file_path)) window_size = 8

all_results = readRDS(file_path)
probabilistic_heads = names(all_results)

#functions 

get_speculative_metrics = function(probabilistic_heads, all_results, window_size) {
  # calculates mean acceptance rate and average tokens generated per draft verification round
  sapply(probabilistic_heads, function(h) {
    # attach the results list to the global search path
    attach(all_results[[h]])
    
    # calculate metrics using matrix rows and valid rounds
    tot = rowSums(acceptance_matrix, na.rm = TRUE)
    v = sum(!is.na(acceptance_matrix))
    
    # detach the list immediately to clean up memory
    detach(all_results[[h]])
    
    # return only the bare minimum numerical values
    c(global_acc = (sum(tot) / (v * window_size)) * 100, 
      mean_tokens = sum(tot) / v)
  })
}

# extract performance matrix for all models
build_perf_matrix = function(probabilistic_heads, all_results, window_size) {
  # constructs a performance matrix of acceptance rates across all models
  sapply(probabilistic_heads, function(h) {
    rowMeans(all_results[[h]]$acceptance_matrix, na.rm = TRUE) / as.numeric(window_size)
  })
}

# compute parametric and bootstrap confidence intervals
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

# run normality test on each model column
run_shapiro_tests = function(perf_matrix) {
  # runs shapiro-wilk normality test on each model column
  sapply(colnames(perf_matrix), function(h) shapiro.test(perf_matrix[, h])$p.value)
}

# reshape performance matrix into a long format dataframe
build_perf_df = function(perf_matrix) {
  # reshapes the performance matrix into a long format data frame for anova/friedman tests
  data.frame(
    score = as.vector(perf_matrix),
    model = factor(rep(colnames(perf_matrix), each = nrow(perf_matrix))),
    block = factor(rep(1:nrow(perf_matrix), ncol(perf_matrix)))
  )
}


#analisys

# extract the core performance matrix using our defined function
perf_matrix = build_perf_matrix(probabilistic_heads, all_results, window_size)

# plot density distribution for each head (exploratory data analysis)
par(mfrow = c(1, ncol(perf_matrix)))
for (h in colnames(perf_matrix)) {
  plot(density(perf_matrix[, h], na.rm = TRUE), 
       main = toupper(h), xlab = "acceptance rate", col = "darkblue", lwd = 2)
}
par(mfrow = c(1, 1)) # reset plot layout


# compute and display parametric and bootstrap confidence intervals
ci_results = compute_cis(perf_matrix)
print(ci_results)


# run normality tests and reshape data using our defined functions
shapiro_p = run_shapiro_tests(perf_matrix)
perf_df   = build_perf_df(perf_matrix)


# execute hypothesis testing based on shapiro-wilk results
use_non_parametric = any(shapiro_p < 0.05, na.rm = TRUE)

if (use_non_parametric) {
  f_res = friedman.test(perf_matrix)
  cat(sprintf("  - Friedman Chi-Sq: %.4f | p-value: %.4e\n", f_res$statistic, f_res$p.value))
  
  # run post-hoc pairwise wilcoxon if friedman test is significant
  if (f_res$p.value < 0.05) {
    cat("\nSignificant differences found. Post-hoc Pairwise Wilcoxon (Bonferroni):\n")
    print(pairwise.wilcox.test(perf_df$score, perf_df$model, p.adjust.method = "bonferroni", paired = TRUE)$p.value)
  }

} else {
  anova_res = anova(lm(score ~ model, data = perf_df))
  print(anova_res)
  
  # run post-hoc tukey hsd if anova test is significant
  if (anova_res[["Pr(>F)"]][1] < 0.05) {
    cat("\nSignificant differences found. Post-hoc Tukey HSD:\n")
    print(TukeyHSD(aov(score ~ model, data = perf_df)))
  }
}