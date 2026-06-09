# Statistical analysis of the speculative decoding results
# (uses the methods from lesson 27 - bootstrap, and lesson 32 - comparing classifiers)

library(boot)

par(mar=c(4,4,2,1))

window = 6
results = readRDS("benchmark_results/results_benchmark_w6.rds")
models = names(results)

# per-prompt acceptance rate for every model
# rows = prompts, columns = models (same setup as comparing classifiers over datasets)
acc = NULL
for (m in models) {
  am = results[[m]]$acceptance_matrix       # prompts x rounds, accepted tokens per round
  rate = rowMeans(am, na.rm=TRUE) / window  # mean accepted tokens per round, divided by the window
  acc = cbind(acc, rate)
}
colnames(acc) = models

# 1. Descriptive statistics
cat("Mean acceptance rate per model (%)\n")
for (m in models) cat(m, round(mean(acc[,m])*100, 2), "\n")

# boxplot and densities of the acceptance rate (in tokens per round)
boxplot(acc*window, names=toupper(models), ylab="tokens accepted per round", main="Acceptance rate by model")
par(mfrow=c(2,2))
for (m in models) plot(density(acc[,m]*window), main=toupper(m), xlab="tokens per round")
par(mfrow=c(1,1))

# 2. Confidence intervals for the mean (t-test and bootstrap, lesson 27)
mean.pos = function(d, pos) mean(d[pos])   # estimator for boot()
cat("\n95% confidence intervals (%)\n")
for (m in models) {
  x = acc[,m]
  t_ci = t.test(x)$conf.int                # parametric confidence interval
  b = boot(x, mean.pos, R=1000)            # bootstrap
  b_ci = boot.ci(b, type="perc")$percent[4:5]
  cat(m, "t-test", round(t_ci[1]*100,2), round(t_ci[2]*100,2),
      "bootstrap", round(b_ci[1]*100,2), round(b_ci[2]*100,2), "\n")
}

# 3. Check the assumptions before choosing the test (lesson 32)
cat("\nShapiro-Wilk normality p-values\n")
for (m in models) cat(m, shapiro.test(acc[,m])$p.value, "\n")

# long format for the next tests
score = as.vector(acc)
model = factor(rep(models, each=nrow(acc)))

cat("\nBartlett test for equal variances\n")
bartlett.test(score ~ model)

# 4. Comparing the models (lesson 32)
# the acceptance rates are not normal, so we use the non-parametric Friedman test
cat("\nFriedman test\n")
friedman.test(acc)

# post-hoc: which pairs of models differ (Wilcoxon with Bonferroni correction)
cat("\nPost-hoc pairwise Wilcoxon (Bonferroni)\n")
pairwise.wilcox.test(score, model, p.adjust.method="bonferroni", paired=TRUE)

# if the data were normal we would use ANOVA + Tukey instead:
# fit = aov(score ~ model)
# summary(fit)
# TukeyHSD(fit)
