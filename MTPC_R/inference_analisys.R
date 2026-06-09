# analysis of the speculative decoding results

library(boot)

par(mar=c(4,4,2,1))

window = 6
results = readRDS("benchmark_results/results_benchmark_w6.rds")
models = names(results)

# acceptance rate per prompt, one column per model
acc = NULL
for (m in models) {
  am = results[[m]]$acceptance_matrix       # prompts x rounds
  rate = rowMeans(am, na.rm=TRUE) / window  # mean accepted tokens per round
  acc = cbind(acc, rate)
}
colnames(acc) = models

# mean acceptance rate
cat("Mean acceptance rate per model (%)\n")
for (m in models) cat(m, round(mean(acc[,m])*100, 2), "\n")

# boxplot and densities
boxplot(acc*window, names=toupper(models), ylab="tokens accepted per round", main="Acceptance rate by model")
par(mfrow=c(2,2))
for (m in models) plot(density(acc[,m]*window), main=toupper(m), xlab="tokens per round")
par(mfrow=c(1,1))

# confidence intervals for the mean (t-test and bootstrap)
mean.pos = function(d, pos) mean(d[pos])
cat("\n95% confidence intervals (%)\n")
for (m in models) {
  x = acc[,m]
  t_ci = t.test(x)$conf.int
  b = boot(x, mean.pos, R=1000)
  b_ci = boot.ci(b, type="perc")$percent[4:5]
  cat(m, "t-test", round(t_ci[1]*100,2), round(t_ci[2]*100,2),
      "bootstrap", round(b_ci[1]*100,2), round(b_ci[2]*100,2), "\n")
}

# check normality
cat("\nShapiro-Wilk normality p-values\n")
for (m in models) cat(m, shapiro.test(acc[,m])$p.value, "\n")

# long format
score = as.vector(acc)
model = factor(rep(models, each=nrow(acc)))

# equal variances
cat("\nBartlett test\n")
bartlett.test(score ~ model)

# the rates are not normal, so use the Friedman test
cat("\nFriedman test\n")
friedman.test(acc)

# which pairs of models differ
cat("\nPairwise Wilcoxon (Bonferroni)\n")
pairwise.wilcox.test(score, model, p.adjust.method="bonferroni", paired=TRUE)

# parametric version, if the data were normal:
# fit = aov(score ~ model)
# summary(fit)
# TukeyHSD(fit)
