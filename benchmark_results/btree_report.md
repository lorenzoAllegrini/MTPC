# MTPC BTree — Self-Speculative Decoding Benchmark Report

**Date:** 2026-06-07
**Model under test:** newly trained BTree probabilistic circuit (Google Colab), `window = 6`, `ranks = 32`
**Checkpoints installed:** `saved_models/lora_btree_w6/mtp_backbone_lora_btree_w6` + `saved_models/mtp_head_btree_w6_final.pth`

## 1. Methodology

- **Backbone / verifier:** byT5-small with the Phase-0 SFT LoRA adapter (`byt5_standard_lora_phase0`), non-cheat.
- **Decoding:** self-speculative decoding (paper Algorithm 3) — the BTree MTP head drafts a 6-token window, the verifier accepts/rejects with probability `min(1, p/q)`, and a residual/bonus token is sampled on the first rejection.
- **Data:** `ai2-adapt-dev/flan_v2_converted` test split; **50 prompts** (`set.seed(42)`, identical to the prior FF/CP/HMM run, so the four heads are evaluated on the same prompts).
- **Budget:** `max_new_tokens = 60` per prompt.
- **Metric:** *acceptance rate* = accepted drafted tokens / proposed drafted tokens; *mean tokens/round* = average drafted tokens accepted per draft-verify round (∈ [0, 6]).

The new BTree was benchmarked with `scratch/btree_benchmark.R` and merged into `benchmark_results/results_benchmark_w6.rds`; the comparative statistics below come from `inference_analisys.R`.

## 2. Global metrics (N = 50, window = 6)

| Head      | Global acceptance | Mean tokens/round | Rounds | Accepted tokens |
|-----------|------------------:|------------------:|-------:|----------------:|
| **CP**    | **30.08 %**       | 1.805             | 1095   | 1976            |
| **BTree** | **27.61 %**       | 1.657             | 1156   | 1915            |
| HMM       | 22.18 %           | 1.331             | 1308   | 1741            |
| FF        | 18.38 %           | 1.103             | 1442   | 1590            |

> The new BTree ranks **second of four**, just behind CP and clearly ahead of HMM and FF.
> For reference, the previous (under-trained, 120-sample) BTree toy scored **1.21 %** — the retrained model is a **≈ 23× improvement**.

## 3. Per-round acceptance distribution

How many of the 6 drafted tokens are accepted in a round (% of rounds):

| Head  |    0 |    1 |    2 |    3 |   4 |   5 |   6 |
|-------|-----:|-----:|-----:|-----:|----:|----:|----:|
| CP    | 11.7 | 34.7 | 32.5 | 12.1 | 3.7 | 2.1 | 3.2 |
| BTree |  9.1 | 46.5 | 29.7 |  7.4 | 3.0 | 0.8 | 3.6 |
| HMM   | 19.9 | 44.7 | 24.8 |  6.3 | 2.7 | 0.5 | 1.1 |
| FF    | 11.3 | 72.6 | 13.3 |  1.6 | 0.4 | 0.0 | 0.8 |

> BTree advances by **≥ 1 token in 90.9 % of rounds** (lowest zero-acceptance rate after CP), and reaches **full 6-token acceptance 3.6 % of the time** — slightly more often than CP (3.2 %). Its richer tail vs FF (which is essentially capped at 1 token/round) reflects the expressiveness of the hierarchical latent tree.

## 4. Point estimates & 95 % confidence intervals

Mean per-prompt acceptance rate (bootstrap, R = 1000):

| Head  | Mean   | t-test 95 % CI     | Bootstrap percentile |
|-------|-------:|--------------------|----------------------|
| CP    | 31.20 % | [29.03 %, 33.37 %] | [29.15 %, 33.31 %] |
| BTree | 29.62 % | [25.99 %, 33.24 %] | [26.60 %, 33.73 %] |
| HMM   | 23.32 % | [21.11 %, 25.54 %] | [21.32 %, 25.81 %] |
| FF    | 19.35 % | [17.05 %, 21.66 %] | [17.61 %, 21.71 %] |

> The BTree and CP confidence intervals **overlap substantially**.

## 5. Statistical significance

- **Normality (Shapiro–Wilk):** CP normal (p = 0.65); HMM, FF, BTree non-normal → a **non-parametric** test is used.
- **Variance homogeneity (Bartlett):** heteroscedastic (p = 1.9e-4).
- **Friedman rank-sum test:** χ²(3) = **76.14**, p = **2.07e-16** → the four heads differ significantly.
- **Post-hoc pairwise Wilcoxon (Bonferroni-corrected p-values):**

| pair        | p-value   | verdict                          |
|-------------|-----------|----------------------------------|
| BTree vs CP | **0.244** | **no significant difference**    |
| BTree vs HMM| 0.0326    | BTree > HMM (significant)         |
| BTree vs FF | < 1e-6    | BTree > FF (significant)          |
| CP vs HMM   | 1e-6      | CP > HMM                          |
| CP vs FF    | < 1e-16   | CP > FF                           |
| HMM vs FF   | 4.2e-4    | HMM > FF                          |

> **Key statistical result:** the new BTree is **statistically indistinguishable from the best head (CP)** and **significantly better than both HMM and FF**.

## 6. Sample generations (new BTree)

Prompt tail → generated continuation (first ~110 bytes; coherent, no byte-salad):

```
[5]  ...Italian language in literature." Options: - yes - no Please think gradually:
     GEN: Before Alighieri helped popularize the use of the Italian language in l...

[9]  ...textbooks?. Me: Hmmm, let me think. I think this is the detailed solution:
     GEN: 20% of $45. Which amount costs 20% less. One thing is the detailed sol...

[20] ...Italian coffee shop named xname. ; Categories: location, price, cuisine A:
     GEN: location[riverside] is no high priced Italian coffee shop. You will be g...
```

(A minority of hard prompts, e.g. open-ended "guess who" tasks, still degrade — expected for a small byte-level draft model.)

## 7. Conclusion

The retrained BTree circuit is a strong draft model: **27.6 % global acceptance / 1.66 tokens per round**, statistically on par with CP and significantly ahead of HMM and FF. The hierarchical latent tree delivers the expressiveness gain over the fully-factorised (FF) baseline that the paper predicts, while remaining competitive with the CP circuit.

## 8. Reproduction

```bash
# 1. benchmark the new btree on the same 50 prompts and merge into the rds
BT_N=50 Rscript scratch/btree_benchmark.R
# 2. produce this comparative analysis
Rscript inference_analisys.R
```

Artefacts: `benchmark_results/results_benchmark_w6.rds` (raw), `benchmark_results/btree_benchmark_plots.pdf` (boxplot + per-head densities).
