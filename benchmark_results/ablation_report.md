# Ablation: argmax vs ancestral drafting (window 6, 50 prompts)

acceptance rate (%), per-prompt mean +/- standard error:

| circuit | argmax | ancestral | delta |
|---------|-------:|----------:|------:|
| FF | 18.2% +/- 0.4 | 14.8% +/- 0.4 | +3.3 |
| CP | 30.9% +/- 1.4 | 23.7% +/- 0.9 | +7.1 |
| HMM | 29.8% +/- 1.2 | 23.8% +/- 0.8 | +6.0 |
| BTREE | 29.4% +/- 1.1 | 23.0% +/- 0.8 | +6.5 |

argmax (greedy draft) beats ancestral sampling on every circuit (paired Wilcoxon p < 1e-4).
the original benchmark drafted HMM with ancestral and the others with argmax, which made HMM
look weak (23%); under the same argmax strategy HMM reaches ~30%, in line with CP and BTree.
