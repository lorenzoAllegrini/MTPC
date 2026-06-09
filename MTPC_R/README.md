# MTPC — Fast & Expressive Multi-Token Prediction with Probabilistic Circuits (R re-implementation)

R re-implementation of the **MTPC** method (paper *"Fast and Expressive Multi-Token
Prediction with Probabilistic Circuits"*, 2511.11346). A small encoder–decoder language
model (**byT5-small**) is retrofitted with a probabilistic-circuit head that models the
joint distribution over the next *n* tokens, and that head is used as the **draft model**
in self-speculative decoding (verified token-by-token against the standard single-token
head).

Four circuit families are implemented and compared:

| circuit | latent structure |
|---------|------------------|
| **FF**    | fully factorised — independent per-position heads |
| **CP**    | canonical polyadic — one shared latent (mixture of rank-`R` factorisations) |
| **HMM**   | inhomogeneous hidden Markov chain over the window |
| **BTree** | balanced binary tree of latents over the window |

## The code is pure R

The whole pipeline (circuits, three-phase training, self-speculative decoding, statistical
analysis) is implemented in **R**. The PyTorch / Hugging Face back-ends (`torch`,
`transformers`, `peft`, `datasets`) are driven through the **`reticulate`** package — the
standard R↔Python bridge — exactly as R uses any compiled numerical library. **There is no
project Python source code.**

## Repository structure

```
training.R                        # 3-phase training of any circuit (ff/cp/hmm/btree)
speculative_inference_testing.R   # self-speculative-decoding benchmark over the circuits
inference_analisys.R              # statistical analysis + plots of the benchmark results
utils.R                           # shared helpers (batching, MTPC loss, metrics, decode)
mtpc/
  llm.R                           # LLMWrapper: byT5 backbone + LoRA + the speculative head
  probabilistic_circuits.R        # FF / CP / HMM / BTree circuit definitions
  speculative_decoding.R          # Algorithm 3: draft + verify + residual bonus token
  utils.R                         # device, padding/batching, MTPC loss, safe_decode
benchmark_results/                # report, plots and raw results of the experiments
saved_models/                     # trained checkpoints (NOT included — see below)
```

## Requirements

- **R** (≥ 4.1) with packages: `reticulate`, `boot` (for the analysis).
- A **Python environment** reachable by reticulate with: `torch`, `transformers`,
  `peft`, `datasets`. The scripts auto-detect a local `.venv/` if present
  (`mtpc/llm.R`), otherwise reticulate uses the configured Python
  (`reticulate::use_virtualenv(...)` or `RETICULATE_PYTHON`).

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch transformers peft datasets
```

## Trained models

The trained checkpoints (~1.3 GB) are **not bundled** in this code package; place them
under `saved_models/` with these names:

```
saved_models/byt5_standard_lora_phase0/                       # backbone SFT (verifier)
saved_models/lora_ff_w6_phase1/    + mtp_head_ff_w6_phase1.pth # FF warm-up
saved_models/lora_<c>_w6/mtp_backbone_lora_<c>_w6/  + mtp_head_<c>_w6_final.pth   # c in cp/hmm/btree
```

Without them, `training.R` can still train from scratch (it downloads the dataset via
`datasets`), and the inference/analysis scripts reproduce the reported results.

## Running

```r
# train a circuit (set HEAD_TYPE / phase flags at the top of the file)
Rscript training.R

# benchmark every circuit with self-speculative decoding -> benchmark_results/*.rds
#   set SAMPLING = "argmax" or "ancestral" at the top to pick the draft strategy
Rscript speculative_inference_testing.R

# statistical analysis + plots of the benchmark
Rscript inference_analisys.R
```

## Results (window = 6, 50 prompts, self-speculative, argmax draft)

Mean acceptance rate (tokens accepted per draft/verify round, out of 6):

| circuit | tokens/round | acceptance |
|---------|-------------:|-----------:|
| CP      | 1.80 / 6     | 30.1 %     |
| **BTree** | **1.66 / 6** | **27.6 %** |
| HMM     | 1.33 / 6     | 22.2 %     |
| FF      | 1.10 / 6     | 18.4 %     |

BTree is statistically tied with CP (post-hoc Wilcoxon p = 0.24) and significantly above
HMM and FF (Friedman χ² = 76.1, p ≈ 2e-16). Full write-up, plots and hypothesis tests in
[`benchmark_results/btree_report.md`](benchmark_results/btree_report.md).
