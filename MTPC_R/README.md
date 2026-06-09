# MTPC — Multi-Token Prediction with Probabilistic Circuits — R re-implementation

R re-implementation of *"Fast and Expressive Multi-Token Prediction with Probabilistic
Circuits"* 
The code splits into a **core library** (`mtpc/`) and **three entry-point scripts** that
follow the workflow of the project: **training → inference → analysis**.


## Training 
`training.R` trains a chosen circuit (`HEAD_TYPE` ∈ `ff` / `cp` / `hmm` / `btree`) in three
phases: **(0)** autoregressive backbone fine-tuning, **(1)** feed-forward warm-up, **(2)**
joint training of the target circuit, initialised from the trained FF head so it starts
equivalent to it. It builds on three library files:

- **`mtpc/llm.R` — `LLMWrapper`** is the central object: it wraps the byT5 backbone with a
  LoRA adapter and the *active* speculative head, and exposes `swap_head()`, decoder
  hidden-state extraction, and head weight save/load. Both training and inference build on it.
- **`mtpc/probabilistic_circuits.R` — the four circuits** (`FF`, `CP`, `HMM`, `BTree`). Each
  defines `inject_head()` (creates and initialises its parameters in the wrapper) and
  `forward()` (the per-position marginals `q(x_i)` consumed by the loss). Phase 2 copies the
  trained FF emissions into the target circuit's parameters.
- **`utils.R` / `mtpc/utils.R`** supply the dataset loading and batching, the discounted
  multi-token objective `compute_mtpc_loss`, the device helper, and the circuit factory.

**Output → `saved_models/`**: the LoRA adapter and the head weights (`.pth`) for the circuit.

## 2. Inference — `speculative_inference_testing.R`

`speculative_inference_testing.R` loads a frozen **verifier** (the backbone's standard
single-token head) and a **draft** circuit — its head weights restored with
`LLMWrapper$load_weights()` (a pure-R state-dict loader) — then runs self-speculative
decoding over a sample of prompts and records the acceptance metrics. It builds on:

- **`mtpc/speculative_decoding.R`** implements the paper's Algorithm 3:
  `self_speculative_decoding_step()` (draft a window, verify it, accept/reject each token
  with prob `min(1, p/q)`, sample a residual bonus token on the first rejection) and
  `generate_speculative()` (the full generation loop). The `SAMPLING` config selects how the
  circuit drafts — `"argmax"` (greedy) or `"ancestral"` (sample the circuit's latents).
- **`mtpc/probabilistic_circuits.R` (inference side)** — beyond `forward()`, each circuit
  provides `get_draft_probs()`, `generate_draft(sampling = …)`, `compute_prefix_probs()` and
  `get_conditional_dist()`: the quantities the acceptance test and the bonus token need.
- **`mtpc/utils.R` — `safe_decode()`** (token ids → text) and **`utils.R`** — the
  speculative-decoding metrics aggregator.

**Output → `benchmark_results/results_benchmark_w6.rds`**: per-prompt accepted-token counts
and generated text, for every circuit.

## 3. Analysis — `inference_analisys.R`

`inference_analisys.R` reads the benchmark `.rds` and produces the statistical comparison
across the circuits: descriptive metrics (acceptance rate, mean tokens/round), 95 %
confidence intervals (t-test and bootstrap, via the `boot` package), assumption checks
(Shapiro–Wilk normality, Bartlett homogeneity), the Friedman test with a post-hoc pairwise
Wilcoxon, and the comparison plots.

**Output → `benchmark_results/`**: the written report and the bar / box / density plots
(metric = mean acceptance rate, i.e. tokens accepted per round, out of the window size).

BTree is statistically tied with CP (post-hoc Wilcoxon *p* = 0.24) and significantly above
HMM and FF (Friedman χ² = 76.1, *p* ≈ 2e-16). Full write-up in
[`benchmark_results/btree_report.md`](benchmark_results/btree_report.md).

## Setup and running

R (≥ 4.1) with `reticulate` and `boot`, plus a Python env reachable by reticulate with
`torch`, `transformers`, `peft`, `datasets`:

```bash
python -m venv .venv && . .venv/bin/activate && pip install torch transformers peft datasets
```

```r
Rscript training.R                       
Rscript speculative_inference_testing.R  
Rscript inference_analisys.R          
```
