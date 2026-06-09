# MTPC — Multi-Token Prediction with Probabilistic Circuits — R re-implementation

**Project report — code structure**

This package is an **R** re-implementation of *"Fast and Expressive Multi-Token Prediction
with Probabilistic Circuits"* (MTPC, arXiv 2511.11346). We retrofit a small encoder–decoder
language model (**byT5-small**) with a probabilistic-circuit head that models the *joint*
distribution over the next *n* tokens, and use it as the **draft model** in self-speculative
decoding, verified token-by-token against the model's standard single-token head.

The whole pipeline is written in R; the PyTorch / Hugging Face tensors and model
(`torch`, `transformers`, `peft`, `datasets`) are driven through the **`reticulate`**
bridge. There is no project Python source code.

## What is implemented

- **Four probabilistic-circuit heads** parameterising `q(x_{t+1..t+n} | context)`:
  - `FF` — fully factorised (independent per-position heads),
  - `CP` — canonical polyadic (one shared latent, rank-`R` mixture),
  - `HMM` — inhomogeneous hidden Markov chain over the window,
  - `BTree` — balanced binary tree of latents over the window.
- **Three-phase training** (backbone SFT → FF warm-up → target-circuit joint training, with
  the target circuit initialised from the trained FF head so it starts equivalent to FF).
- **Self-speculative decoding** (paper Algorithm 3): the circuit drafts an `n`-token window
  (by `argmax` or by `ancestral` sampling of its latents), the verifier accepts/rejects
  with probability `min(1, p/q)`, and a residual bonus token is sampled on the first reject.
- **Statistical analysis** of the acceptance rates across circuits (CIs, normality/variance
  checks, Friedman + post-hoc tests, plots).

## Code structure

```
mtpc/                              core library
  probabilistic_circuits.R         the 4 circuits: inject_head (parameters + init),
                                    forward (marginals q(x_i)), generate_draft,
                                    compute_prefix_probs / get_conditional_dist (for decoding)
  llm.R                            LLMWrapper: byT5 backbone + LoRA + the active head;
                                    hidden-state extraction, draft verification, load/save
  speculative_decoding.R           Algorithm 3: one decoding step + the generation loop
  utils.R                          device, batching, MTPC loss, safe_decode

training.R                         3-phase training of any circuit (config at the top)
speculative_inference_testing.R    self-speculative-decoding benchmark over all circuits
inference_analisys.R               statistical analysis + plots of the benchmark results
utils.R                            dataset loading, metrics, circuit factory

benchmark_results/                 report, plots and raw results of the experiments
saved_models/                      trained checkpoints (not bundled — see "Trained models")
```

## How the pieces fit together

```
                 training.R                 speculative_inference_testing.R     inference_analisys.R
 dataset ─▶ phase 0: backbone SFT ─┐
           phase 1: FF warm-up ────┤        verifier (STP head) ┐
           phase 2: target circuit ┴─▶ ─┐   draft   (circuit)   ┴─▶ generate_speculative ─▶ results.rds ─▶ stats + plots
                                        └── saved_models/ ──────────┘
```

`mtpc/` is the reusable library; the three top-level scripts are the entry points
(train → benchmark → analyse). Every script is configured by a small block of constants
at its top (head type, window size, ranks, sampling strategy, phase flags).

## Results (window = 6, 50 prompts, self-speculative, `argmax` draft)

Mean acceptance rate = tokens accepted per draft/verify round (out of 6):

| circuit | tokens/round | acceptance |
|---------|-------------:|-----------:|
| CP        | 1.80 / 6   | 30.1 % |
| **BTree** | **1.66 / 6** | **27.6 %** |
| HMM       | 1.33 / 6   | 22.2 % |
| FF        | 1.10 / 6   | 18.4 % |

BTree is statistically tied with CP (post-hoc Wilcoxon *p* = 0.24) and significantly above
HMM and FF (Friedman χ² = 76.1, *p* ≈ 2e-16). Full write-up, plots and tests in
[`benchmark_results/btree_report.md`](benchmark_results/btree_report.md).

## Requirements and running

R (≥ 4.1) with `reticulate` and `boot`, plus a Python env reachable by reticulate with
`torch`, `transformers`, `peft`, `datasets`:

```bash
python -m venv .venv && . .venv/bin/activate && pip install torch transformers peft datasets
```

```r
Rscript training.R                       # train a circuit (set HEAD_TYPE at the top)
Rscript speculative_inference_testing.R  # benchmark (set SAMPLING = "argmax" | "ancestral")
Rscript inference_analisys.R             # statistics + plots
```

### Trained models

The checkpoints (~1.3 GB) are not bundled; place them under `saved_models/` as
`byt5_standard_lora_phase0/` (verifier), `lora_<c>_w6/…` + `mtp_head_<c>_w6_final.pth`
for `c ∈ {cp, hmm, btree}`, and `lora_ff_w6_phase1/` + `mtp_head_ff_w6_phase1.pth`.
Without them, `training.R` still trains from scratch (the dataset is downloaded
automatically).
