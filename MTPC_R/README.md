# MTPC in R

This is our re-implementation of the paper "Fast and Expressive Multi-Token
Prediction with Probabilistic Circuits".

We put a small probabilistic-circuit head on top of byt5-small and use it to draft
several tokens at once for speculative decoding. The model runs from R through the
reticulate package (which calls torch and transformers).

## Files

- training.R: trains one circuit (ff, cp, hmm or btree) and saves it in saved_models/.
- speculative_inference_testing.R: runs the speculative decoding and saves the
  acceptance results in benchmark_results/.
- inference_analisys.R: reads those results and does the statistics and the plots
  (confidence intervals, normality, Friedman test).
- utils.R: data loading, batching and the loss/metric helpers.
- mtpc/: the library used by the scripts
  - llm.R: loads byt5 with LoRA and the chosen head
  - probabilistic_circuits.R: the four circuits (ff, cp, hmm, btree)
  - speculative_decoding.R: the draft and verify loop
  - utils.R: small helpers

## How to run

You need R and a python environment with torch, transformers, peft and datasets
(used through reticulate).

    Rscript training.R
    Rscript speculative_inference_testing.R
    Rscript inference_analisys.R

The trained models are too big to include, they go in saved_models/. The results we
got are in benchmark_results/.
