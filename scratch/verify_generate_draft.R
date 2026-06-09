# verifies every R circuit's generate_draft in both modes:
#   argmax    -> deterministic and equal to the per-position marginal argmax
#   ancestral -> valid, stochastic, and (with trained heads) its empirical marginal
#                matches get_full_vocab_dist, proving the latent factorisation is correct
suppressMessages({source("mtpc/llm.R"); source("mtpc/utils.R"); source("utils.R")
                  source("mtpc/probabilistic_circuits.R")})
device = get_device(); W = 6L; R = 32L; N = 4000L
paths = list(
  ff    = list(lora = "saved_models/lora_ff_w6/mtp_backbone_lora_ff_w6",       head = "saved_models/mtp_head_ff_w6_final.pth"),
  hmm   = list(lora = "saved_models/lora_hmm_w6/mtp_backbone_lora_hmm_w6",     head = "saved_models/mtp_head_hmm_w6_final.pth"),
  cp    = list(lora = "saved_models/lora_cp_w6/mtp_backbone_lora_cp_w6",       head = "saved_models/mtp_head_cp_w6_final.pth"),
  btree = list(lora = "saved_models/lora_btree_w6/mtp_backbone_lora_btree_w6", head = "saved_models/mtp_head_btree_w6_final.pth")
)
torch$manual_seed(0L)
# small hidden -> bias-dominated, byte-realistic SPREAD marginals (not a near-delta),
# so ancestral sampling actually varies and a wrong factorisation would mismatch get_full_vocab_dist
h = torch$randn(1L, 3L, 1472L)$mul(0.02)$to(device)
tv = function(p, q) 0.5 * sum(abs(p - q))
cat(sprintf("%-6s %-9s %-11s %-11s %-10s %-11s %-8s %-8s\n",
            "head", "shape", "argmax_det", "==marginal", "anc_valid", "anc_varies", "top1_p", "max_TV"))
for (ht in names(paths)) {
  m = LLMWrapper(model_id = "google/byt5-small", head_type = ht, window_size = W, ranks = R, lora_path = paths[[ht]]$lora, cheat = FALSE)
  m$load_weights(paths[[ht]]$head, device = "cpu", shift_offset_minus_1 = FALSE)
  m$to(device); m$eval()
  hh = h$to(m$backbone$device)
  probs = m$circuit$get_draft_probs(m, hh)
  a1 = m$circuit$generate_draft(probs, sampling = "argmax")
  a2 = m$circuit$generate_draft(probs, sampling = "argmax")
  ref = sapply(seq_len(W), function(t) which.max(m$circuit$get_full_vocab_dist(probs, t, 1L)) - 1L)
  anc = replicate(N, m$circuit$generate_draft(probs, sampling = "ancestral"))   # [W, N]
  V = length(m$circuit$get_full_vocab_dist(probs, 1L, 1L))
  tvs = sapply(seq_len(W), function(t) {
    emp = tabulate(anc[t, ] + 1L, nbins = V) / N
    tv(emp, m$circuit$get_full_vocab_dist(probs, t, 1L))
  })
  anc_varies = all(apply(anc, 1, function(r) length(unique(r)) > 1L))
  top1 = max(m$circuit$get_full_vocab_dist(probs, 1L, 1L))
  cat(sprintf("%-6s %-9s %-11s %-11s %-10s %-11s %-8.3f %-8.4f\n", toupper(ht),
      paste(length(a1)), all(a1 == a2), all(a1 == ref),
      all(anc >= 0 & anc < V), anc_varies, top1, max(tvs)))
  rm(m); gc()
}
cat("\n(argmax_det & ==marginal & anc_valid must all be TRUE; max_TV small => ancestral samples the correct marginal)\n")
