# =============================================================================
#  training.R  —  MTPC speculative-head training (R port of src/training.py)
#
#  Trains any probabilistic circuit (ff / cp / hmm / btree) with the SAME three
#  phase recipe as the Python code, just written in a single, linear script:
#
#    Phase 0  Backbone SFT          autoregressive fine-tuning of the byT5 LoRA
#    Phase 1  FF warm-up            FF head, emissions initialised from the
#                                   backbone single-token-prediction (STP) matrix
#    Phase 2  Target circuit        cp / hmm / btree initialised from the TRAINED
#                                   FF head (emissions copied, sum-gates uniform)
#                                   -> the circuit starts == FF, then trained jointly
#
#  Set HEAD_TYPE and the SKIP_PHASE_* flags below, then run:
#
#      Rscript training.R
#
#  Note: this uses reticulate to drive torch / transformers / peft (the standard
#  way to do deep learning from R). It does NOT depend on the project's Python
#  source — every training step is implemented here in R.
# =============================================================================

source("mtpc/llm.R")
source("mtpc/utils.R")
source("utils.R")
source("mtpc/probabilistic_circuits.R")

# ---- configuration ----------------------------------------------------------
MODEL_ID    = "google/byt5-small"
HEAD_TYPE   = "cp"        # circuit to train: "ff" | "cp" | "hmm" | "btree"
WINDOW_SIZE = 6L          # number of future tokens predicted jointly
RANKS       = 32L         # latent states / CP rank
BATCH_SIZE  = 2L          # micro-batch (gradient accumulation targets eff. batch 16)
MAX_LEN     = 512L        # max sequence length (bytes)
EPOCHS      = 1L          # epochs per phase
GAMMA       = 0.8         # MTPC loss per-step discount factor
TASK_FILTER = "ai2-adapt-dev/flan_v2_converted"
MAX_SAMPLES = 20000L      # dataset samples to load
SAVE_DIR    = "saved_models"

# learning rates per phase (head / backbone-LoRA)
PHASE0_LR      = 5e-4
PHASE1_HEAD_LR = 1e-3 ; PHASE1_LLM_LR = 1e-4
PHASE2_HEAD_LR = 3e-4 ; PHASE2_LLM_LR = 1e-4

# which phases to run. Skipping a phase reuses its saved checkpoint instead of
# retraining it (Phase 2 reuses the Phase-1 FF backbone + FF head).
SKIP_PHASE_0 = TRUE       # reuse <SAVE_DIR>/byt5_standard_lora_phase0
SKIP_PHASE_1 = TRUE       # reuse <SAVE_DIR>/{lora_ff_w*_phase1, mtp_head_ff_w*_phase1.pth}
SKIP_PHASE_2 = FALSE

device = get_device()
cat(sprintf("[CONFIG] head=%s window=%d ranks=%d batch=%d max_len=%d device=%s\n",
            HEAD_TYPE, WINDOW_SIZE, RANKS, BATCH_SIZE, MAX_LEN, device$type))

# ---- dataset ----------------------------------------------------------------
tokenizer = transformers$AutoTokenizer$from_pretrained(MODEL_ID)
tokenizer$chat_template = CHAT_TEMPLATE
batches = prepare_train_data(TASK_FILTER, tokenizer, MAX_LEN, BATCH_SIZE, max_samples = MAX_SAMPLES)
cat(sprintf("[DATA] %d micro-batches ready\n", length(batches)))

# ---- FF -> target transfer (mirror of Python init_{cp,hmm,btree}_from_ff) ----
init_target_from_ff = function(model, ff_head_path, head_type, window_size, ranks) {
  # Copies the TRAINED FF per-position emission matrices into the target circuit's
  # emission tensor (broadcast over the rank dimension) with tiny symmetry-breaking
  # noise. swap_head() already zeroed the sum-gates (uniform mixture) and set the HMM
  # transitions to identity, so after this the target circuit reproduces the FF marginals
  # exactly -> "every circuit starts equivalent to Fully-Factorised, then improves".
  if (!file.exists(ff_head_path)) {
    cat(sprintf("[WARN] FF head not found at %s -> keeping STP init from swap_head.\n", ff_head_path))
    return(invisible(FALSE))
  }
  # torch$load auto-converts the state_dict to an R named list (values stay py tensors)
  ff_sd = torch$load(ff_head_path, map_location = "cpu")
  ht = tolower(head_type)

  if (ht == "ff") {
    # target IS FF: just continue from the trained FF head
    model$heads$load_state_dict(ff_sd)
    cat(sprintf("[INIT] Loaded trained FF head from %s (continue training).\n", ff_head_path))
    return(invisible(TRUE))
  }

  with(torch$no_grad(), {
    target_w = model$heads[["input_units_phi"]]$weight
    H = as.integer(target_w$shape[1])
    # per-window FF emission matrices, each [vocab, embed]
    ff_w = lapply(seq_len(window_size),
                  function(t) ff_sd[[sprintf("input_units_phi_%d.weight", t)]])
    emiss = torch$stack(ff_w, dim = 0L)                               # [W, V, H]
    if (ht == "cp") {
      emiss = emiss$unsqueeze(0L)$expand(ranks, -1L, -1L, -1L)$reshape(-1L, H)   # [R,W,V] layout
    } else {
      emiss = emiss$unsqueeze(1L)$expand(-1L, ranks, -1L, -1L)$reshape(-1L, H)   # [W,R,V] layout (hmm/btree)
    }
    emiss = emiss$add(torch$randn_like(emiss)$mul(1e-4))             # break rank symmetry
    target_w$copy_(emiss$to(target_w$device))
    nn$init$zeros_(model$heads[["input_units_phi"]]$bias)
  })
  cat(sprintf("[INIT] Transferred trained FF emissions -> %s head (gates uniform).\n", toupper(head_type)))
  invisible(TRUE)
}

# ---- training loop ----------------------------------------------------------
training_loop = function(model, optimizer, batches, phase, epochs = 1L,
                         window_size = 6L, gamma = 0.8, is_log_probs = FALSE, device) {
  # one training phase: optimise the backbone (phase 0) or the circuit head + backbone
  accumulation_steps = max(1L, as.integer(16L / BATCH_SIZE))
  cat(sprintf("[SYSTEM] micro-batch=%d | accumulation=%d (effective batch=%d)\n",
              BATCH_SIZE, accumulation_steps, BATCH_SIZE * accumulation_steps))

  losses = numeric(0); idx = 1L
  for (epoch in seq_len(epochs)) {
    total_loss = 0.0
    optimizer$zero_grad()
    for (i in seq_along(batches)) {
      batch = batches[[i]]
      input_ids      = batch$input_ids$to(device)
      labels         = batch$labels$to(device)
      attention_mask = batch$attention_mask$to(device)

      if (phase == 0L) {
        # mask assistant tokens in the encoder input to prevent future-token leakage
        decoder_start_token_id = model$backbone$config$decoder_start_token_id
        encoder_input_ids = input_ids$clone()
        encoder_input_ids[labels$ne(-100L)] = as.integer(decoder_start_token_id)
        outputs = model$backbone(input_ids = encoder_input_ids,
                                 attention_mask = attention_mask, labels = labels)
        loss = outputs$loss
      } else {
        # speculative circuit forward -> per-step MTPC loss
        dec_in = model$prepare_decoder_input_ids(input_ids)
        hidden_states = model$get_hidden_states(input_ids, dec_in,
                          attention_mask = attention_mask, labels = labels)$x
        mtp_logits = model$circuit$forward(model, hidden_states)
        loss = compute_mtpc_loss(mtp_logits, labels, window_size, gamma, is_log_probs = is_log_probs)
      }

      scaled_loss = loss$div(accumulation_steps)
      scaled_loss$backward()
      if (i %% accumulation_steps == 0L || i == length(batches)) {
        optimizer$step(); optimizer$zero_grad()
      }

      loss_val = loss$item()
      total_loss = total_loss + loss_val
      losses[idx] = loss_val; idx = idx + 1L
      if (i %% 5L == 0L || i == length(batches))
        cat(sprintf("  Phase %d | Batch %d/%d | Loss %.4f | Avg %.4f\n",
                    phase, i, length(batches), loss_val, total_loss / i))
    }
  }

  # save a loss curve for the phase
  tryCatch({
    dir.create(SAVE_DIR, showWarnings = FALSE, recursive = TRUE)
    png(file.path(SAVE_DIR, sprintf("loss_phase%d.png", phase)))
    plot(losses, type = "l", col = "blue", xlab = "Iteration", ylab = "Loss",
         main = sprintf("Phase %d Training Loss", phase)); dev.off()
  }, error = function(e) cat(sprintf("[WARN] loss plot failed: %s\n", e$message)))
  invisible(losses)
}

# ---- backbone resume path ---------------------------------------------------
# Phase 0 from scratch -> fresh LoRA; Phase 1 -> phase-0 backbone; Phase 2 -> the
# FF-warmed phase-1 backbone (so joint training continues from the FF checkpoint).
resume_lora = file.path(SAVE_DIR, sprintf("lora_ff_w%d_phase1", WINDOW_SIZE))  # Phase 2 default
if (!SKIP_PHASE_0) {
  resume_lora = NULL
} else if (!SKIP_PHASE_1) {
  resume_lora = file.path(SAVE_DIR, "byt5_standard_lora_phase0")
}
if (!is.null(resume_lora) && !dir.exists(resume_lora)) {
  cat(sprintf("[WARN] resume backbone %s missing -> falling back to phase-0 backbone.\n", resume_lora))
  resume_lora = file.path(SAVE_DIR, "byt5_standard_lora_phase0")
  if (!dir.exists(resume_lora)) resume_lora = NULL
}

model = LLMWrapper(model_id = MODEL_ID, lora_path = resume_lora, lora_r = 8L, lora_alpha = 16L)
model$to(device)
cat(sprintf("[MODEL] backbone loaded (resume_lora=%s)\n", if (is.null(resume_lora)) "fresh" else resume_lora))

# =============================================================================
# PHASE 0 — autoregressive backbone fine-tuning
# =============================================================================
if (SKIP_PHASE_0) {
  cat("\n[PHASE 0] skipped (using pre-trained backbone).\n")
} else {
  cat("\n[PHASE 0] Backbone autoregressive fine-tuning\n")
  model$train(); model$heads$eval()
  lora_params = model$get_parameter_groups(0, PHASE0_LR)[[2]]$params
  optimizer = torch$optim$Adam(lora_params, lr = PHASE0_LR)
  training_loop(model, optimizer, batches, phase = 0L, epochs = EPOCHS, device = device)
  dir.create(SAVE_DIR, showWarnings = FALSE, recursive = TRUE)
  model$backbone$save_pretrained(file.path(SAVE_DIR, "byt5_standard_lora_phase0"))
  cat("[PHASE 0] backbone saved.\n")
}

# =============================================================================
# PHASE 1 — Feed-Forward warm-up (emissions <- backbone STP)
# =============================================================================
if (SKIP_PHASE_1) {
  cat("\n[PHASE 1] skipped (using pre-trained FF warm-up).\n")
} else {
  cat("\n[PHASE 1] Feed-Forward head warm-up\n")
  model$swap_head("ff", WINDOW_SIZE, RANKS)   # inject_head initialises emissions from STP
  model$to(device); model$train()
  optimizer = torch$optim$Adam(model$get_parameter_groups(PHASE1_HEAD_LR, PHASE1_LLM_LR))
  training_loop(model, optimizer, batches, phase = 1L, epochs = EPOCHS,
                window_size = WINDOW_SIZE, gamma = GAMMA, is_log_probs = FALSE, device = device)
  dir.create(SAVE_DIR, showWarnings = FALSE, recursive = TRUE)
  torch$save(model$heads$state_dict(), file.path(SAVE_DIR, sprintf("mtp_head_ff_w%d_phase1.pth", WINDOW_SIZE)))
  model$backbone$save_pretrained(file.path(SAVE_DIR, sprintf("lora_ff_w%d_phase1", WINDOW_SIZE)))
  cat("[PHASE 1] FF head + backbone saved.\n")
}

# =============================================================================
# PHASE 2 — target circuit joint training (initialised from the trained FF)
# =============================================================================
cat(sprintf("\n[PHASE 2] Target %s joint training\n", toupper(HEAD_TYPE)))

# free the previous phase's optimizer before allocating the new head
if (exists("optimizer")) rm(optimizer)
gc()
if (device$type == "mps")  torch$mps$empty_cache()
if (device$type == "cuda") torch$cuda$empty_cache()

# swap to the target circuit (STP init) then transfer the trained FF emissions
model$swap_head(HEAD_TYPE, WINDOW_SIZE, RANKS)
model$to(device)
ff_head_path = file.path(SAVE_DIR, sprintf("mtp_head_ff_w%d_phase1.pth", WINDOW_SIZE))
init_target_from_ff(model, ff_head_path, HEAD_TYPE, WINDOW_SIZE, RANKS)
model$to(device)
model$train()

is_log_probs = tolower(HEAD_TYPE) %in% c("cp", "hmm", "btree")
optimizer = torch$optim$Adam(model$get_parameter_groups(PHASE2_HEAD_LR, PHASE2_LLM_LR))
training_loop(model, optimizer, batches, phase = 2L, epochs = EPOCHS,
              window_size = WINDOW_SIZE, gamma = GAMMA, is_log_probs = is_log_probs, device = device)

# =============================================================================
# SAVE final head weights + backbone LoRA adapter
# =============================================================================
dir.create(SAVE_DIR, showWarnings = FALSE, recursive = TRUE)
head_path = file.path(SAVE_DIR, sprintf("mtp_head_%s_w%d_final.pth", tolower(HEAD_TYPE), WINDOW_SIZE))
torch$save(model$heads$state_dict(), head_path)
lora_path = file.path(SAVE_DIR, sprintf("lora_%s_w%d", tolower(HEAD_TYPE), WINDOW_SIZE),
                      sprintf("mtp_backbone_lora_%s_w%d", tolower(HEAD_TYPE), WINDOW_SIZE))
model$backbone$save_pretrained(lora_path)
cat(sprintf("\n[DONE] head -> %s\n       backbone -> %s\n", head_path, lora_path))
