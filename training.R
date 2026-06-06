# training entry point for the mtpc speculative heads (ff / cp / hmm / btree)
source("mtpc/llm.R")
source("mtpc/utils.R")
source("utils.R")
source("mtpc/probabilistic_circuits.R")

MODEL_ID    = "google/byt5-small"
HEAD_TYPE   = "cp"        # one of "ff", "cp", "hmm", "btree"
WINDOW_SIZE = 6L
RANKS       = 32L
BATCH_SIZE  = 2L
MAX_LEN     = 512L
EPOCHS      = 1L
GAMMA       = 0.8
TASK_FILTER = "ai2-adapt-dev/flan_v2_converted"
MAX_SAMPLES = 20000L
SAVE_DIR    = "saved_models"

PHASE0_LR      = 5e-4
PHASE1_HEAD_LR = 1e-3
PHASE1_LLM_LR  = 1e-4
PHASE2_HEAD_LR = 3e-4
PHASE2_LLM_LR  = 1e-4

# skipping a phase reuses its saved checkpoint instead of retraining it
SKIP_PHASE_0 = TRUE
SKIP_PHASE_1 = TRUE

device = get_device()
dir.create(SAVE_DIR, showWarnings = FALSE, recursive = TRUE)

tokenizer = transformers$AutoTokenizer$from_pretrained(MODEL_ID)
tokenizer$chat_template = CHAT_TEMPLATE
batches = prepare_train_data(TASK_FILTER, tokenizer, MAX_LEN, BATCH_SIZE, max_samples = MAX_SAMPLES)

# copies the trained ff emissions into the target circuit, leaving the sum gates uniform
init_target_from_ff = function(model, ff_head_path, head_type, window_size, ranks) {
  if (!file.exists(ff_head_path)) return(invisible(FALSE))
  ff_sd = torch$load(ff_head_path, map_location = "cpu")
  ht = tolower(head_type)
  if (ht == "ff") {
    model$heads$load_state_dict(ff_sd)
    return(invisible(TRUE))
  }
  with(torch$no_grad(), {
    target_w = model$heads[["input_units_phi"]]$weight
    H = as.integer(target_w$shape[1])
    ff_w = lapply(seq_len(window_size), function(t) ff_sd[[sprintf("input_units_phi_%d.weight", t)]])
    emiss = torch$stack(ff_w, dim = 0L)
    # cp uses [ranks, window, vocab]; hmm and btree use [window, ranks, vocab]
    if (ht == "cp") {
      emiss = emiss$unsqueeze(0L)$expand(ranks, -1L, -1L, -1L)$reshape(-1L, H)
    } else {
      emiss = emiss$unsqueeze(1L)$expand(-1L, ranks, -1L, -1L)$reshape(-1L, H)
    }
    emiss = emiss$add(torch$randn_like(emiss)$mul(1e-4))
    target_w$copy_(emiss$to(target_w$device))
    nn$init$zeros_(model$heads[["input_units_phi"]]$bias)
  })
  invisible(TRUE)
}

# runs one training phase, accumulating gradients to an effective batch of 16
training_loop = function(model, optimizer, batches, phase, epochs = 1L,
                         window_size = 6L, gamma = 0.8, is_log_probs = FALSE, device) {
  accumulation_steps = max(1L, as.integer(16L / BATCH_SIZE))
  for (epoch in seq_len(epochs)) {
    optimizer$zero_grad()
    for (i in seq_along(batches)) {
      batch = batches[[i]]
      input_ids      = batch$input_ids$to(device)
      labels         = batch$labels$to(device)
      attention_mask = batch$attention_mask$to(device)
      if (phase == 0L) {
        # mask assistant tokens in the encoder input to prevent leakage
        decoder_start_token_id = model$backbone$config$decoder_start_token_id
        encoder_input_ids = input_ids$clone()
        encoder_input_ids[labels$ne(-100L)] = as.integer(decoder_start_token_id)
        loss = model$backbone(input_ids = encoder_input_ids,
                              attention_mask = attention_mask, labels = labels)$loss
      } else {
        dec_in = model$prepare_decoder_input_ids(input_ids)
        hidden_states = model$get_hidden_states(input_ids, dec_in,
                          attention_mask = attention_mask, labels = labels)$x
        mtp_logits = model$circuit$forward(model, hidden_states)
        loss = compute_mtpc_loss(mtp_logits, labels, window_size, gamma, is_log_probs = is_log_probs)
      }
      loss$div(accumulation_steps)$backward()
      if (i %% accumulation_steps == 0L || i == length(batches)) {
        optimizer$step()
        optimizer$zero_grad()
      }
    }
  }
}

# phase 2 resumes the ff-warmed backbone; earlier phases resume the prior checkpoint
resume_lora = file.path(SAVE_DIR, sprintf("lora_ff_w%d_phase1", WINDOW_SIZE))
if (!SKIP_PHASE_0) {
  resume_lora = NULL
} else if (!SKIP_PHASE_1) {
  resume_lora = file.path(SAVE_DIR, "byt5_standard_lora_phase0")
}
if (!is.null(resume_lora) && !dir.exists(resume_lora)) {
  resume_lora = file.path(SAVE_DIR, "byt5_standard_lora_phase0")
  if (!dir.exists(resume_lora)) resume_lora = NULL
}

model = LLMWrapper(model_id = MODEL_ID, lora_path = resume_lora, lora_r = 8L, lora_alpha = 16L)
model$to(device)

# phase 0: autoregressive backbone fine-tuning
if (!SKIP_PHASE_0) {
  model$train()
  model$heads$eval()
  optimizer = torch$optim$Adam(model$get_parameter_groups(0, PHASE0_LR)[[2]]$params, lr = PHASE0_LR)
  training_loop(model, optimizer, batches, phase = 0L, epochs = EPOCHS, device = device)
  model$backbone$save_pretrained(file.path(SAVE_DIR, "byt5_standard_lora_phase0"))
}

# phase 1: feed-forward warm-up with emissions initialised from the backbone stp matrix
if (!SKIP_PHASE_1) {
  model$swap_head("ff", WINDOW_SIZE, RANKS)
  model$to(device)
  model$train()
  optimizer = torch$optim$Adam(model$get_parameter_groups(PHASE1_HEAD_LR, PHASE1_LLM_LR))
  training_loop(model, optimizer, batches, phase = 1L, epochs = EPOCHS,
                window_size = WINDOW_SIZE, gamma = GAMMA, is_log_probs = FALSE, device = device)
  torch$save(model$heads$state_dict(), file.path(SAVE_DIR, sprintf("mtp_head_ff_w%d_phase1.pth", WINDOW_SIZE)))
  model$backbone$save_pretrained(file.path(SAVE_DIR, sprintf("lora_ff_w%d_phase1", WINDOW_SIZE)))
}

if (exists("optimizer")) rm(optimizer)
gc()
if (device$type == "mps")  torch$mps$empty_cache()
if (device$type == "cuda") torch$cuda$empty_cache()

# phase 2: target circuit initialised from the trained ff head, then trained jointly
model$swap_head(HEAD_TYPE, WINDOW_SIZE, RANKS)
model$to(device)
init_target_from_ff(model, file.path(SAVE_DIR, sprintf("mtp_head_ff_w%d_phase1.pth", WINDOW_SIZE)),
                    HEAD_TYPE, WINDOW_SIZE, RANKS)
model$train()

is_log_probs = tolower(HEAD_TYPE) %in% c("cp", "hmm", "btree")
optimizer = torch$optim$Adam(model$get_parameter_groups(PHASE2_HEAD_LR, PHASE2_LLM_LR))
training_loop(model, optimizer, batches, phase = 2L, epochs = EPOCHS,
              window_size = WINDOW_SIZE, gamma = GAMMA, is_log_probs = is_log_probs, device = device)

torch$save(model$heads$state_dict(),
           file.path(SAVE_DIR, sprintf("mtp_head_%s_w%d_final.pth", tolower(HEAD_TYPE), WINDOW_SIZE)))
model$backbone$save_pretrained(file.path(SAVE_DIR, sprintf("lora_%s_w%d", tolower(HEAD_TYPE), WINDOW_SIZE),
                                         sprintf("mtp_backbone_lora_%s_w%d", tolower(HEAD_TYPE), WINDOW_SIZE)))
