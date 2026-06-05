source("mtpc/llm.R")
source("mtpc/utils.R")
source("utils.R")
source("mtpc/probabilistic_circuits.R")

# target models and architectures
MODEL_ID    = "google/byt5-small"
HEAD_TYPE   = "hmm" # ("cp", "ff", "hmm")
WINDOW_SIZE = 6L # (number of future tokens)
RANKS       = 32L # number of hidden states
BATCH_SIZE  = 2L  
MAX_LEN     = 512L # maximum sequence length
EPOCHS      = 1L # number of epochs per phase
GAMMA       = 0.8  # MTPC loss sequence discount factor
TASK_FILTER = "ai2-adapt-dev/flan_v2_converted" #easier task
 
# learning rates for the 3 distinct training phases
PHASE0_LR       = 5e-5    # Backbone-only autoregressive fine-tuning LR
PHASE1_HEAD_LR  = 1e-3    # Phase 1: FF head warm-up LR
PHASE1_LLM_LR   = 5e-5    # Phase 1: Backbone LoRA adapter LR
PHASE2_HEAD_LR  = 3e-4    # Phase 2: Joint target circuit head LR
PHASE2_LLM_LR   = 1e-5    # Phase 2: Joint backbone LoRA adapter LR (lowered for stability)
 
MAX_SAMPLES = 20000L
device = get_device()

#DATASET SETUP & PREPROCESSING

tokenizer = transformers$AutoTokenizer$from_pretrained(MODEL_ID)
tokenizer$chat_template = CHAT_TEMPLATE

# Prepare training data batches 
batches = prepare_train_data(TASK_FILTER, tokenizer, MAX_LEN, BATCH_SIZE, max_samples = MAX_SAMPLES)

training_loop = function(model, optimizer, batches, phase, epochs = 1L, window_size = 6L, gamma = 0.8, is_log_probs = FALSE, device) {
  # runs the training loop for a phase, optimizing the backbone and/or circuit heads
  # Automatically compute accumulation steps to target an effective batch size of 16
  accumulation_steps = max(1L, as.integer(16L / BATCH_SIZE))
  cat(sprintf("[SYSTEM] Using micro-batch size = %d | Gradient accumulation steps = %d (Effective batch size = %d)\n", BATCH_SIZE, accumulation_steps, BATCH_SIZE * accumulation_steps))
  
  losses = numeric(length(batches) * epochs)
  idx = 1

  for (epoch in seq_len(epochs)) {
    total_loss = 0.0
    optimizer$zero_grad()
    for (i in seq_along(batches)) {
      batch = batches[[i]]
      # input_ids shape: [batch_size, seq_len], labels shape: [batch_size, seq_len]
      input_ids = batch$input_ids$to(device)
      labels = batch$labels$to(device)
      attention_mask = batch$attention_mask$to(device)
      
      if (phase == 0L) {
        # Mask assistant tokens in the encoder input to prevent future token leakage
        decoder_start_token_id = model$backbone$config$decoder_start_token_id
        encoder_input_ids = input_ids$clone()
        encoder_input_ids[labels$ne(-100L)] = as.integer(decoder_start_token_id)
        
        # standard Seq2Seq forward pass through the backbone (calculates autoregressive loss)
        outputs = model$backbone(
          input_ids = encoder_input_ids,
          attention_mask = attention_mask,
          labels = labels
        )
        loss = outputs$loss
      } else {
        # prep decoder input IDs 
        dec_in = model$prepare_decoder_input_ids(input_ids)
        
        # get hidden states and compute speculative circuit logits
        hidden_states = model$get_hidden_states(input_ids, dec_in, attention_mask = attention_mask, labels = labels)$x # [batch_size, seq_len, embed_dim]
        mtp_logits = model$circuit$forward(model, hidden_states) # [batch_size, seq_len, window_size, vocab_size]
        
        # Speculative sequence loss
        loss = compute_mtpc_loss(mtp_logits, labels, window_size, gamma, is_log_probs = is_log_probs)
      }
      
      # Normalize loss by accumulation steps
      scaled_loss = loss$div(accumulation_steps)
      scaled_loss$backward()
      
      if (i %% accumulation_steps == 0L || i == length(batches)) {
        optimizer$step()
        optimizer$zero_grad()
      }
      
      loss_val = loss$item()
      total_loss = total_loss + loss_val
      losses[idx] = loss_val
      idx = idx + 1
      
      if (i %% 5L == 0L || i == length(batches)) { cat(sprintf("  Phase %d | Batch %d/%d | Loss: %.4f | Avg Loss: %.4f\n", phase, i, length(batches), loss_val, total_loss / i)); flush.console() }
    }
  }

  # Plot losses at the end of the phase
  tryCatch({
    dir.create("saved_models", showWarnings = FALSE, recursive = TRUE)
    png_path = file.path("saved_models", sprintf("loss_phase%d.png", phase))
    png(png_path)
    plot(losses, type = "l", col = "blue", xlab = "Iteration", ylab = "Loss", 
         main = sprintf("Phase %d Training Loss", phase))
    dev.off()
    cat(sprintf("[SYSTEM] Phase %d loss plot saved to %s\n", phase, png_path))
  }, error = function(e) {
    cat(sprintf("[WARNING] Failed to plot Phase %d loss: %s\n", phase, e$message))
  })
}


SKIP_PHASE_0 = TRUE  
SKIP_PHASE_1 = TRUE
USE_PRETRAINED = TRUE  
RESUME_LORA_PATH = "saved_models/byt5_standard_lora_phase0"

# Instantiate speculative model wrapper (explicitly configuring LoRA parameters for training)
model = LLMWrapper(
  model_id = MODEL_ID,
  lora_path = if (SKIP_PHASE_0 && USE_PRETRAINED) RESUME_LORA_PATH else NULL,
  lora_r = 8L,
  lora_alpha = 16L
)
model$to(device)

# PHASE 0: AUTOREGRESSIVE LLM BACKBONE FINE-TUNING ONLY
if (SKIP_PHASE_0) {
  cat("\n[SYSTEM] Skipping PHASE 0: Pre-trained LoRA backbone loaded successfully.\n")
} else {
  cat("PHASE 0: Standard LoRA Backbone Autoregressive Fine-Tuning\n")
  
  model$train()
  model$heads$eval()
  
  lora_params = model$get_parameter_groups(0, PHASE0_LR)[[2]]$params
  
  optimizer = torch$optim$Adam(lora_params, lr = PHASE0_LR)
  
  # run training loop for Phase 0
  training_loop(
    model = model,
    optimizer = optimizer,
    batches = batches,
    phase = 0L,
    epochs = EPOCHS,
    device = device
  )
  
  # Checkpoint automatico per la Fase 0
  dir.create("saved_models", showWarnings = FALSE, recursive = TRUE)
  phase0_save_path = file.path("saved_models", "byt5_standard_lora_phase0")
  model$backbone$save_pretrained(phase0_save_path)
  cat(sprintf("\n[SYSTEM] Phase 0 Backbone LoRA adapter salvato in: %s\n\n", phase0_save_path))
}

#PHASE 1: SPECULATIVE WARM-UP WITH FEED-FORWARD (FF) HEAD
if (SKIP_PHASE_1) {
  cat("\n[SYSTEM] Skipping PHASE 1: Starting HMM training directly on top of pre-loaded backbone.\n")
} else {
  cat("PHASE 1: Speculative Warm-up with Feed-Forward (FF) Head\n")
  
  # swap to Feed-Forward head
  model$swap_head("ff", WINDOW_SIZE, RANKS)
  model$to(device)
  model$train()
  
  param_groups = model$get_parameter_groups(PHASE1_HEAD_LR, PHASE1_LLM_LR)
  optimizer = torch$optim$Adam(param_groups)
  
  # run training loop for Phase 1
  training_loop(
    model = model,
    optimizer = optimizer,
    batches = batches,
    phase = 1L,
    epochs       = EPOCHS,
    window_size  = WINDOW_SIZE,
    gamma        = GAMMA,
    is_log_probs = FALSE,
    device       = device
  )
  
  # Checkpoint automatico per la Fase 1
  torch$save(model$heads$state_dict(), file.path("saved_models", sprintf("mtp_head_ff_w%d_phase1.pth", WINDOW_SIZE)))
  model$backbone$save_pretrained(file.path("saved_models", sprintf("lora_ff_w%d_phase1", WINDOW_SIZE)))
  cat(sprintf("\n[SYSTEM] Phase 1 FF Head e Backbone salvati con successo!\n\n"))
}

# PHASE 2: TARGET PROBABILISTIC HEAD JOINT TRAINING
cat(sprintf("PHASE 2: Target %s Joint Training with Differential LRs\n", toupper(HEAD_TYPE)))

# clear memory
if (exists("optimizer")) rm(optimizer)
# force garbage collection and flush device caches to free VRAM before allocating the new head
gc()
if (device$type == "mps") torch$mps$empty_cache()
if (device$type == "cuda") torch$cuda$empty_cache()

# swap to the target probabilistic circuit
model$swap_head(HEAD_TYPE, WINDOW_SIZE, RANKS)
model$to(device)
model$train()

param_groups = model$get_parameter_groups(PHASE2_HEAD_LR, PHASE2_LLM_LR)
optimizer = torch$optim$Adam(param_groups)

is_log_probs = tolower(HEAD_TYPE) %in% c("cp", "hmm")

# run training loop for Phase 2
training_loop(
  model = model,
  optimizer = optimizer,
  batches = batches,
  phase = 2L,
  epochs = EPOCHS,
  window_size = WINDOW_SIZE,
  gamma = GAMMA,
  is_log_probs = is_log_probs,
  device = device
)

# SAVING MODEL WEIGHTS and ADAPTERS

# persist circuit head weights and backbone LoRA adapter separately for modular reloading
dir.create("saved_models", showWarnings = FALSE, recursive = TRUE)

filename = sprintf("mtp_head_%s_w%d_final.pth", tolower(HEAD_TYPE), WINDOW_SIZE)
save_path = file.path("saved_models", filename)
torch$save(model$heads$state_dict(), save_path)

lora_save_path = file.path("saved_models", sprintf("lora_%s_w%d", tolower(HEAD_TYPE), WINDOW_SIZE), sprintf("mtp_backbone_lora_%s_w%d", tolower(HEAD_TYPE), WINDOW_SIZE))
model$backbone$save_pretrained(lora_save_path)
