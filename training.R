source("mtpc/llm.R")
source("mtpc/utils.R")
source("data_utils.R")
source("mtpc/probabilistic_circuits.R")

MODEL_ID    <- "google/byt5-small"
HEAD_TYPE   <- "cp"
WINDOW_SIZE <- 8L
RANKS       <- 32L
BATCH_SIZE  <- 2L
MAX_LEN     <- 2048L    
EPOCHS      <- 1L

PHASE1_LR          <- 0.001
PHASE2_HEAD_LR     <- 5e-4
PHASE2_LLM_LR      <- 5e-5

GAMMA       <- 0.9
TASK_FILTER <- "ai2-adapt-dev/flan_v2_converted"

device <- get_device()
cat(sprintf("\n[SYSTEM] Training on Device: %s\n", device$type))

training_loop <- function(model, circuit, optimizer, batches, epochs, WINDOW_SIZE, GAMMA, device) {
  for (epoch in seq_len(epochs)) {
    cat(sprintf("\n=== Epoch %d/%d ===\n", epoch, epochs))
    total_loss <- 0.0
    n_batches  <- length(batches)

    for (i in seq_len(n_batches)) {
      batch <- batches[[i]]
      input_ids      <- batch$input_ids$to(device)
      attention_mask <- batch$attention_mask$to(device)
      labels         <- batch$labels$to(device)

      optimizer$zero_grad()
      
      decoder_start_token_id <- as.integer(model$backbone$config$decoder_start_token_id)
      batch_size <- as.integer(input_ids$size(0L))
      start_tokens <- torch$full(list(batch_size, 1L), decoder_start_token_id, dtype=torch$long, device=device)
      shifted_ids <- input_ids$narrow(1L, 0L, as.integer(input_ids$size(1L) - 1L))
      decoder_input_ids <- torch$cat(list(start_tokens, shifted_ids), dim = 1L)

      res_h <- engine_get_hidden_states(
        model, 
        prompt_ids = input_ids, 
        decoder_ids = decoder_input_ids,
        attention_mask = attention_mask
      )
      hidden_states <- res_h$x
      mtp_logits <- circuit$forward(model, hidden_states)
      loss <- compute_mtpc_loss(
        mtp_logits, 
        labels, 
        WINDOW_SIZE, 
        GAMMA, 
        is_log_probs = circuit$is_log_probs
      )

      if (!is.null(loss$grad_fn)) {
        loss$backward()
        optimizer$step()
      }

      batch_loss <- loss$item()
      total_loss <- total_loss + batch_loss

      if (i %% 10L == 0L || i == n_batches) {
        cat(sprintf("\r  Batch %d/%d | Loss: %.4f | Avg: %.4f", i, n_batches, batch_loss, total_loss / i))
      }
    }
    cat(sprintf("\nEpoch %d completata. Loss Media: %.4f\n", epoch, total_loss / n_batches))
  }
}

get_grouped_params <- function(model) {
  circuit_params <- reticulate::iterate(model$heads$parameters())
  lora_params <- list()
  all_backbone_params <- reticulate::iterate(model$backbone$named_parameters())
  for (p in all_backbone_params) {
    name  <- p[[1]]
    param <- p[[2]]
    is_lora <- grepl("lora_", name)
    param$requires_grad <- is_lora
    if (is_lora) lora_params <- c(lora_params, list(param))
  }
  return(list(circuit = circuit_params, lora = lora_params))
}

model <- SpeculativeEngine(MODEL_ID)
peft <- import("peft")
lora_config <- peft$LoraConfig(
  r = 8L, 
  lora_alpha = 16L, 
  bias = "none", 
  task_type = "SEQ_2_SEQ_LM"
)
model$backbone <- peft$get_peft_model(model$backbone, lora_config)

tokenizer <- transformers$AutoTokenizer$from_pretrained(MODEL_ID)
tokenizer$chat_template <- CHAT_TEMPLATE
batches <- prepare_train_data(TASK_FILTER, tokenizer, MAX_LEN, BATCH_SIZE, max_samples = 100L)

# --- FASE 1: Warm-up con teste Feed-Forward ---
cat("\n[FASE 1] Warm-up con teste Feed-Forward (FF)...\n")
circuit_ff <- create_circuit("ff", WINDOW_SIZE)
circuit_ff$inject_head(model)
model$to(device)
model$heads$train()
model$backbone$train() # Iniziamo ad addestrare LoRA già da qui

params <- get_grouped_params(model)
optimizer <- torch$optim$Adam(params$circuit, lr = PHASE1_LR)
training_loop(model, circuit_ff, optimizer, batches, EPOCHS, WINDOW_SIZE, GAMMA, device)

# --- FASE 2: Transizione a HMM e Joint Finetuning ---
cat("\n[FASE 2] Switch ad architettura HMM e Joint Training...\n")
circuit_hmm <- create_circuit(HEAD_TYPE, WINDOW_SIZE, RANKS)
circuit_hmm$inject_head(model) # Sostituisce FF con HMM in model$heads
model$to(device)
model$heads$train()
model$backbone$train()

params <- get_grouped_params(model)
param_groups <- list(
  list(params = params$circuit, lr = PHASE2_HEAD_LR),
  list(params = params$lora,    lr = PHASE2_LLM_LR)
)
optimizer <- torch$optim$Adam(param_groups)
training_loop(model, circuit_hmm, optimizer, batches, EPOCHS, WINDOW_SIZE, GAMMA, device)

save_model(model, HEAD_TYPE, WINDOW_SIZE)
cat("\n[SYSTEM] Training terminato.\n")
