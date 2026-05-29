source("mtpc/llm.R")
source("mtpc/utils.R")
source("data_utils.R")
source("mtpc/probabilistic_circuits.R")
source("mtpc/speculative_decoding.R")

torch_py <- import("torch")
datasets_py <- import("datasets")

# --- CONFIGURAZIONE ---
MODEL_ID           <- "google/byt5-small"
PROBABILISTIC_HEADS <- c("cp", "ff", "hmm") 
WINDOW_SIZE        <- 6L
RANKS              <- 32L
MAX_LEN            <- 2048L
N_SAMPLES          <- 100L

CHAT_TEMPLATE <- "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"

#' Funzione Core per il Testing dell'Inferenza
run_inference_experiment <- function(dataset, verifier_model, draft_model, circuit, tokenizer, n_samples = 100, max_new_tokens = 60L) {
  set.seed(42)
  total_available <- as.integer(dataset$num_rows)
  sample_indices <- sample(1:total_available, min(n_samples, total_available))
  
  global_accepted <- 0
  global_proposed <- 0
  acceptance_list <- list()
  generated_texts <- character(0)
  prompt_texts    <- character(0)
  
  device <- verifier_model$backbone$device

  for (i in seq_along(sample_indices)) {
    sample_idx <- sample_indices[i]
    sample_data <- dataset[as.integer(sample_idx - 1)]
    
    cat(sprintf("\nESEMPIO %d/%d (Index: %d)\n", i, length(sample_indices), sample_idx))
    
    messages <- sample_data$messages
    n_msgs <- length(messages)
    prompt_msgs <- messages[1:(n_msgs-1)]
    
    prompt_text <- tokenizer$apply_chat_template(prompt_msgs, chat_template=CHAT_TEMPLATE, tokenize=FALSE, add_generation_prompt=TRUE)
    prompt_texts <- c(prompt_texts, prompt_text)
    
    target_full_text <- messages[[n_msgs]]$content
    prefix_text <- substr(target_full_text, 1, 10)
    
    input_ids <- tokenizer$encode(prompt_text, return_tensors="pt")$to(device)
    initial_decoder_ids <- tokenizer$encode(prefix_text, add_special_tokens=FALSE, return_tensors="pt")$to(device)
    
    res_obj <- generate_speculative(
      verifier_model = verifier_model,
      draft_model    = draft_model,
      circuit        = circuit,
      prompt_ids     = input_ids,
      max_new_tokens = max_new_tokens,
      tokenizer      = tokenizer,
      initial_decoder_ids = initial_decoder_ids,
      verbose = FALSE
    )
    
    acceptance_list[[i]] <- res_obj$round_accepted
    final_text <- tokenizer$decode(as.integer(res_obj$tokens))
    generated_texts <- c(generated_texts, final_text)
    
    global_accepted <- global_accepted + res_obj$total_accepted
    global_proposed <- global_proposed + res_obj$total_proposed
    
    cat(sprintf("[STATS] Round: %d | Mean Acceptance: %.2f%%\n", length(res_obj$round_accepted), res_obj$mean_acceptance * 100))
  }
  
  # --- NA PADDING (Distinzione tra 0 accettati e round non avvenuti) ---
  max_rounds <- max(sapply(acceptance_list, length))
  results_matrix <- matrix(NA, nrow = length(acceptance_list), ncol = max_rounds)
  for (i in seq_along(acceptance_list)) {
    curr_len <- length(acceptance_list[[i]])
    if (curr_len > 0) {
      results_matrix[i, 1:curr_len] <- acceptance_list[[i]]
    }
  }
  
  cat(sprintf("\n[SYSTEM] Experiment terminato. Global Acceptance: %.2f%%\n", (global_accepted / global_proposed) * 100))
  
  return(list(
    acceptance_matrix = results_matrix,
    generated_texts   = generated_texts,
    prompt_texts      = prompt_texts
  ))
}


device <- get_device()
tokenizer <- transformers$AutoTokenizer$from_pretrained(MODEL_ID)

# ==============================================================================
# FASE 1: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("FASE 1: EXPLORATORY DATA ANALYSIS (EDA)\n")
cat(paste(rep("=", 60), collapse=""), "\n")

cat("[SYSTEM] Caricamento e filtraggio dataset...\n")
dataset_full <- datasets_lib$load_dataset("allenai/tulu-3-sft-mixture", split="train")
col_dict <- dataset_full$select_columns("source")$to_dict()
indices <- which(col_dict[["source"]] == "ai2-adapt-dev/flan_v2_converted")
dataset <- dataset_full$select(as.integer(indices - 1))

# Analisi linguistica del ground truth (campione di 500)
gt_stats <- analyze_dataset(dataset, max_samples = 500)

cat("\n[OK] EDA Completata.\n")


# ==============================================================================
# FASE 2: INFERENCE & BENCHMARKING
# ==============================================================================
cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("FASE 2: INFERENCE & BENCHMARKING\n")
cat(paste(rep("=", 60), collapse=""), "\n")

# Caricamento Verifier (Comune a tutti)
VERIFIER_LORA_DIR <- "saved_models/byt5_standard_lora_verifier/byt5_standard_lora"
cat("[SYSTEM] Inizializzazione Verifier Model...\n")
verifier_model <- SpeculativeEngine(model_id = MODEL_ID, lora_path = VERIFIER_LORA_DIR)
verifier_model$to(device)
verifier_model$eval()

# Contenitore per tutti i risultati
all_results <- list()

for (head_type in PROBABILISTIC_HEADS) {
  cat(sprintf("\n>>> TEST ARCHITETTURA: %s <<<\n", toupper(head_type)))
  
  # Caricamento Modello di Draft specifico
  paths <- get_model_paths(head_type, WINDOW_SIZE)
  draft_model <- SpeculativeEngine(model_id = MODEL_ID, lora_path = paths$lora_dir)
  
  # Iniezione e caricamento pesi circuito
  circuit <- init_probabilistic_circuit(head_type, WINDOW_SIZE, RANKS)
  circuit$inject_head(draft_model) 
  
  if (file.exists(paths$weights_path)) {
    cat(paste("Caricamento pesi MTP Head da:", paths$weights_path, "\n"))
    state_dict <- torch_py$load(paths$weights_path, map_location = "cpu")
    draft_model$heads$load_state_dict(state_dict)
  } else {
    cat(sprintf("[WARNING] Pesi non trovati a %s, uso inizializzazione casuale.\n", paths$weights_path))
  }
  
  draft_model$to(device)
  draft_model$eval()
  
  # Esecuzione Esperimento
  all_results[[head_type]] <- run_inference_experiment(
    dataset = dataset,
    verifier_model = verifier_model,
    draft_model = draft_model,
    circuit = circuit,
    tokenizer = tokenizer,
    n_samples = N_SAMPLES
  )
  
  # Pulizia memoria tra i test
  rm(draft_model, circuit)
  gc()
}


# ==============================================================================
# SALVATAGGIO DEI RISULTATI
# ==============================================================================
output_file <- sprintf("results_benchmark_w%d.rds", WINDOW_SIZE)
saveRDS(all_results, output_file)

cat(sprintf("\n[SYSTEM] Pipeline completata. Risultati salvati in: %s\n", output_file))
cat("[SYSTEM] Usa 'Rscript analyze_results.R' per visualizzare il report dettagliato.\n")
