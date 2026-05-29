library(ggplot2)
library(dplyr)
library(tidyr)

# Auto-installazione e caricamento di cld2 per l'analisi linguistica
if (!require("cld2", quietly = TRUE)) {
  cat("[SYSTEM] Installazione del pacchetto 'cld2'...\n")
  install.packages("cld2", repos = "https://cloud.r-project.org/")
}
library(cld2)

# ==============================================================================
# SEZIONE 1: UTILS PER I DATI E IL MODELLO (Originariamente in data_utils.R)
# ==============================================================================

CHAT_TEMPLATE <- paste0(
  "{% for message in messages %}",
  "{{ '<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n' }}",
  "{% endfor %}",
  "{% if add_generation_prompt %}",
  "{{ '<|assistant|>\\n' }}",
  "{% endif %}"
)

load_tulu_dataset <- function(task_filter, max_samples = 1000L) {
  full_dataset <- datasets_lib$load_dataset("allenai/tulu-3-sft-mixture", split = "train")
  col_dict <- full_dataset$select_columns("source")$to_dict()
  sources <- col_dict[["source"]]
  indices <- which(sources == task_filter)
  if (length(indices) == 0) return(NULL)
  filtered <- full_dataset$select(as.integer(indices - 1L))
  filtered <- filtered$shuffle(seed = 42L)
  actual_max <- min(max_samples, as.integer(filtered$num_rows))
  filtered <- filtered$select(seq(0L, actual_max - 1L))
  splits <- filtered$train_test_split(test_size = 0.05)
  return(list(train = splits$train, test = splits$test))
}

preprocess_conversations <- function(messages_list, tokenizer, max_len = 4096L, template = CHAT_TEMPLATE) {
  n <- length(messages_list)
  all_input_ids      <- list()
  all_attention_mask <- list()
  all_labels         <- list()
  for (idx in seq_len(n)) {
    conversation <- messages_list[[idx]]
    full_text <- tokenizer$apply_chat_template(
      conversation, 
      chat_template = template,
      tokenize = FALSE, 
      add_generation_prompt = FALSE
    )
    encoding <- tokenizer(full_text, add_special_tokens = FALSE, truncation = TRUE, max_length = max_len)
    input_ids      <- as.integer(encoding$input_ids)
    attention_mask <- as.integer(encoding$attention_mask)
    labels         <- input_ids
    seq_len_actual <- length(input_ids)
    current_offset <- 0L
    for (msg in conversation) {
      role    <- as.character(msg$role)
      content <- as.character(msg$content)
      prefix  <- sprintf("<|%s|>\n", role)
      suffix  <- "<|end|>\n"
      msg_text <- paste0(prefix, content, suffix)
      msg_len    <- nchar(msg_text, type = "bytes")
      prefix_len <- nchar(prefix, type = "bytes")
      suffix_len <- nchar(suffix, type = "bytes")
      if (role != "assistant") {
        s <- current_offset + 1L
        e <- min(current_offset + msg_len, seq_len_actual)
        if (s <= seq_len_actual) labels[s:e] <- -100L
      } else {
        s <- current_offset + 1L
        e <- min(current_offset + prefix_len, seq_len_actual)
        if (s <= seq_len_actual) labels[s:e] <- -100L
        s <- current_offset + msg_len - suffix_len + 1L
        e <- min(current_offset + msg_len, seq_len_actual)
        if (s <= seq_len_actual) labels[s:e] <- -100L
      }
      current_offset <- current_offset + msg_len
      if (current_offset >= seq_len_actual) break
    }
    if (all(labels == -100L)) next
    all_input_ids[[length(all_input_ids) + 1L]]      <- input_ids
    all_attention_mask[[length(all_attention_mask) + 1L]] <- attention_mask
    all_labels[[length(all_labels) + 1L]]            <- labels
  }
  return(list(input_ids = all_input_ids, attention_mask = all_attention_mask, labels = all_labels))
}

prepare_train_data <- function(task_filter, tokenizer, max_len, batch_size, max_samples = 1000L) {
  splits <- load_tulu_dataset(task_filter, max_samples = max_samples)
  messages_list <- splits$train$to_dict()$messages
  processed <- preprocess_conversations(messages_list, tokenizer, max_len)
  return(create_batches(processed, batch_size))
}

init_probabilistic_circuit <- function(head_type, window_size, ranks) {
  if (head_type == "ff") {
    return(FFCircuit$new(window_size = window_size, ranks = ranks))
  } else if (head_type == "hmm") {
    return(HMMCircuit$new(window_size = window_size, ranks = ranks))
  } else if (head_type == "cp") {
    return(CPCircuit$new(window_size = window_size, ranks = ranks))
  } else {
    stop(sprintf("Tipo di testa non supportato: %s", head_type))
  }
}

get_model_paths <- function(head_type, window_size) {
  # Logica per gestire le inconsistenze nei nomi dei file salvati
  if (head_type == "ff") {
    lora_dir <- sprintf("saved_models/lora_ff_w%d/mtp_backbone_lora_ff_w%d", window_size, window_size)
    weights  <- sprintf("saved_models/mtp_head_ff_w%d_final.pth", window_size)
  } else if (head_type == "hmm") {
    lora_dir <- sprintf("saved_models/lora_hmm_w%d/mtp_backbone_lora_mtpc_hmm_w%d_ft", window_size, window_size)
    weights  <- sprintf("saved_models/mtp_head_mtpc_hmm_w%d_ft.pth", window_size)
  } else if (head_type == "cp") {
    lora_dir <- sprintf("saved_models/lora_cp_w%d/mtp_backbone_lora_cp_w%d", window_size, window_size)
    weights  <- sprintf("saved_models/mtp_head_cp_w%d_final.pth", window_size)
  } else {
    lora_dir <- sprintf("saved_models/lora_%s_w%d/mtp_backbone_lora_%s_w%d", head_type, window_size, head_type, window_size)
    weights  <- sprintf("saved_models/mtp_head_%s_w%d_final.pth", head_type, window_size)
  }
  
  return(list(lora_dir = lora_dir, weights_path = weights))
}


# ==============================================================================
# SEZIONE 2: ANALISI LINGUISTICA (Originariamente in text_analysis.R)
# ==============================================================================

#' Analizza la lingua del Dataset Ground Truth
analyze_dataset <- function(dataset, max_samples = 1000) {
  cat("[SYSTEM] Analisi linguistica del Dataset Ground Truth...\n")
  dataset_shuffled <- dataset$shuffle(seed = 42L)
  n_to_analyze <- min(max_samples, as.integer(dataset_shuffled$num_rows))
  
  all_messages <- dataset_shuffled$select(as.integer(0:(n_to_analyze-1)))$to_dict()$messages
  languages <- character(n_to_analyze)
  
  for (i in 1:n_to_analyze) {
    chat_text <- paste(sapply(all_messages[[i]], function(m) m$content), collapse = " ")
    languages[i] <- if (is.na(lang <- cld2::detect_language(substr(chat_text, 1, 1000)))) "unknown" else lang
  }
  
  print(table(languages))
  return(data.frame(language = languages, stringsAsFactors = FALSE))
}

#' Analizza l'accettazione media per ogni lingua rilevata
analyze_acceptance_by_language <- function(all_results, window_size = 4L) {
  cat("[SYSTEM] Calcolo performance medie per lingua...\n")
  
  full_report <- list()
  
  for (head_type in names(all_results)) {
    cat(sprintf("\nTesta: %s\n", toupper(head_type)))
    
    # Utilizziamo i prompt per il rilevamento lingua (più affidabile per il contesto)
    texts <- all_results[[head_type]]$prompt_texts
    matrix_acc <- all_results[[head_type]]$acceptance_matrix
    
    # Rilevamento lingua
    languages <- cld2::detect_language(texts)
    languages[is.na(languages)] <- "unknown"
    
    # Calcolo Acceptance Rate per campione (media dei round / finestra)
    sample_ar <- rowMeans(matrix_acc, na.rm = TRUE) / as.numeric(window_size)
    
    df <- data.frame(
      language = languages,
      ar = sample_ar,
      stringsAsFactors = FALSE
    )
    
    # Aggregazione per lingua
    report <- aggregate(ar ~ language, data = df, FUN = function(x) {
      c(mean = mean(x), count = length(x))
    })
    
    # Pulizia del formato
    report <- data.frame(
      language = report$language,
      mean_ar = report$ar[, "mean"],
      sample_count = report$ar[, "count"]
    )
    
    # Ordiniamo per AR decrescente
    report <- report[order(-report$mean_ar), ]
    
    print(report)
    full_report[[head_type]] <- report
  }
  
  return(full_report)
}


# ==============================================================================
# SEZIONE 3: ANALISI DEI RISULTATI (Originariamente in analyze_results.R)
# ==============================================================================

analyze_results <- function(file_path = "results_benchmark_w6.rds", window_size = 6) {
  if (!file.exists(file_path)) {
    stop(sprintf("File %s non trovato. Esegui prima il benchmark.", file_path))
  }
  
  all_results <- readRDS(file_path)
  
  cat("\n============================================================\n")
  cat("       REPORT ANALISI SPECULATIVE DECODING (W =", window_size, ")\n")
  cat("============================================================\n")
  
  # 1. Analisi Performance Linguistica
  cat("\n--- ANALISI LINGUISTICA AGGREGATA ---\n")
  lang_report <- analyze_acceptance_by_language(all_results, window_size = window_size)
  
  # 2. Analisi Dettagliata per Architettura
  cat("\n--- DETTAGLIO ARCHITETTURE ---\n")
  for (head_type in names(all_results)) {
    res <- all_results[[head_type]]
    mat <- res$acceptance_matrix
    
    # Metriche base
    valid_rounds <- sum(!is.na(mat))
    total_acc    <- sum(mat, na.rm = TRUE)
    total_prop   <- valid_rounds * window_size
    global_ar    <- (total_acc / total_prop) * 100
    
    cat(sprintf("\nARCHITETTURA: %s\n", toupper(head_type)))
    cat(sprintf("  - Global Acceptance Rate: %.2f%%\n", global_ar))
    cat(sprintf("  - Media Token per Round:  %.2f / %d\n", total_acc / valid_rounds, window_size))
    
    # Distribuzione
    counts <- table(factor(mat, levels = 0:window_size))
    cat("  - Distribuzione Token Accettati per Round:\n")
    for (val in names(counts)) {
      perc <- (counts[[val]] / valid_rounds) * 100
      cat(sprintf("    [%s] token: %4d round (%4.1f%%)\n", val, counts[[val]], perc))
    }
    
    # Esempio Best Case Qualitativo
    row_sums <- rowSums(mat, na.rm = TRUE)
    best_sample_idx <- which.max(row_sums)
    
    cat(sprintf("  - Miglior Esempio (Index %d): %d token bonus accettati\n", best_sample_idx, row_sums[best_sample_idx]))
    cat(sprintf("    Prompt: %s...\n", substr(res$prompt_texts[best_sample_idx], 1, 100)))
    cat(sprintf("    Output: %s...\n", substr(res$generated_texts[best_sample_idx], 1, 150)))
    cat("\n", paste(rep("-", 40), collapse=""), "\n")
  }
  
  cat("\n============================================================\n")
  cat("Report completato.\n")
}


# ==============================================================================
# SEZIONE 4: GATE DI ESECUZIONE DIRETTA (Per prevenire l'esecuzione automatica al source)
# ==============================================================================

is_run_directly <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_name <- sub("^--file=", "", file_arg)
    return(basename(script_name) == "data_utils.R")
  }
  return(FALSE)
}

if (is_run_directly()) {
  args <- commandArgs(trailingOnly = TRUE)
  file_to_analyze <- if (length(args) > 0) args[1] else "results_benchmark_w6.rds"
  
  window_size <- 6
  if (grepl("w4", file_to_analyze)) window_size <- 4
  if (grepl("w8", file_to_analyze)) window_size <- 8
  
  analyze_results(file_to_analyze, window_size = window_size)
}
