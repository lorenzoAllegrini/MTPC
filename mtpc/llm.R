library(reticulate)
if (dir.exists(".venv")) {
  use_virtualenv(file.path(getwd(), ".venv"), required = TRUE)
} else if (dir.exists("../.venv")) {
  use_virtualenv(file.path(getwd(), "..", ".venv"), required = TRUE)
}

torch        <- import("torch")
nn           <- import("torch.nn")
F_           <- import("torch.nn.functional")
transformers <- import("transformers")
datasets_lib <- import("datasets")

# --- CLASSE SPECULATIVE ENGINE NATIVA IN R ---
SpeculativeEngine <- setRefClass("SpeculativeEngine",
  fields = list(
    backbone   = "ANY",       # Modello PyTorch Python (T5/ByT5)
    heads      = "ANY",       # nn.ModuleDict Python
    embed_dim  = "integer",   # Dimensione embedding del backbone
    vocab_size = "integer"    # Dimensione del vocabolario
  ),
  methods = list(
    initialize = function(model_id, lora_path = NULL) {
      best_dtype <- if (torch$cuda$is_available()) {
        torch$bfloat16
      } else {
        torch$float32
      }

      cat("[SYSTEM] Inizializzazione SpeculativeEngine R Nativo per:", model_id, "\n")
      
      # Caricamento del backbone tramite Hugging Face Transformers
      backbone <<- transformers$T5ForConditionalGeneration$from_pretrained(
        model_id,
        device_map = "auto",
        torch_dtype = best_dtype
      )

      # Integrazione LoRA se specificata
      if (!is.null(lora_path)) {
        cat("[SYSTEM] Caricamento adattatore LoRA da:", lora_path, "\n")
        peft <- reticulate::import("peft")
        backbone <<- peft$PeftModel$from_pretrained(backbone, lora_path)
      }

      # Creazione heads come ModuleDict nativo
      heads <<- nn$ModuleDict()

      # Salvataggio metadati
      embed_dim  <<- as.integer(backbone$config$d_model)
      vocab_size <<- as.integer(backbone$config$vocab_size)
    },

    to = function(device) {
      backbone$to(device)
      heads$to(device)
      invisible(.self)
    },

    eval = function() {
      backbone$eval()
      heads$eval()
      invisible(.self)
    },

    train = function(mode = TRUE) {
      backbone$train(mode)
      heads$train(mode)
      invisible(.self)
    },

    get_hidden_states = function(prompt_ids, decoder_ids, attention_mask = NULL, encoder_outputs = NULL) {
      if (!is.null(encoder_outputs) && !inherits(encoder_outputs, "python.builtin.tuple")) {
        encoder_outputs <- reticulate::tuple(encoder_outputs[[1]])
      }

      outputs <- backbone(
        input_ids = if (is.null(encoder_outputs)) prompt_ids else NULL,
        attention_mask = attention_mask,
        decoder_input_ids = decoder_ids,
        encoder_outputs = encoder_outputs,
        use_cache = FALSE,
        output_hidden_states = TRUE
      )
      
      h_states <- outputs$decoder_hidden_states
      
      if (!is.null(h_states)) {
        hidden_states <- h_states[[length(h_states)]]
      } else {
        hidden_states <- outputs$last_hidden_state
      }

      return(list(
        x = hidden_states
      ))
    },

    verify_draft = function(draft_tokens, encoder_outputs, decoder_ids, prompt_ids, attention_mask = NULL) {
      N <- length(draft_tokens)
      
      reticulate::with(torch$no_grad(), {
        draft_tensor <- torch$tensor(matrix(draft_tokens, nrow = 1), dtype = torch$long, device = backbone$device)
        next_ids <- torch$cat(list(decoder_ids, draft_tensor), dim = 1L)
        
        has_peft <- reticulate::py_has_attr(backbone, "disable_adapter")
        
        if (!is.null(encoder_outputs) && !inherits(encoder_outputs, "python.builtin.tuple")) {
          encoder_outputs <- reticulate::tuple(encoder_outputs[[1]])
        }

        if (has_peft) {
          outputs <- reticulate::with(backbone$disable_adapter(), {
            backbone(
              input_ids = if (is.null(encoder_outputs)) prompt_ids else NULL,
              attention_mask = attention_mask,
              decoder_input_ids = next_ids,
              encoder_outputs = encoder_outputs,
              use_cache = FALSE
            )
          })
        } else {
          outputs <- backbone(
            input_ids = if (is.null(encoder_outputs)) prompt_ids else NULL,
            attention_mask = attention_mask,
            decoder_input_ids = next_ids,
            encoder_outputs = encoder_outputs,
            use_cache = FALSE
          )
        }
        
        all_logits <- outputs$logits
        L <- as.integer(decoder_ids$size(1L))
        
        relevant_logits <- all_logits$narrow(1L, as.integer(L - 1L), as.integer(N + 1L))
        all_p_dist <- F_$softmax(relevant_logits, dim = -1L)
        
        draft_p_dist <- all_p_dist$narrow(1L, 0L, as.integer(N))
        p_vals <- draft_p_dist$gather(dim = 2L, index = draft_tensor$unsqueeze(-1L))$squeeze(-1L)
        next_p_dist <- all_p_dist$select(1L, as.integer(N))
        
        list(
          p = p_vals$squeeze(0L)$cpu()$numpy(),
          next_p = next_p_dist,
          full_p_dist = all_p_dist
        )
      })
    },

    load_weights = function(weights_path, device = NULL) {
      if (is.null(device)) {
        device <- backbone$device
      }
      state_dict <- torch$load(weights_path, map_location = device, weights_only = TRUE)
      heads$load_state_dict(state_dict)
      invisible(.self)
    },

    save_weights = function(save_path) {
      torch$save(heads$state_dict(), save_path)
      invisible(.self)
    }
  )
)

# --- FUNZIONI DI COMPATIBILITA RETROATTIVA ---
engine_get_hidden_states <- function(model, prompt_ids, decoder_ids, attention_mask = NULL, encoder_outputs = NULL) {
  model$get_hidden_states(prompt_ids, decoder_ids, attention_mask, encoder_outputs)
}

engine_verify_draft <- function(model, draft_tokens, encoder_outputs, decoder_ids, prompt_ids, attention_mask = NULL) {
  model$verify_draft(draft_tokens, encoder_outputs, decoder_ids, prompt_ids, attention_mask)
}