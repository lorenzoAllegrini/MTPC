library(reticulate)
if (dir.exists(".venv")) {
  use_virtualenv(file.path(getwd(), ".venv"), required = TRUE)
} else if (dir.exists("../.venv")) {
  use_virtualenv(file.path(getwd(), "..", ".venv"), required = TRUE)
}

torch = import("torch")
nn = import("torch.nn")
F_ = import("torch.nn.functional")
transformers = import("transformers")
datasets_lib = import("datasets")

LLMWrapper = setRefClass("LLMWrapper",
  fields = list(
    backbone   = "ANY",
    heads      = "ANY",
    circuit    = "ANY",
    embed_dim  = "integer",
    vocab_size = "integer",
    cheat      = "logical"
  ),
  methods = list(
    initialize = function(model_id, head_type = NULL, window_size = 6L, ranks = 32L, lora_path = NULL, lora_r = 8L, lora_alpha = 16L, cheat = FALSE) {
      cheat <<- as.logical(cheat)
      # initializes the llm wrapper by loading the backbone model, configuring lora, and setting up the module dict
      # float32 on every device: the heads are float32, so a bf16 backbone would raise a dtype
      # mismatch in the head forward on CUDA. byT5-small is small enough that float32 is fine.
      best_dtype = torch$float32

      backbone <<- transformers$T5ForConditionalGeneration$from_pretrained(
        model_id,
        torch_dtype = best_dtype
      )

      if (!is.null(lora_path)) {
        peft = reticulate::import("peft")
        backbone <<- peft$PeftModel$from_pretrained(backbone, lora_path)
      } else {
        peft = reticulate::import("peft")
        lora_config = peft$LoraConfig(
          r = as.integer(lora_r), 
          lora_alpha = as.integer(lora_alpha), 
          bias = "none", 
          task_type = "SEQ_2_SEQ_LM"
        )
        backbone <<- peft$get_peft_model(backbone, lora_config)
      }

      heads <<- nn$ModuleDict()
      embed_dim <<- as.integer(backbone$config$d_model)
      vocab_size <<- as.integer(backbone$config$vocab_size)
      
      if (!is.null(head_type)) {
        swap_head(head_type, window_size, ranks)
      }
    },

    swap_head = function(head_type, window_size = 6L, ranks = 32L) {
      # swaps the active speculative prediction head and injects it into the backbone model
      heads <<- nn$ModuleDict()
      circuit_obj = create_circuit(head_type, window_size, ranks)
      circuit_obj$inject_head(.self)
      circuit <<- circuit_obj
      invisible(.self)
    },

    get_parameter_groups = function(head_lr, lora_lr) {
      # returns separated lists of model parameter groups for applying different learning rates
      head_params = reticulate::iterate(heads$parameters())
      lora_params = list()
      for (p in reticulate::iterate(backbone$parameters())) {
        if (p$requires_grad) {
          lora_params[[length(lora_params) + 1L]] = p
        }
      }
      list(
        list(params = head_params, lr = head_lr),
        list(params = lora_params, lr = lora_lr)
      )
    },

    to = function(device) {
      # moves the backbone and head modules to the target computing device
      backbone$to(device)
      heads$to(device)
      invisible(.self)
    },

    eval = function() {
      # sets the backbone and heads to evaluation mode
      backbone$eval()
      heads$eval()
      invisible(.self)
    },

    train = function(mode = TRUE) {
      # sets the backbone and heads to training mode
      backbone$train(mode)
      heads$train(mode)
      invisible(.self)
    },

    prepare_decoder_input_ids = function(input_ids) {
      # prepends the decoder start token to the input tokens and shifts them for training
      
      # input_ids shape: [batch_size, seq_len]
      device = input_ids$device
      dec_in = torch$cat(list(
        # [batch_size, 1]
        torch$full(list(as.integer(input_ids$size(0L)), 1L), 
                   as.integer(backbone$config$decoder_start_token_id), 
                   dtype = torch$long, device = device),
        # [batch_size, seq_len - 1]
        input_ids$narrow(1L, 0L, as.integer(input_ids$size(1L) - 1L))
      ), dim = 1L) # returns shape: [batch_size, seq_len]
      return(dec_in)
    },

    get_hidden_states = function(prompt_ids, decoder_ids, attention_mask = NULL, encoder_outputs = NULL, labels = NULL) {
      # performs a forward pass to extract the hidden states of the decoder last layer
      
      # prompt_ids shape: [batch_size, prompt_len]
      # decoder_ids shape: [batch_size, decoder_len]
      if (!is.null(encoder_outputs) && !inherits(encoder_outputs, "python.builtin.tuple")) {
        encoder_outputs = reticulate::tuple(encoder_outputs[[1]])
      }

      pad_token_id = as.integer(backbone$config$pad_token_id)
      
      if (!is.null(labels)) {
        if (cheat) {
          # Training (cheating): do not mask assistant tokens
          encoder_input_ids = prompt_ids
        } else {
          # Training (standard): mask assistant tokens in prompt_ids
          encoder_input_ids = prompt_ids$clone()
          encoder_input_ids[labels$ne(-100L)] = as.integer(backbone$config$decoder_start_token_id)
        }
      } else {
        P = as.integer(prompt_ids$size(1L))
        if (decoder_ids$size(1L) > P + 1L) {
          generated_tokens = decoder_ids$narrow(1L, P + 1L, as.integer(decoder_ids$size(1L) - (P + 1L)))
          if (cheat) {
            encoder_input_ids = torch$cat(list(prompt_ids, generated_tokens), dim = 1L)
          } else {
            encoder_input_ids = prompt_ids
          }
        } else {
          encoder_input_ids = prompt_ids
        }
      }

      attention_mask = encoder_input_ids$ne(pad_token_id)$to(torch$long)

      outputs = backbone(
        input_ids = if (is.null(encoder_outputs)) encoder_input_ids else NULL,
        attention_mask = attention_mask,
        decoder_input_ids = decoder_ids,
        encoder_outputs = encoder_outputs,
        use_cache = FALSE,
        output_hidden_states = TRUE
      )
      
      h_states = outputs$decoder_hidden_states
      
      if (!is.null(h_states)) {
        hidden_states = h_states[[length(h_states)]]
      } else {
        hidden_states = outputs$last_hidden_state
      } # hidden_states shape: [batch_size, decoder_len, embed_dim]

      return(list(
        x = hidden_states
      ))
    },

    verify_draft = function(draft_tokens, encoder_outputs, decoder_ids, prompt_ids, attention_mask = NULL) {
      # verifies the drafted tokens against the target verifier model and returns verification probability metrics
      N = length(draft_tokens)
      
      with(torch$no_grad(), {
        #concatenates prefix tokens with drafted ones
        draft_tensor = torch$tensor(matrix(draft_tokens, nrow = 1), dtype = torch$long, device = backbone$device)
        next_ids = torch$cat(list(decoder_ids, draft_tensor), dim = 1L)
        
        if (is.list(encoder_outputs)) {encoder_outputs = reticulate::tuple(encoder_outputs[[1]])} #correct format for hugging face
 

        # In cheating mode: we concatenate generated context up to this point to the encoder input
        P = as.integer(prompt_ids$size(1L))
        if (decoder_ids$size(1L) > P + 1L) {
          generated_tokens = decoder_ids$narrow(1L, P + 1L, as.integer(decoder_ids$size(1L) - (P + 1L)))
          if (cheat) {
            encoder_input_ids = torch$cat(list(prompt_ids, generated_tokens), dim = 1L)
          } else {
            encoder_input_ids = prompt_ids
          }
        } else {
          encoder_input_ids = prompt_ids
        }
        
        pad_token_id = as.integer(backbone$config$pad_token_id)
        attention_mask = encoder_input_ids$ne(pad_token_id)$to(torch$long)

        outputs = backbone(
          input_ids = if (is.null(encoder_outputs)) encoder_input_ids else NULL,
          attention_mask = attention_mask,
          decoder_input_ids = next_ids,
          encoder_outputs = encoder_outputs,
          use_cache = FALSE # here we do not use kv cache for simplification
        ) #here an additional logit is generated
        
        all_logits = outputs$logits
        L = as.integer(decoder_ids$size(1L))
        
        # take only the logits of the draft, cut the logits of decoder_ids
        relevant_logits = all_logits$narrow(1L, as.integer(L - 1L), as.integer(N + 1L)) # [batch_size, len(draft_tokens)+1, vocab_size]
        all_p_dist = F_$softmax(relevant_logits, dim = -1L)
        
        draft_p_dist = all_p_dist$narrow(1L, 0L, as.integer(N)) #discard the additional logit

        #take the probability of the drafted tokens
        p_vals = draft_p_dist$gather(dim = 2L, index = draft_tensor$unsqueeze(-1L))$squeeze(-1L)

        #take probability distribution of the bonus token
        next_p_dist = all_p_dist$select(1L, as.integer(N))
        
        list(
          p = p_vals$squeeze(0L)$cpu()$numpy(),
          next_p = next_p_dist,
          full_p_dist = all_p_dist
        )
      })
    },

    load_weights = function(weights_path, device = NULL, shift_offset_minus_1 = FALSE) {
      # loads pre-trained speculative head weights from a file path
      if (is.null(device)) {
        device = backbone$device
      }
      
      # Import our training module to reuse the Python load_head_weights logic
      reticulate::py_run_string("import sys")
      sys_path = reticulate::py_eval("sys.path")
      if (!("/Users/lorenzoallegrini/Documents/MTP/src" %in% sys_path)) {
        reticulate::py_run_string("sys.path.append('/Users/lorenzoallegrini/Documents/MTP/src')")
      }
      training_py = reticulate::import("training")
      
      target_window_size = as.integer(circuit$window_size)
      target_ranks = as.integer(circuit$ranks)
      
      training_py$load_head_weights(
        model_heads = heads,
        weights_path = weights_path,
        device = device,
        shift_offset_minus_1 = shift_offset_minus_1,
        target_window_size = target_window_size,
        ranks = target_ranks
      )
      invisible(.self)
    },

    save_weights = function(save_path) {
      # saves speculative head weights to a file path
      torch$save(heads$state_dict(), save_path)
      invisible(.self)
    }
  )
)

llm_get_hidden_states = function(model, prompt_ids, decoder_ids, attention_mask = NULL, encoder_outputs = NULL) {
  # extracts decoder hidden states using the provided wrapper model
  model$get_hidden_states(prompt_ids, decoder_ids, attention_mask, encoder_outputs)
}

llm_verify_draft = function(model, draft_tokens, encoder_outputs, decoder_ids, prompt_ids, attention_mask = NULL) {
  # runs draft verification on the provided wrapper model
  model$verify_draft(draft_tokens, encoder_outputs, decoder_ids, prompt_ids, attention_mask)
}