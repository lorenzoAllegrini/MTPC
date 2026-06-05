import sys
sys.modules['torchvision'] = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from peft import LoraConfig, get_peft_model

class MTP_LLM(nn.Module):
    """
    Container model matching R's LLMWrapper structure.
    
    Key attributes (matching R):
        - self.backbone: the LLM model (CausalLM or Seq2Seq)
        - self.heads: nn.ModuleDict containing the circuit's named layers
        - self.embed_dim: backbone embedding dimension
        - self.vocab_size: backbone vocabulary size
    """
    def __init__(self, model_id, head_class, window_size=8, ranks=32, lora_path=None, cheat=False):
        super().__init__()
        self.cheat = cheat
        
        # Use float32 on every device. The circuit heads are created in float32, so a bf16
        # backbone would raise "mat1 and mat2 must have the same dtype" in the head forward
        # on CUDA. byT5-small is small enough that float32 is fine and keeps the circuit's
        # log-space math numerically stable.
        best_dtype = torch.float32

        self.backbone = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=best_dtype,
            trust_remote_code=True
        )
            
        # Wrap backbone in LoRA or load existing adapter
        if lora_path:
            from peft import PeftModel
            print(f"Loading LoRA adapter from: {lora_path} (is_trainable=True)")
            self.backbone = PeftModel.from_pretrained(self.backbone, lora_path, is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            self.backbone = get_peft_model(self.backbone, lora_config)

        
        config = self.backbone.config
        
        self.embed_dim = getattr(config, "d_model", getattr(config, "hidden_size", None))
        self.vocab_size = config.vocab_size
        
        if self.embed_dim is None:
            raise ValueError("Impossibile determinare la dimensione dell'embedding dal config.")
        
        # Instantiate the circuit to get its layers
        circuit = head_class(
            embedding_size=self.embed_dim, 
            vocabulary_size=self.vocab_size, 
            window_size=window_size,
            **({'ranks': ranks} if 'ranks' in head_class.__init__.__code__.co_varnames else {})
        )
        
        # Extract named layers into nn.ModuleDict (matching R's LLMWrapper)
        if head_class.__name__ == 'MTPC_HMM':
            layers = {
                "sum_unit_omega_init": circuit.sum_unit_omega_init,
                "input_units_phi": circuit.input_units_phi,
                "sum_unit_omega_transitions": circuit.sum_unit_omega_transitions
            }
        elif head_class.__name__ == 'FF':
            # Maps the ModuleList of Python to the input_units_phi_1, input_units_phi_2... names in R
            layers = {f"input_units_phi_{i+1}": layer for i, layer in enumerate(circuit.input_units_phi)}
        elif head_class.__name__ == 'CanonicPolyidiac':
            layers = {
                "sum_unit_omega": circuit.sum_unit_omega,
                "input_units_phi": circuit.input_units_phi
            }
        else:
            layers = {}

        self.heads = nn.ModuleDict(layers)
        
        # Keep a reference to the circuit's forward for the forward pass
        self._circuit = circuit
        self._head_class_name = head_class.__name__
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Il passaggio in avanti completo: Input -> Backbone -> MTP Head
        """
        # Per i modelli Seq2Seq, l'encoder non deve vedere i token futuri 
        # che il decoder sta cercando di predire. Usiamo un segnaposto per evitare leak di cross-attention.
        decoder_start_token_id = self.backbone.config.decoder_start_token_id
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        if labels is not None:
            if self.cheat:
                # In cheating mode: we do not mask assistant tokens to let the encoder see them
                encoder_input_ids = input_ids
            else:
                encoder_input_ids = input_ids.clone()
                encoder_input_ids[labels != -100] = decoder_start_token_id
            
            # Construct decoder_input_ids to match standard Seq2Seq SFT: shift labels right and map -100 to pad_token_id (0)
            shifted_labels = labels.new_zeros(labels.shape)
            shifted_labels[..., 1:] = labels[..., :-1].clone()
            shifted_labels[..., 0] = decoder_start_token_id
            shifted_labels.masked_fill_(shifted_labels == -100, 0)
            decoder_input_ids = shifted_labels
        else:
            encoder_input_ids = input_ids
            # Inference fallback
            start_tokens = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device)
            decoder_input_ids = torch.cat([start_tokens, input_ids[:, :-1]], dim=1)

        outputs = self.backbone(
            input_ids=encoder_input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            output_hidden_states=True
        )
        hidden_states = outputs.decoder_hidden_states[-1]
        
        # Forward through the circuit (which uses self.heads layers via _circuit)
        mtp_logits = self._circuit(hidden_states)
        
        return outputs.logits, mtp_logits, hidden_states

    def get_hidden_states(self, prompt_ids, decoder_ids, attention_mask=None, encoder_outputs=None, labels=None):
        if encoder_outputs is not None and not isinstance(encoder_outputs, tuple):
            encoder_outputs = (encoder_outputs[0],) if isinstance(encoder_outputs, list) else encoder_outputs

        pad_token_id = self.backbone.config.pad_token_id

        if labels is not None:
            if self.cheat:
                # SFT training mode (cheating)
                encoder_input_ids = prompt_ids
            else:
                # SFT training mode (standard)
                decoder_start_token_id = self.backbone.config.decoder_start_token_id
                encoder_input_ids = prompt_ids.clone()
                encoder_input_ids[labels != -100] = decoder_start_token_id
        else:
            if self.cheat:
                # Inference mode (cheating): concatenate prompt_ids and generated tokens from decoder_ids
                if decoder_ids.size(1) > 1:
                    generated_tokens = decoder_ids[:, 1:]
                    encoder_input_ids = torch.cat([prompt_ids, generated_tokens], dim=1)
                else:
                    encoder_input_ids = prompt_ids
            else:
                # Inference mode (standard): only prompt_ids
                encoder_input_ids = prompt_ids

        attention_mask = encoder_input_ids.ne(pad_token_id).to(torch.long)

        outputs = self.backbone(
            input_ids=encoder_input_ids if encoder_outputs is None else None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_ids,
            encoder_outputs=encoder_outputs,
            use_cache=False,
            output_hidden_states=True
        )
        
        h_states = getattr(outputs, "decoder_hidden_states", None)
        if h_states is not None:
            hidden_states = h_states[-1]
        else:
            hidden_states = outputs.last_hidden_state

        return {"x": hidden_states}

    @torch.no_grad()
    def verify_draft(self, draft_tokens, encoder_outputs, decoder_ids, prompt_ids, attention_mask=None):
        N = len(draft_tokens)
        
        if isinstance(draft_tokens, torch.Tensor):
            draft_tensor = draft_tokens.view(1, -1).to(self.backbone.device)
        else:
            draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=self.backbone.device)
            
        next_ids = torch.cat([decoder_ids, draft_tensor], dim=1)
        
        # In cheating mode: concatenate generated context up to this point to the encoder input
        if self.cheat:
            if decoder_ids.size(1) > 1:
                generated_tokens = decoder_ids[:, 1:]
                encoder_input_ids = torch.cat([prompt_ids, generated_tokens], dim=1)
            else:
                encoder_input_ids = prompt_ids
        else:
            encoder_input_ids = prompt_ids
            
        pad_token_id = self.backbone.config.pad_token_id
        attention_mask = encoder_input_ids.ne(pad_token_id).to(torch.long)

        has_peft = hasattr(self.backbone, "disable_adapter")
        
        if encoder_outputs is not None and not isinstance(encoder_outputs, tuple):
            encoder_outputs = (encoder_outputs[0],) if isinstance(encoder_outputs, list) else encoder_outputs

        if has_peft:
            with self.backbone.disable_adapter():
                outputs = self.backbone(
                    input_ids=encoder_input_ids if encoder_outputs is None else None,
                    attention_mask=attention_mask,
                    decoder_input_ids=next_ids,
                    encoder_outputs=encoder_outputs,
                    use_cache=False
                )
        else:
            outputs = self.backbone(
                input_ids=encoder_input_ids if encoder_outputs is None else None,
                attention_mask=attention_mask,
                decoder_input_ids=next_ids,
                encoder_outputs=encoder_outputs,
                use_cache=False
            )
            
        all_logits = outputs.logits
        L = decoder_ids.size(1)
        
        relevant_logits = all_logits.narrow(1, L - 1, N + 1)
        all_p_dist = F.softmax(relevant_logits, dim=-1)
        
        draft_p_dist = all_p_dist.narrow(1, 0, N)
        p_vals = draft_p_dist.gather(dim=2, index=draft_tensor.unsqueeze(-1)).squeeze(-1)
        next_p_dist = all_p_dist.select(1, N)
        
        return {
            "p": p_vals.squeeze(0).cpu().numpy(),
            "next_p": next_p_dist,
            "full_p_dist": all_p_dist
        }

    def load_weights(self, weights_path, device=None):
        if device is None:
            device = self.backbone.device
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        self.heads.load_state_dict(state_dict)
        return self

    def save_weights(self, save_path):
        torch.save(self.heads.state_dict(), save_path)
        return self