import sys
sys.modules['torchvision'] = None

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from peft import LoraConfig, get_peft_model

class MTP_LLM(nn.Module):
    """
    Container model matching R's SpeculativeEngine structure.
    
    Key attributes (matching R):
        - self.backbone: the LLM model (CausalLM or Seq2Seq)
        - self.heads: nn.ModuleDict containing the circuit's named layers
        - self.embed_dim: backbone embedding dimension
        - self.vocab_size: backbone vocabulary size
    """
    def __init__(self, model_id, head_class, window_size=8, ranks=32, lora_path=None):
        super().__init__()
        
        if torch.cuda.is_available():
            best_dtype = torch.bfloat16
        else:
            best_dtype = torch.float32

        self.backbone = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=best_dtype,
            trust_remote_code=True
        )
            
        # Wrap backbone in LoRA or load existing adapter
        if lora_path:
            from peft import PeftModel
            print(f"Loading LoRA adapter from: {lora_path}")
            self.backbone = PeftModel.from_pretrained(self.backbone, lora_path)
            # Ensure it's in training mode if we wanted to train, but here we are in inference
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
        
        # Extract named layers into nn.ModuleDict (matching R's SpeculativeEngine)
        if head_class.__name__ == 'MTPC_HMM':
            layers = {
                "init_gate": circuit.init_gate,
                "emissions": circuit.emissions,
                "transitions": circuit.transitions
            }
        elif head_class.__name__ == 'FF':
            # Mappa il ModuleList di Python ai nomi emission_1, emission_2... di R
            layers = {f"emission_{i+1}": layer for i, layer in enumerate(circuit.emission_projs)}
        elif head_class.__name__ == 'CanonicPolyidiac':
            layers = {
                "gate": circuit.gate,
                "emission_projs": circuit.emission_projs
            }
        else:
            layers = {}

        self.heads = nn.ModuleDict(layers)
        
        # Keep a reference to the circuit's forward for the forward pass
        self._circuit = circuit
        self._head_class_name = head_class.__name__
        
    def forward(self, input_ids, attention_mask=None):
        """
        Il passaggio in avanti completo: Input -> Backbone -> MTP Head
        """
        # Per i modelli Seq2Seq, l'encoder non deve vedere i token futuri 
        # che il decoder sta cercando di predire. Durante il training MTP su intera sequenza,
        # usiamo un segnaposto per evitare leak di cross-attention.
        
        # Prepariamo i decoder_input_ids con il pad token iniziale (standard T5)
        decoder_start_token_id = self.backbone.config.decoder_start_token_id
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Creiamo un decoder_input_ids che inizia con il token di start
        start_tokens = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device)
        # Concateniamo con input_ids (senza l'ultimo token per mantenere la lunghezza)
        decoder_input_ids = torch.cat([start_tokens, input_ids[:, :-1]], dim=1)

        outputs = self.backbone(
            input_ids=input_ids, # L'encoder vede il contesto
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True
        )
        hidden_states = outputs.decoder_hidden_states[-1]
        
        # Forward through the circuit (which uses self.heads layers via _circuit)
        mtp_logits = self._circuit(hidden_states)
        
        return outputs.logits, mtp_logits, hidden_states