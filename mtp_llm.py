import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, T5ForConditionalGeneration

class MTP_LLM(nn.Module):
    def __init__(self, model_id, head_class, window_size=8):
        super().__init__()
        self.backbone = T5ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto"
        )
        
        config = self.backbone.config
        
        embed_dim = getattr(config, "d_model", getattr(config, "hidden_size", None))
        vocab_size = config.vocab_size
        
        if embed_dim is None:
            raise ValueError("Impossibile determinare la dimensione dell'embedding dal config.")
            
        self.mtp_head = head_class(
            embedding_size=embed_dim, 
            vocabulary_size=vocab_size, 
            window_size=window_size
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        Il passaggio in avanti completo: Input -> Backbone -> MTP Head
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            output_hidden_states=True
        ) 
        
        # Detachiamo gli hidden states per evitare di calcolare i gradienti per il backbone
        # ma permettiamo il calcolo dei gradienti per la mtp_head
        hidden_states = outputs.decoder_hidden_states[-1].detach()
        mtp_logits = self.mtp_head(hidden_states)
        
        return outputs.logits, mtp_logits