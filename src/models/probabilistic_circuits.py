import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class FF(nn.Module):
    def __init__(self, embedding_size, vocabulary_size, window_size):
        super().__init__()
        self.window_size = window_size
        self.emission_projs = nn.ModuleList([
            nn.Linear(embedding_size, vocabulary_size) for _ in range(window_size)
        ])
    
    def forward(self, embeddings):
        return torch.stack([
            proj(embeddings) for proj in self.emission_projs
        ], dim=2)


class CanonicPolyidiac(nn.Module):
    def __init__(self, embedding_size, vocabulary_size, window_size, ranks=32):
        super().__init__()
        self.window_size = window_size
        self.ranks = ranks
        self.vocabulary_size = vocabulary_size
        self.is_log_probs = True

        
        self.gate = nn.Linear(embedding_size, ranks)
        self.emission_projs = nn.Linear(embedding_size, ranks * window_size * vocabulary_size)

        # --- INIZIALIZZAZIONE ---
        with torch.no_grad():
            # Inizializzazione Gate (Uniforme: Pesi = 0, Bias = 0)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
    
    def forward(self, embeddings):
        batch_size, seq_len, _ = embeddings.shape
        
        gate_logits = self.gate(embeddings) # [batch, seq_len, ranks]
        log_weights = F.log_softmax(gate_logits, dim=-1)

        flat_emissions = self.emission_projs(embeddings)
        
        emissions = flat_emissions.view(
            batch_size, seq_len, self.ranks, self.window_size, self.vocabulary_size
        )
        log_token_probs = F.log_softmax(emissions, dim=-1)
        
        log_weights_expanded = log_weights.unsqueeze(3).unsqueeze(4)

        log_marginal_probs = torch.logsumexp(log_weights_expanded + log_token_probs, dim=2)
        
        return log_marginal_probs

    @torch.no_grad() # Cruciale: spegne i gradienti per velocizzare
    def generate_draft(self, embeddings):

        last_emb = embeddings[:, -1:, :]
        batch_size = last_emb.shape[0]
        
        gate_logits = self.gate(last_emb) # [batch, 1, ranks]
        gate_probs = F.softmax(gate_logits, dim=-1).squeeze(1) # [batch, ranks]
        
        selected_ranks = Categorical(probs=gate_probs).sample() # [batch]
        
        flat_emissions = self.emission_projs(last_emb)
        emissions = flat_emissions.view(
            batch_size, 1, self.ranks, self.window_size, self.vocabulary_size
        )
        
        batch_indices = torch.arange(batch_size, device=embeddings.device)
        selected_logits = emissions[batch_indices, 0, selected_ranks, :, :]
        
        token_probs = F.softmax(selected_logits, dim=-1)
        draft_tokens = Categorical(probs=token_probs).sample() # [batch, window_size]
        
        return draft_tokens


class MTPC_HMM(nn.Module):
    def __init__(self, embedding_size, vocabulary_size, emission_matrix=None, transition_matrix=None, window_size=16, ranks=32):
        """
        Testa HMM Contestuale e Inomogenea con Identity Initialisation e Init Uniforme.
        Valori paper: window_size=16, ranks=32.
        """
        super().__init__()
        self.window_size = window_size
        self.ranks = ranks
        self.vocabulary_size = vocabulary_size
        
        self.register_buffer("step_counter", torch.tensor(0))
        
        # --- CREAZIONE DEI LAYER ---
        self.init_gate = nn.Linear(embedding_size, ranks)
        self.emissions = nn.Linear(embedding_size, window_size * ranks * vocabulary_size)
        self.transitions = nn.Linear(embedding_size, (window_size - 1) * ranks * ranks)

        # --- INIZIALIZZAZIONE ---
        with torch.no_grad():
            
            # 1. Inizializzazione Init Gate (Uniforme: Pesi = 0, Bias = 0)
            nn.init.zeros_(self.init_gate.weight)
            nn.init.zeros_(self.init_gate.bias)

            # 2. Inizializzazione Emissioni
            if emission_matrix is not None:
                e_tensor = torch.tensor(emission_matrix, dtype=torch.float)
                self.emissions.weight.copy_(e_tensor)
                nn.init.zeros_(self.emissions.bias)
                
            # 3. Inizializzazione Transizioni
            if transition_matrix is not None:
                t_tensor = torch.tensor(transition_matrix, dtype=torch.float)
                self.transitions.weight.copy_(t_tensor)
                nn.init.zeros_(self.transitions.bias)
                self._apply_identity_init(override_weights=False)
            else:
                self._apply_identity_init(override_weights=True)

    def _apply_identity_init(self, override_weights=True):
        if override_weights:
            nn.init.normal_(self.transitions.weight, mean=0.0, std=1e-4)
        
        bias_tensor = torch.zeros(self.window_size - 1, self.ranks, self.ranks)
        
        # Logit alto per forzare l'identità dopo il Softmax
        bias_tensor.diagonal(dim1=-2, dim2=-1).fill_(5.0)
        
        with torch.no_grad():
            self.transitions.bias.copy_(bias_tensor.flatten())
            

    def forward(self, embeddings):

        batch_size, seq_len, _ = embeddings.shape
        
        log_alpha = F.log_softmax(self.init_gate(embeddings), dim=-1) # [B, S, R]
        
        flat_emiss = self.emissions(embeddings)
        log_emiss = F.log_softmax(flat_emiss.view(
            batch_size, seq_len, self.window_size, self.ranks, self.vocabulary_size
        ), dim=-1)
        
        flat_trans = self.transitions(embeddings)
        log_trans = F.log_softmax(flat_trans.view(
            batch_size, seq_len, self.window_size - 1, self.ranks, self.ranks
        ), dim=-1) # dim -2: R_prev, dim -1: R_next
        
        if self.training:
            self.step_counter += 1
            if self.step_counter % 50 == 0:
                with torch.no_grad():
                    trans_probs = torch.exp(log_trans)
                    diags = torch.diagonal(trans_probs, dim1=-2, dim2=-1)
                    avg_diag = diags.mean().item()
                    off_diag_mask = ~torch.eye(self.ranks, device=trans_probs.device).bool()
                    avg_off_diag = trans_probs[:, :, :, off_diag_mask].mean().item()
                    batch_std = trans_probs.std(dim=0).mean().item()
                    print(f"\n[DEBUG TRAINING HMM] Step {self.step_counter.item()} | Avg Diag: {avg_diag:.4f} | Avg Off-Diag: {avg_off_diag:.4f} | Context Std: {batch_std:.6f}")
        
        log_marginal_probs = []

        for t in range(self.window_size):
            curr_emission = log_emiss[:, :, t, :, :] # [B, S, R, Vocab]
            step_prob = torch.logsumexp(log_alpha.unsqueeze(-1) + curr_emission, dim=2)
            log_marginal_probs.append(step_prob)

            if t < self.window_size - 1:
                curr_trans = log_trans[:, :, t, :, :] 
                log_alpha = torch.logsumexp(log_alpha.unsqueeze(-1) + curr_trans, dim=2)

        return torch.stack(log_marginal_probs, dim=2) # [B, S, W, Vocab]

    @torch.no_grad()
    def get_draft_probabilities(self, embeddings):
        last_emb = embeddings[:, -1:, :]
        batch_size = last_emb.shape[0]
        
        init_probs = F.softmax(self.init_gate(last_emb).squeeze(1), dim=-1)
        
        flat_emiss = self.emissions(last_emb)
        emiss_probs = F.softmax(flat_emiss.view(
            batch_size, self.window_size, self.ranks, self.vocabulary_size
        ), dim=-1)
        
        flat_trans = self.transitions(last_emb)
        trans_probs = F.softmax(flat_trans.view(
            batch_size, self.window_size - 1, self.ranks, self.ranks
        ), dim=-1)

        return {
            "init": init_probs.cpu().numpy(),
            "emiss": emiss_probs.cpu().numpy(),
            "trans": trans_probs.cpu().numpy()
        }

    @torch.no_grad()
    def generate_draft(self, embeddings):
        last_emb = embeddings[:, -1:, :] # Prendi solo l'ultimo token
        batch_size = last_emb.shape[0]
        batch_indices = torch.arange(batch_size, device=embeddings.device)
        

        init_probs = F.softmax(self.init_gate(last_emb).squeeze(1), dim=-1)
        
        flat_emiss = self.emissions(last_emb)
        emiss_probs = F.softmax(flat_emiss.view(
            batch_size, self.window_size, self.ranks, self.vocabulary_size
        ), dim=-1)
        
        flat_trans = self.transitions(last_emb)
        trans_probs = F.softmax(flat_trans.view(
            batch_size, self.window_size - 1, self.ranks, self.ranks
        ), dim=-1)
        
        draft_tokens = []
        
        z_t = Categorical(probs=init_probs).sample()
        
        for t in range(self.window_size):
            # Step 2: Estrazione del token x_t condizionato a z_t
            curr_emiss = emiss_probs[batch_indices, t, z_t, :]
            x_t = Categorical(probs=curr_emiss).sample()
            draft_tokens.append(x_t)
            
            # Step 3: Transizione allo stato z_{t+1} (catena di Markov)
            if t < self.window_size - 1:
                curr_trans = trans_probs[batch_indices, t, z_t, :]
                z_t = Categorical(probs=curr_trans).sample()
                
        return torch.stack(draft_tokens, dim=1) # [B, W]