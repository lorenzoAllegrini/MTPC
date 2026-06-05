import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

def stable_logsumexp(x, dim, keepdim=False):
    """
    A stable manual implementation of logsumexp to prevent MPS backend compiler deadlocks
    on Apple Silicon during the backward pass.
    """
    max_val = x.max(dim=dim, keepdim=True).values
    result = max_val + torch.log(torch.exp(x - max_val).sum(dim=dim, keepdim=True))
    if not keepdim:
        result = result.squeeze(dim)
    return result

class FF(nn.Module):
    def __init__(self, embedding_size, vocabulary_size, window_size):
        super().__init__()
        self.window_size = window_size
        # input_units_phi parameterizes the categorical distributions over individual tokens.
        self.input_units_phi = nn.ModuleList([
            nn.Linear(embedding_size, vocabulary_size) for _ in range(window_size)
        ])
    
    def forward(self, embeddings):
        return torch.stack([
            proj(embeddings) for proj in self.input_units_phi
        ], dim=2)


class CanonicPolyidiac(nn.Module):
    def __init__(self, embedding_size, vocabulary_size, window_size, ranks=32):
        super().__init__()
        self.window_size = window_size
        self.ranks = ranks
        self.vocabulary_size = vocabulary_size
        self.is_log_probs = True

        # sum_unit_omega parameterizes the root sum unit weights omega_j = q(z_j | x_{\le t})
        self.sum_unit_omega = nn.Linear(embedding_size, ranks)
        # input_units_phi parameterizes the r input units per token (phi emissions)
        self.input_units_phi = nn.Linear(embedding_size, ranks * window_size * vocabulary_size)

        # --- INITIALIZATION ---
        with torch.no_grad():
            # Initialization (Uniform: Weights = 0, Bias = 0)
            nn.init.zeros_(self.sum_unit_omega.weight)
            nn.init.zeros_(self.sum_unit_omega.bias)
    
    def forward(self, embeddings):
        batch_size, seq_len, _ = embeddings.shape
        
        gate_logits = self.sum_unit_omega(embeddings) # [batch, seq_len, ranks]
        log_weights = F.log_softmax(gate_logits, dim=-1)

        flat_input_units = self.input_units_phi(embeddings)
        
        input_units = flat_input_units.view(
            batch_size, seq_len, self.ranks, self.window_size, self.vocabulary_size
        )
        log_token_probs = F.log_softmax(input_units, dim=-1)
        
        # Explicit Sum Unit over Ranks to compute the marginal distribution for each step in the window:
        # P(x_{t+j} | x_{\le t}) = \sum_r q(z = r | x_{\le t}) * P(x_{t+j} | z = r)
        # In log-space: log_marginal_probs = logsumexp(log_weights + log_token_probs, dim=2)
        log_weights_expanded = log_weights.unsqueeze(3).unsqueeze(4) # [batch_size, seq_len, ranks, 1, 1]
        log_marginal_probs = stable_logsumexp(log_weights_expanded + log_token_probs, dim=2) # [batch_size, seq_len, window_size, vocab_size]
        
        return log_marginal_probs

    @torch.no_grad()
    def generate_draft(self, embeddings):
        last_emb = embeddings[:, -1:, :]
        batch_size = last_emb.shape[0]
        
        gate_logits = self.sum_unit_omega(last_emb) # [batch, 1, ranks]
        gate_probs = F.softmax(gate_logits, dim=-1).squeeze(1) # [batch, ranks]
        
        # CPU Fallback for Categorical sampling due to PyTorch MPS backend bug
        selected_ranks = Categorical(probs=gate_probs.to("cpu")).sample().to(embeddings.device) # [batch]
        
        flat_input_units = self.input_units_phi(last_emb)
        input_units = flat_input_units.view(
            batch_size, 1, self.ranks, self.window_size, self.vocabulary_size
        )
        
        batch_indices = torch.arange(batch_size, device=embeddings.device)
        selected_logits = input_units[batch_indices, 0, selected_ranks, :, :]
        
        token_probs = F.softmax(selected_logits, dim=-1)
        draft_tokens = Categorical(probs=token_probs.to("cpu")).sample().to(embeddings.device) # [batch, window_size]
        
        return draft_tokens


class MTPC_HMM(nn.Module):
    def __init__(self, embedding_size, vocabulary_size, emission_matrix=None, transition_matrix=None, window_size=16, ranks=32):
        """
        Contextual Inhomogeneous HMM Speculative Head aligning with paper nomenclature.
        """
        super().__init__()
        self.window_size = window_size
        self.ranks = ranks
        self.vocabulary_size = vocabulary_size
        
        self.register_buffer("step_counter", torch.tensor(0))
        
        # --- LAYER CREATION ---
        # sum_unit_omega_init parameterizes the root sum unit for the first latent state z_1
        self.sum_unit_omega_init = nn.Linear(embedding_size, ranks)
        # input_units_phi parameterizes the input unit distributions q_phi(x_{t+i} | z_i, x_{\le t})
        self.input_units_phi = nn.Linear(embedding_size, window_size * ranks * vocabulary_size)
        # sum_unit_omega_transitions parameterizes transition probabilities q(z_i | z_{i-1}, x_{\le t})
        self.sum_unit_omega_transitions = nn.Linear(embedding_size, (window_size - 1) * ranks * ranks)

        # --- INITIALIZATION ---
        with torch.no_grad():
            # 1. Initialization (Uniform: Weights = 0, Bias = 0)
            nn.init.zeros_(self.sum_unit_omega_init.weight)
            nn.init.zeros_(self.sum_unit_omega_init.bias)

            # 2. Emission matrix initialization (from backbone)
            if emission_matrix is not None:
                e_tensor = torch.tensor(emission_matrix, dtype=torch.float)
                self.input_units_phi.weight.copy_(e_tensor)
                nn.init.zeros_(self.input_units_phi.bias)
                
            # 3. Transition matrix initialization
            if transition_matrix is not None:
                t_tensor = torch.tensor(transition_matrix, dtype=torch.float)
                self.sum_unit_omega_transitions.weight.copy_(t_tensor)
                nn.init.zeros_(self.sum_unit_omega_transitions.bias)
                self._apply_identity_init(override_weights=False)
            else:
                self._apply_identity_init(override_weights=True)

    def _apply_identity_init(self, override_weights=True):
        if override_weights:
            nn.init.normal_(self.sum_unit_omega_transitions.weight, mean=0.0, std=1e-4)
        
        bias_tensor = torch.zeros(self.window_size - 1, self.ranks, self.ranks)
        
        # Logit to force identity transitions after Softmax
        bias_tensor.diagonal(dim1=-2, dim2=-1).fill_(5.0)
        
        with torch.no_grad():
            self.sum_unit_omega_transitions.bias.copy_(bias_tensor.flatten())

    def forward(self, embeddings):
        batch_size, seq_len, _ = embeddings.shape
        
        log_alpha = F.log_softmax(self.sum_unit_omega_init(embeddings), dim=-1) # [B, S, R]
        
        flat_input_units = self.input_units_phi(embeddings)
        log_emiss = F.log_softmax(flat_input_units.view(
            batch_size, seq_len, self.window_size, self.ranks, self.vocabulary_size
        ), dim=-1)
        
        flat_trans = self.sum_unit_omega_transitions(embeddings)
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
            step_prob = stable_logsumexp(log_alpha.unsqueeze(-1) + curr_emission, dim=2)
            log_marginal_probs.append(step_prob)

            if t < self.window_size - 1:
                curr_trans = log_trans[:, :, t, :, :] 
                log_alpha = stable_logsumexp(log_alpha.unsqueeze(-1) + curr_trans, dim=2)

        return torch.stack(log_marginal_probs, dim=2) # [B, S, W, Vocab]

    @torch.no_grad()
    def get_draft_probabilities(self, embeddings):
        last_emb = embeddings[:, -1:, :]
        batch_size = last_emb.shape[0]
        
        init_probs = F.softmax(self.sum_unit_omega_init(last_emb).squeeze(1), dim=-1)
        
        flat_input_units = self.input_units_phi(last_emb)
        emiss_probs = F.softmax(flat_input_units.view(
            batch_size, self.window_size, self.ranks, self.vocabulary_size
        ), dim=-1)
        
        flat_trans = self.sum_unit_omega_transitions(last_emb)
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
        last_emb = embeddings[:, -1:, :]
        batch_size = last_emb.shape[0]
        batch_indices = torch.arange(batch_size, device=embeddings.device)
        
        init_probs = F.softmax(self.sum_unit_omega_init(last_emb).squeeze(1), dim=-1)
        
        flat_input_units = self.input_units_phi(last_emb)
        emiss_probs = F.softmax(flat_input_units.view(
            batch_size, self.window_size, self.ranks, self.vocabulary_size
        ), dim=-1)
        
        flat_trans = self.sum_unit_omega_transitions(last_emb)
        trans_probs = F.softmax(flat_trans.view(
            batch_size, self.window_size - 1, self.ranks, self.ranks
        ), dim=-1)
        
        draft_tokens = []
        
        # CPU Fallback for Categorical sampling due to PyTorch MPS backend bug
        z_t = Categorical(probs=init_probs.to("cpu")).sample().to(embeddings.device)
        
        for t in range(self.window_size):
            curr_emiss = emiss_probs[batch_indices, t, z_t, :]
            x_t = Categorical(probs=curr_emiss.to("cpu")).sample().to(embeddings.device)
            draft_tokens.append(x_t)
            
            if t < self.window_size - 1:
                curr_trans = trans_probs[batch_indices, t, z_t, :]
                z_t = Categorical(probs=curr_trans.to("cpu")).sample().to(embeddings.device)
                
        return torch.stack(draft_tokens, dim=1) # [B, W]