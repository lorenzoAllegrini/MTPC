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


def build_btree_topology(window_size):
    """
    Builds the balanced binary-tree topology of latents over `window_size` token positions,
    recursively splitting a span at floor(len/2) down to the base case n=1 (a leaf token), as in
    MTPC-BTree (Fig. 2 of the paper). Returns:
      node_parent  : list[int]  -> for each internal (sum) node, the index of its parent node (-1 for root)
      token_parent : list[int]  -> for each token position, the index of the internal node that emits it
      n_internal   : int        -> number of internal latent/sum nodes (= window_size - 1 for window_size>=2)
    Internal nodes are numbered in pre-order (parent index < child index), so a single forward
    pass over nodes 0..n_internal-1 always processes a parent before its children.
    Non-root node k owns transition matrix index (k-1) -> q(z_k | z_{parent(k)}).
    """
    node_parent = []
    token_parent = [None] * window_size

    def build(toks, parent):
        idx = len(node_parent)
        node_parent.append(parent)
        h = len(toks) // 2
        for sub in (toks[:h], toks[h:]):
            if len(sub) == 1:
                token_parent[sub[0]] = idx          # leaf token emitted from this node's latent
            else:
                build(sub, idx)                     # internal child node
        return idx

    if window_size == 1:
        # degenerate: single token, single latent emitting it
        node_parent = [-1]
        token_parent = [0]
    else:
        build(list(range(window_size)), -1)
    return node_parent, token_parent, len(node_parent)


class BTree(nn.Module):
    """
    Binary-tree probabilistic circuit (MTPC-BTree): a hierarchy of latent variables over the MTP
    window. Mirrors MTPC_HMM's parameterisation (root prior + transitions + per-position emissions)
    but wires the latents as a balanced binary tree instead of a chain, so latents and tokens can be
    sampled with log(n) depth. The per-position marginal q(x_{t+i} | x<=t) is obtained by propagating
    the latent marginals from the root down to token i's parent latent (other branches marginalise to 1).
    """
    def __init__(self, embedding_size, vocabulary_size, window_size=8, ranks=32):
        super().__init__()
        self.window_size = window_size
        self.ranks = ranks
        self.vocabulary_size = vocabulary_size
        self.is_log_probs = True

        self.node_parent, self.token_parent, self.n_internal = build_btree_topology(window_size)
        self.n_trans = max(0, self.n_internal - 1)   # non-root internal nodes own one transition each

        # sum_unit_omega_init: root prior q(z_root); sum_unit_omega_transitions: tree edges q(z_child|z_parent);
        # input_units_phi: per-position emissions q(x_i | z_{parent(i)}) -> view [window, ranks, vocab] (HMM layout)
        self.sum_unit_omega_init = nn.Linear(embedding_size, ranks)
        self.sum_unit_omega_transitions = nn.Linear(embedding_size, max(1, self.n_trans) * ranks * ranks)
        self.input_units_phi = nn.Linear(embedding_size, window_size * ranks * vocabulary_size)

        # --- INITIALIZATION --- uniform mixtures at every sum node (gate weights = 0) so the tree
        # reduces to the FF marginal at init (see init_btree_from_ff for the emission transfer).
        with torch.no_grad():
            nn.init.zeros_(self.sum_unit_omega_init.weight)
            nn.init.zeros_(self.sum_unit_omega_init.bias)
            nn.init.zeros_(self.sum_unit_omega_transitions.weight)
            nn.init.zeros_(self.sum_unit_omega_transitions.bias)

    def _latent_log_marginals(self, log_init, log_trans):
        # log_init: [B,S,r]; log_trans: [B,S,n_trans,r_parent,r_child] (or None). Returns list of [B,S,r].
        log_p = [None] * self.n_internal
        log_p[0] = log_init
        for k in range(1, self.n_internal):
            parent = self.node_parent[k]
            # q(z_k) = sum_{z_parent} q(z_parent) q(z_k | z_parent)
            log_p[k] = stable_logsumexp(log_p[parent].unsqueeze(-1) + log_trans[:, :, k - 1, :, :], dim=-2)
        return log_p

    def forward(self, embeddings):
        batch_size, seq_len, _ = embeddings.shape
        log_init = F.log_softmax(self.sum_unit_omega_init(embeddings), dim=-1)  # [B,S,r]
        log_trans = None
        if self.n_trans > 0:
            flat_trans = self.sum_unit_omega_transitions(embeddings)
            log_trans = F.log_softmax(
                flat_trans[..., :self.n_trans * self.ranks * self.ranks].view(
                    batch_size, seq_len, self.n_trans, self.ranks, self.ranks), dim=-1)
        log_emiss = F.log_softmax(self.input_units_phi(embeddings).view(
            batch_size, seq_len, self.window_size, self.ranks, self.vocabulary_size), dim=-1)  # [B,S,W,r,V]

        log_p = self._latent_log_marginals(log_init, log_trans)
        log_marginal_probs = []
        for i in range(self.window_size):
            pl = self.token_parent[i]
            # q(x_i) = sum_z q(z_parent(i)=z) q(x_i | z)
            log_marginal_probs.append(
                stable_logsumexp(log_p[pl].unsqueeze(-1) + log_emiss[:, :, i, :, :], dim=-2))  # [B,S,V]
        return torch.stack(log_marginal_probs, dim=2)  # [B,S,W,V]

    @torch.no_grad()
    def generate_draft(self, embeddings):
        last_emb = embeddings[:, -1:, :]
        batch_size = last_emb.shape[0]
        bidx = torch.arange(batch_size, device=embeddings.device)

        init_probs = F.softmax(self.sum_unit_omega_init(last_emb).squeeze(1), dim=-1)  # [B,r]
        trans_probs = None
        if self.n_trans > 0:
            trans_probs = F.softmax(self.sum_unit_omega_transitions(last_emb).view(
                batch_size, self.n_trans, self.ranks, self.ranks), dim=-1)  # [B,n_trans,r,r]
        emiss_probs = F.softmax(self.input_units_phi(last_emb).view(
            batch_size, self.window_size, self.ranks, self.vocabulary_size), dim=-1)  # [B,W,r,V]

        # ancestral sampling of the latent tree (CPU fallback for MPS Categorical bug)
        z = [None] * self.n_internal
        z[0] = Categorical(probs=init_probs.to("cpu")).sample().to(embeddings.device)
        for k in range(1, self.n_internal):
            parent = self.node_parent[k]
            curr = trans_probs[bidx, k - 1, z[parent], :]  # [B,r]
            z[k] = Categorical(probs=curr.to("cpu")).sample().to(embeddings.device)

        draft_tokens = []
        for i in range(self.window_size):
            pl = self.token_parent[i]
            curr_emiss = emiss_probs[bidx, i, z[pl], :]  # [B,V]
            draft_tokens.append(Categorical(probs=curr_emiss.to("cpu")).sample().to(embeddings.device))
        return torch.stack(draft_tokens, dim=1)  # [B,W]