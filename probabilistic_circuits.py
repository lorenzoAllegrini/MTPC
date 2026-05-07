import torch
import numpy as np
import torch.nn as nn

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
        
