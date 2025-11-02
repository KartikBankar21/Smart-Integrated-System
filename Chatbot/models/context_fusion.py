# context_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextFusion(nn.Module):
    """
    Multi-turn context fusion using per-dimension weighted attention.
    Learns to weight the last K turn vectors and produce a single context vector.
    """
    def __init__(self, hidden_dim, context_window=3):
        super(ContextFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        # Learned positional embeddings for each turn index
        self.turn_pos_emb = nn.Embedding(context_window, hidden_dim)
        # Learned weight vector for dimension-wise attention
        self.attn_weight = nn.Parameter(torch.ones(hidden_dim))
        
    def forward(self, turn_vecs):
        """
        Args:
            turn_vecs: Tensor(batch, K, hidden_dim) containing vectors for past turns.
        Returns:
            context_vec: Tensor(batch, hidden_dim) = fused context vector
        """
        batch, K, D = turn_vecs.size()
        # Add turn-position embeddings
        pos = torch.arange(K, device=turn_vecs.device)
        pos_emb = self.turn_pos_emb(pos)           # (K, D)
        pos_emb = pos_emb.unsqueeze(0).expand(batch, K, D)
        turn_vecs = turn_vecs + pos_emb
        
        # Compute attention scores and weights (per-dimension):contentReference[oaicite:10]{index=10}
        scores = turn_vecs * self.attn_weight    # (batch, K, D)
        attn = F.softmax(scores, dim=1)         # softmax over time dimension K
        # Weighted sum over time for each dimension
        context_vec = torch.sum(attn * turn_vecs, dim=1)  # (batch, D)
        return context_vec

