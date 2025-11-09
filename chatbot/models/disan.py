# disan.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiSAN(nn.Module):
    """
    Directional Self-Attention Network (DiSAN) encoder for utterances.
    Applies forward/backward self-attention and a fusion gate to produce:
      - utt_vec: a context-aware utterance-level vector
      - token_reps: contextual token representations
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(DiSAN, self).__init__()
        self.hidden_dim = hidden_dim
        # Positional embeddings for absolute position encoding
        self.pos_emb = nn.Embedding(500, input_dim)  # max seq_len = 500
        # Linear projections for directional self-attention
        self.query_fw = nn.Linear(input_dim, hidden_dim)
        self.key_fw   = nn.Linear(input_dim, hidden_dim)
        self.value_fw = nn.Linear(input_dim, hidden_dim)
        self.query_bw = nn.Linear(input_dim, hidden_dim)
        self.key_bw   = nn.Linear(input_dim, hidden_dim)
        self.value_bw = nn.Linear(input_dim, hidden_dim)
        # Gate to fuse forward/backward outputs
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        # Utterance-level projection
        self.fc_utt = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Tensor(batch, seq_len, input_dim) = token embeddings (with no position yet)
        Returns:
            utt_vec: Tensor(batch, hidden_dim) = encoded utterance vector
            token_reps: Tensor(batch, seq_len, hidden_dim) = token-level representations
        """
        batch, seq_len, _ = x.size()
        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        x_pos = x + self.pos_emb(positions)
        
        # Forward-direction self-attention (causal mask)
        Q_fw = self.query_fw(x_pos)  # (batch, seq_len, H)
        K_fw = self.key_fw(x_pos)
        V_fw = self.value_fw(x_pos)
        mask_fw = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)  # (1, seq_len, seq_len)
        scores_fw = torch.bmm(Q_fw, K_fw.transpose(-2, -1)) / (self.hidden_dim**0.5)
        scores_fw = scores_fw.masked_fill(mask_fw == 0, float('-inf'))
        attn_fw = torch.bmm(F.softmax(scores_fw, dim=-1), V_fw)  # (batch, seq_len, H)
        
        # Backward-direction self-attention (reverse causal mask)
        Q_bw = self.query_bw(x_pos)
        K_bw = self.key_bw(x_pos)
        V_bw = self.value_bw(x_pos)
        mask_bw = torch.triu(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)
        scores_bw = torch.bmm(Q_bw, K_bw.transpose(-2, -1)) / (self.hidden_dim**0.5)
        scores_bw = scores_bw.masked_fill(mask_bw == 0, float('-inf'))
        attn_bw = torch.bmm(F.softmax(scores_bw, dim=-1), V_bw)  # (batch, seq_len, H)
        
        # Feature-wise fusion gate (Hochreiter & Schmidhuber, 1997) combining forward/backward:contentReference[oaicite:5]{index=5}
        fused = torch.cat([attn_fw, attn_bw], dim=2)  # (batch, seq_len, 2H)
        gate = torch.sigmoid(self.gate(fused))
        token_reps = gate * attn_fw + (1 - gate) * attn_bw  # (batch, seq_len, H)
        token_reps = self.dropout(token_reps)
        
        # Utterance-level vector: compress via mean pooling (sentence-level source2token):contentReference[oaicite:6]{index=6}
        utt_vec = torch.mean(token_reps, dim=1)       # (batch, H)
        utt_vec = torch.tanh(self.fc_utt(utt_vec))    # (batch, H)
        
        return utt_vec, token_reps

