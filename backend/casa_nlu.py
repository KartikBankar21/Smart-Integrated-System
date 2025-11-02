# casa_nlu.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from disan import DiSAN
from context_fusion import ContextFusion

class CASA_NLU(nn.Module):
    """
    CASA-NLU model: joint Intent Classification and Slot Labeling with context.
    """
    def __init__(self, vocab_size, intent_size, slot_size, 
                 hidden_dim=56, embed_dim=56, context_window=3, sliding_window=3, dropout=0.3):
        super(CASA_NLU, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.sliding_window = sliding_window
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # DiSAN utterance encoder (signal encoding)
        self.disan = DiSAN(embed_dim, hidden_dim, dropout=dropout)
        # Context fusion over past turns
        self.context_fusion = ContextFusion(hidden_dim, context_window)
        
        # 🆕 Secondary IC: Direct from utterance encoding (no context)
        self.secondary_intent_fc = nn.Linear(hidden_dim, intent_size)
        
        # Intent classifier: FC(c_i) + concat with h_utt
        self.intent_fc1 = nn.Linear(hidden_dim, hidden_dim)
        # Final intent layer: (context+utterance) -> intent logits
        self.intent_fc2 = nn.Linear(2*hidden_dim, intent_size)
        
        # Slot history embedding (if provided as IDs)
        self.slot_emb = nn.Embedding(slot_size, hidden_dim)
        # Gating fusion for (token, utterance)
        self.fusion_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        
        # GRU for slot decoding
        # Input dim = hidden_dim * w (sliding window) + hidden_dim (slot_hist) + hidden_dim (IC penultimate)
        ic_penult_dim = hidden_dim
        slot_input_dim = hidden_dim * sliding_window + hidden_dim + ic_penult_dim
        self.slot_gru = nn.GRU(slot_input_dim, hidden_dim, batch_first=True)
        self.slot_classifier = nn.Linear(hidden_dim, slot_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, utterance, turn_history, intent_history=None, slot_history=None):
        """
        Args:
            utterance: LongTensor(batch, seq_len) – current tokens
            turn_history: Tensor(batch, context_window, hidden_dim) – past turn vectors
            intent_history: LongTensor(batch, context_window) – past intents (optional)
            slot_history: LongTensor(batch, seq_len) – past slot labels for this turn (optional)
        Returns:
            intent_logits: Tensor(batch, intent_size) - primary intent (with context)
            slot_logits: Tensor(batch, seq_len, slot_size)
            secondary_intent_logits: Tensor(batch, intent_size) - secondary intent (utterance-only)
        """
        # Encode current utterance
        x = self.embedding(utterance)           # (batch, seq_len, embed_dim)
        utt_vec, token_reps = self.disan(x)     # (batch, H), (batch, seq_len, H)
        
        # 🆕 Secondary IC loss: Direct from utterance encoding
        secondary_intent_logits = self.secondary_intent_fc(utt_vec)
        
        # Context fusion over turn history (last K turns, including current)
        context_vec = self.context_fusion(turn_history)  # (batch, H)
        # Intent: FC(context) + concat(utterance)
        h_int = torch.tanh(self.intent_fc1(context_vec))
        intent_input = torch.cat([h_int, utt_vec], dim=1)  # (batch, 2H)
        intent_logits = self.intent_fc2(intent_input)      # (batch, intent_size)
        
        # Slot labeling
        # Fuse each token with utterance via gated fusion
        seq_len = token_reps.size(1)
        utt_expanded = utt_vec.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, H)
        fusion_cat = torch.cat([token_reps, utt_expanded], dim=2)    # (batch, seq_len, 2H)
        gate = torch.sigmoid(self.fusion_gate(fusion_cat))
        fused_tokens = gate * token_reps + (1 - gate) * utt_expanded  # (batch, seq_len, H)
        
        # Apply sliding window of size w over tokens
        pad = (self.sliding_window - 1) // 2
        padded = F.pad(fused_tokens, (0, 0, pad, pad))  # pad tokens on seq dim
        sliding_feats = []
        for i in range(seq_len):
            win = padded[:, i:i+self.sliding_window, :].contiguous()  # (batch, w, H)
            sliding_feats.append(win.view(win.size(0), -1))         # flatten to (batch, w*H)
        sliding_feats = torch.stack(sliding_feats, dim=1)          # (batch, seq_len, w*H)
        
        # Append slot history and IC head features to each token
        if slot_history is not None:
            # Mean-pool slot history embeddings (shape: batch x H)
            slot_hist_embed = self.slot_emb(slot_history).mean(dim=1)
        else:
            slot_hist_embed = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        ic_feat = h_int  # (batch, H)
        
        # Concatenate for GRU input
        slot_input = torch.cat([
            sliding_feats,                                         # (batch, seq_len, w*H)
            slot_hist_embed.unsqueeze(1).expand(-1, seq_len, -1),  # (batch, seq_len, H)
            ic_feat.unsqueeze(1).expand(-1, seq_len, -1)           # (batch, seq_len, H)
        ], dim=2)  # (batch, seq_len, w*H + 2H)
        
        # Decode slot labels with a GRU
        gru_out, _ = self.slot_gru(slot_input)          # (batch, seq_len, H)
        slot_logits = self.slot_classifier(gru_out)     # (batch, seq_len, slot_size)
        
        return intent_logits, slot_logits, secondary_intent_logits