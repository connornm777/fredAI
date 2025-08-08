import math, torch, torch.nn as nn

class MiniTransformer(nn.Module):
    """
    ONE self-attention layer.  Symbols:

      • x ∈ ℝ^{B×T×d}   (B=batch, T=seq len, d=emb dim)
      • Q = x Wq , K = x Wk , V = x Wv
      • Self-attn  = softmax(QKᵀ / √d_k) V

    With d_k = d_head = d / n_heads.
    """
    def __init__(self, vocab_size, d_model=64, nhead=4):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)

        # Positional encoding: learnable (simplest possible)
        self.pos    = nn.Parameter(torch.zeros(1, 512, d_model))

        self.block  = nn.TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward= d_model,            # 1× width, keeps model tiny
            dropout=0.0,
            batch_first=True                     # (B, T, d)
        )
        self.proj   = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x : (B,T) token indices
        returns logits : (B,T,vocab)
        """
        h = self.embed(x) + self.pos[:, :x.size(1), :]
        h = self.block(h)
        return self.proj(h)
