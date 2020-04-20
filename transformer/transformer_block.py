from torch import nn

from transformer.self_attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads):
        super().__init__()

        self.attention = SelfAttention(emb_dim, heads=heads)

        # The layer normalization is applied over the embedding dimension only.
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        ff = self.ff(x)
        res = self.norm2(ff + x)
        return res
