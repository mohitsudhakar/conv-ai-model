import torch
from torch import nn
import torch.nn.functional as F
from transformer.transformer_block import TransformerBlock


class TCTransformer(nn.Module):
    def __init__(self, emb_dim, heads, depth, seq_length, num_tokens, num_classes, dropout=0.0):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_embeddings=num_tokens, embedding_dim=emb_dim)
        self.pos_emb = nn.Embedding(num_embeddings=seq_length, embedding_dim=emb_dim)  # position_encoding can be used instead of embedding
        self.dropout = nn.Dropout(dropout)

        # Sequence of transformer blocks that does the heavy lifting
        trans_blocks = []
        for i in range(depth):
            trans_blocks.append(TransformerBlock(emb_dim=emb_dim, heads=heads))
        self.trans_blocks = nn.Sequential(*trans_blocks)

        # Maps final output sequence to class logits
        self.to_probs = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        # generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        # generate position embeddings
        positions = torch.arange(t, device=device)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.dropout(x)
        x = self.trans_blocks(x)

        # Avg pool over the t dimension and project to class probs
        x = self.to_probs(x.mean(dim=1))
        out = F.log_softmax(x, dim=1)
        return out
