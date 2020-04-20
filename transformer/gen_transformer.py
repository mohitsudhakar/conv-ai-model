import torch
from torch import nn
import torch.nn.functional as F

from transformer.transformer_block import TransformerBlock


class GenTransformer(nn.Module):
    """
    Generate text (character by character)
    """

    def __init__(self, emb, heads, num_tokens, seq_len, depth, ):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_embeddings=num_tokens, embedding_dim=emb)
        self.pos_emb = nn.Embedding(num_embeddings=seq_len, embedding_dim=emb)

        trans_blocks = []
        for i in range(depth):
            trans_blocks.append(TransformerBlock(emb, heads))
