"""Transformer Module."""

import torch
from torch import nn

class GPT2LMHeadModel(nn.Module):
    """Language Model based on GPT2 transformer model."""

    def __init__(
        self,
        vocab_size,
        hidden_dim,
        device,
        embedding_dim=768,
        num_heads = 12,
        num_layers = 12,
        seq_len=1024,
        dropout=0.1,
    ):
        """
        Initialize the GPT2 transformer.
        Parameters
        ----------
        vocab_size : int
            Vocabulary size
        embedding_dim : int
            Embedding Dimension
        num_heads : int
            Number of Heads
        num_layers : int
            Number of Layers
        seq_len : int
            Sequence Length
        hidden_dim : int
            Hidden Layer Dimension
        device : string
            Device Type
        dropout : float
            Dropout of Network
        """
        super().__init__()
        self.device = device

        # Create Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(seq_len, embedding_dim)

        # Create subsequent layers
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                Transformer(
                    embedding_dim,
                    num_heads,
                    hidden_dim=hidden_dim,
                    mask=True,
                    dropout=dropout,
                )
            )
            self.transformer_blocks.append(nn.BatchNorm1d(seq_len))
        
        self.linear = nn.Linear(embedding_dim, vocab_size)
        # Initialize Weights and Biases
        nn.init.normal_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.normal_(self.linear.bias)

    def forward(self, x):
        """
        Apply GPT2 transformer.
        Parameters
        ----------
        x : torch.tensor
            batch x seq_len tensor of token indices
        Returns
        -------
        out : torch.tensor
            batch_size x num_classes
        """
        token_embed = self.token_embedding(x)

        # Batch size, seq_len, embedding_size
        batch_size, seq_len, embedding_dim = token_embed.size()

        position_embed = torch.arange(seq_len, device=self.device)
        position_embed = self.positional_embedding(position_embed)
        position_embed = position_embed[None, :, :]
        position_embed = position_embed.expand(batch_size, seq_len, embedding_dim)

        # Create the embedding
        x = token_embed + position_embed
        x = self.dropout(x)

        # Pass through the transformer blocks
        for _, layer in enumerate(self.transformer_blocks):
            x = layer(x)

        # Average over the sequence length
        out = x.mean(dim=1)

        # Lower dimension to num classes
        out = self.linear(out)

        # Convert to log-probabilities
        out = nn.functional.log_softmax(out, dim=1)
        return out


class ClfTransformer(nn.Module):
    """Classify Sequences based on a Transformer."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads,
        num_layers,
        seq_len,
        num_classes,
        hidden_dim,
        device,
        dropout=0,
    ):
        """
        Initialize the Classification Transformer.
        Parameters
        ----------
        vocab_size : int
            Vocabulary size
        embedding_dim : int
            Embedding Dimension
        num_heads : int
            Number of Heads
        num_layers : int
            Number of Layers
        seq_len : int
            Sequence Length
        num_classes : int
            Number of Classes
        hidden_dim : int
            Hidden Layer Dimension
        device : string
            Device Type
        dropout : float
            Dropout of Network
        """
        super().__init__()
        self.device = device

        # Create Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEncoding(seq_len, embedding_dim)

        # Create subsequent layers
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                Transformer(
                    embedding_dim,
                    num_heads,
                    hidden_dim=hidden_dim,
                    mask=False,
                    dropout=dropout,
                )
            )
            self.transformer_blocks.append(nn.BatchNorm1d(seq_len))
        self.linear = nn.Linear(embedding_dim, num_classes)

        # Initialize Weights and Biases
        nn.init.normal_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.normal_(self.linear.bias)

    def forward(self, x):
        """
        Apply classification transformation.
        Parameters
        ----------
        x : torch.tensor
            batch x seq_len tensor of token indices
        Returns
        -------
        out : torch.tensor
            batch_size x num_classes
        """

        token_embed = self.token_embedding(x)  # batch_size x seq_len x embedding_dim
        x = self.positional_embedding(token_embed) # Batch size, seq_len, embedding_size
        x = self.dropout(x)  # batch_size x seq_len x embedding_dim

        # Pass through the transformer blocks
        for _, layer in enumerate(self.transformer_blocks):
            x = layer(x)

        # Average over the sequence length
        out = x.mean(dim=1)

        # Lower dimension to num classes
        out = self.linear(out)

        # Convert to log-probabilities
        out = nn.functional.log_softmax(out, dim=1)
        return out

class PositionalEncoding(torch.nn.Module):
    """The Postional Encoding block."""

    def __init__(self, max_seq_len, embedding_dim):
        """
        Initialize the Postional Encoding block.
        Parameters
        ----------
        max_seq_len : int
            Maximum Sequence Length
        embedding_dim : int
            Embedding Dimension
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Creating zeros array of size max_seq_len x embedding_dim
        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embedding_dim)))
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / embedding_dim))
                )
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Apply Postional Encoding.
        Parameters
        ----------
        x : torch.tensor
            batch_size x seq_len x embedding_size
        Returns
        -------
        out : torch.tensor
            batch_size x seq_len x embedding_size
        """
        with torch.no_grad():
            x = x * math.sqrt(self.embedding_dim)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x


class Transformer(nn.Module):
    """A transformer block."""

    def __init__(self, embedding_dim, num_heads, mask, hidden_dim, dropout):
        """
        Initialize the transformer block.
        Parameters
        ----------
        embedding_dim : int
            Embedding Dimension
        num_heads : int
            Number of Heads
        mask : bool
            Mask on/off
        hidden_dim : int
            Hidden Dimension
        dropout : float
            Dropout of Network
        """
        super().__init__()

        self.attention = SelfAttention(embedding_dim, num_heads, mask)
        self.mask = mask

        self.first_norm = nn.LayerNorm(embedding_dim)
        self.second_norm = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Initialize Weights and Biases
        nn.init.normal_(self.first_norm.weight)
        nn.init.normal_(self.second_norm.weight)
        nn.init.uniform_(self.first_norm.bias)
        nn.init.uniform_(self.second_norm.bias)
        self.feedforward.apply(self.init_weights)

    def init_weights(self, layer):
        """
        Initialize the weights and bias of a layer.
        Parameters
        ----------
        layer : nn.Module
            layer
        """
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias)

    def forward(self, x):
        """
        Apply attention, norm, and, feed-forward the input.
        Parameters
        ----------
        x : torch.tensor
            batch x seq_len tensor of token indices
        Returns
        -------
        x : torch.tensor
            batch_size x seq_len x embedding_dim
        """

        # Multi-Head Attention
        attention = self.attention(x)
        # Add and Norm
        x = self.first_norm(attention + x)
        x = self.dropout(x)
        # Feed Forward
        feed_forward = self.feedforward(x)
        # Add and Norm
        x = self.second_norm(feed_forward + x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    """Self Attention Module."""

    def __init__(self, embedding_dim, num_heads, mask=False):
        """
        Initialize SelfAttention Module.
        Parameters
        ----------
        embedding_dim : int
            Embedding Dimension
        num_heads : int
            Number of Heads
        mask : bool
            Mask on/off
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mask = mask

        # Create learning layers for query, key, value
        self.query = nn.Linear(embedding_dim, embedding_dim * num_heads, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim * num_heads, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim * num_heads, bias=False)

        self.add_heads = nn.Linear(num_heads * embedding_dim, embedding_dim)

        # Initialize weights and biases
        layers = [self.query, self.key, self.value]
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        """
        Apply Multi-Headed attention.
        Parameters
        ----------
        x : torch.tensor
            batch x seq_len tensor of token indices
        Returns
        -------
        out : torch.tensor
            batch_size x seq_len x embedding_dim
        """

        batch_size, seq_len, embedding_dim = x.size()

        # Compute q, k, v for a batch
        query = self.query(x).view(batch_size, seq_len, self.num_heads, embedding_dim)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, embedding_dim)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, embedding_dim)

        # Get heads into batch dimension to compute the scalar product
        # matrix shape (batch_size*num_heads, seq_len, embedding_dim)
        query = (
            query.transpose(1, 2)
            .contiguous()
            .view(batch_size * self.num_heads, seq_len, embedding_dim)
        )
        key = (
            key.transpose(1, 2)
            .contiguous()
            .view(batch_size * self.num_heads, seq_len, embedding_dim)
        )
        value = (
            value.transpose(1, 2)
            .contiguous()
            .view(batch_size * self.num_heads, seq_len, embedding_dim)
        )

        # Shape -> batch_size * num_heads, seq_len, seq_len
        query_and_key = torch.bmm(query, key.transpose(1, 2)) / (embedding_dim ** (0.5))

        # Mask values
        if self.mask:
            mask_matrix(query_and_key, mask_value=float("-inf"))

        # Row-wise self-attention probabilites
        softmax = nn.functional.softmax(query_and_key, dim=2)

        # Multiple by values
        out = torch.bmm(softmax, value).view(
            batch_size, seq_len, self.num_heads * embedding_dim
        )

        # Add embeddings from all the heads
        out = self.add_heads(out)
        return out


def mask_matrix(matrix, mask_value=0.0):
    """
    Mask out (in-place) all the values in the batch where i < j.
    Parameters
    ----------
    matrix : torch.tensor
        dot product of query and key
    mask_value : float
        value to mask with
    """
    a, b, c = matrix.size()

    indices = torch.triu(b, c, offset=0)
    matrix[:, indices[0], indices[1]] = mask_value