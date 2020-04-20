import torch
from torch import nn
import torch.nn.functional as F


# This class implements multi-head, scaled dot-product self attention.
class SelfAttention(nn.Module):
    def __init__(self, k, heads = 8):
        super().__init__()
        self.k, self.heads = k, heads

        # These compute the queries, keys and values for all heads (as a single concatenated vector)
        self.toKeys = nn.Linear(k, k * heads, bias=False)
        self.toQueries = nn.Linear(k, k * heads, bias=False)
        self.toValues = nn.Linear(k, k * heads, bias=False)

        # This unifies the outputs of the different heads into a single k-vector
        self.unifyHeads = nn.Linear(heads * k, k)

    def forward(self, x):
        # Input x is a sequence of t vectors (words) of dimension k (emb dim) as a t by k matrix 𝐗.
        # Including a mini-batch dimension b, gives us an input tensor of size (b,t,k).
        b, t, k = x.size()
        h = self.heads

        # The output of each linear module has size (b, t, h*k),
        # which we simply reshape to (b, t, h, k) give each head its own dimension.
        queries = self.toQueries(x).view(b,t,h,k)
        keys = self.toKeys(x).view(b,t,h,k)
        values = self.toValues(x).view(b,t,h,k)

        # Compute dot products

        # This is same for every head, so we fold the heads into the batch dim
        # Since the head and batch dimension are not next to each other,
        # we need to transpose before we reshape. (kinda costly)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        # Dot products have to be scaled later by sqrt(k), so we'll scale queries and keys by sqrt(sqrt),
        # to save memory for longer sequences

        queries = queries / (k ** (1 / 4))
        keys = keys / (k ** (1 / 4))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))  # dot has size (b*h, t, t) containing raw weights

        dot = F.softmax(dot, dim=2)
        # dot now contains row-wise normalized weights

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, k)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyHeads(out)