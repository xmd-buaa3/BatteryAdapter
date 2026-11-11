import torch
from torch import nn


def attend(layer, queries, values, valid):
    """Skip rows without keys; invalid values cannot contaminate attention."""
    active = valid.any(-1)
    result = torch.zeros_like(queries)
    if active.any():
        clean = torch.where(valid[active, :, None], values[active], 0)
        result[active] = layer(queries[active], clean, clean,
                               key_padding_mask=~valid[active], need_weights=False)[0]
    return result


def masked_pool(values, mask):
    return torch.where(mask[..., None], values, 0).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class FeatureBridge(nn.Module):
    def __init__(self, config, native_width, input_channels=None):
        super().__init__()
        self.config = config
        d, n = config.hidden_dim, input_channels or config.compressed_dim + 1
        self.project = nn.Sequential(nn.Linear(native_width, d), nn.LayerNorm(d))
        self.variable = nn.Embedding(n, d)
        self.cycle = nn.Embedding(config.observed_cycles + 1, d)
        self.role = nn.Embedding(3, d)
        self.capacity_query = nn.Linear(1, d)
        self.curve_query = nn.Linear(config.compressed_dim, d)
        self.missing = nn.Linear(n, d, bias=False)
        self.attention = nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.norm = nn.LayerNorm(d)

    def forward(self, encoded, capacity, compressed, input_mask, cycle_mask):
        s = capacity.shape[1]
        query = self.capacity_query(capacity) + self.curve_query(compressed)
        query = query + self.cycle(torch.arange(1, s+1, device=query.device))[None]
        query = query + self.missing((~input_mask).to(query.dtype))
        if encoded is not None:
            values = self.project(encoded.tokens.to(query.dtype))
            if encoded.token_variable.numel() and int(encoded.token_variable.max()) >= self.variable.num_embeddings:
                raise ValueError("early-fusion token channel exceeds the bridge vocabulary")
            values = values + self.variable(encoded.token_variable)
            values = values + self.cycle(encoded.token_cycle)
            # Capacity, compressed curves, and early-fusion condition slots get
            # distinct roles.  token_variable is the packed channel index.
            role = torch.where(encoded.token_variable == 0, 0,
                               torch.where(encoded.token_variable <= self.config.compressed_dim, 1, 2))
            values = values + self.role(role)
            if self.config.bridge_read == "attention":
                read = attend(self.attention, query, values, encoded.token_valid)
            else:
                # A1 control: replace the cross-token read, keep the local query path
                # (capacity/curve/cycle/missing embeddings) and the residual form.
                read = masked_pool(values, encoded.token_valid)[:, None, :].expand_as(query)
            query = query + self.alpha * read
        return torch.where(cycle_mask[..., None], self.norm(query), 0)
