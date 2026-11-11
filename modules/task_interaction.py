import torch
from torch import nn
from modules.feature_bridge import attend


class TaskInteraction(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.queries = nn.ParameterList([nn.Parameter(torch.randn(n, d) * .02)
                                        for n in (config.life_queries, config.shape_queries)])
        self.read = nn.ModuleList([nn.MultiheadAttention(d, config.heads, batch_first=True) for _ in range(2)])
        self.cross = nn.ModuleList([nn.MultiheadAttention(d, config.heads, batch_first=True) for _ in range(2)])
        self.alpha = nn.Parameter(torch.zeros(2))

    def forward(self, lifetime, shape, valid, interact=True):
        b = lifetime.shape[0]
        u, v = [attend(layer, q[None].expand(b, -1, -1), x, valid)
                for layer, q, x in zip(self.read, self.queries, (lifetime, shape))]
        if not interact:
            return u, v
        # Both reads use the pre-interaction pair; no update-order dependence.
        return (u + self.alpha[0] * self.cross[0](u, v, v, need_weights=False)[0],
                v + self.alpha[1] * self.cross[1](v, u, u, need_weights=False)[0])
