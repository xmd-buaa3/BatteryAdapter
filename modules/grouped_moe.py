import torch
from torch import nn
from modules.feature_bridge import masked_pool


class GroupedMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, width = config.hidden_dim, config.expert_dim
        counts = (config.life_experts, config.shape_experts)
        self.experts = nn.ModuleList([
            nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width),
                                         nn.GELU(), nn.Linear(width, d)) for _ in range(count)])
            for count in counts
        ])
        self.gates = nn.ModuleList([nn.Linear(d, count) for count in counts])

    def forward(self, h, mask, task_head=None):
        """Route only the requested task through its expert group.

        ``task_head`` is explicit on protocol runs.  Leaving it unset retains
        the historical two-output behavior for small legacy callers.
        """
        pooled = masked_pool(h, mask)
        selected = None if task_head is None else (0 if task_head == "scalar" else 1)
        groups = range(len(self.experts)) if selected is None else (selected,)
        tasks = [h, h]
        gate_weights = {}
        for group in groups:
            experts = self.experts[group]
            weights = self.gates[group](pooled).softmax(-1)
            outputs = torch.stack([h + expert(h) for expert in experts], dim=1)
            tasks[group] = (outputs * weights[:, :, None, None]).sum(1)
            gate_weights["lifetime" if group == 0 else "shape"] = weights
        return tasks[0], tasks[1], gate_weights
