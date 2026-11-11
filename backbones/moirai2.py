import json
from pathlib import Path

import torch
from backbones.base import FrozenBackbone, patchify, pack


class Backbone(FrozenBackbone):
    def __init__(self, native, trainable=False):
        super().__init__(native, native.d_model, trainable=trainable)

    @classmethod
    def load(cls, model_id, revision, trainable=False):
        from uni2ts.model.moirai2 import Moirai2Module
        return cls(Moirai2Module.from_pretrained(model_id, revision=revision), trainable=trainable)

    @classmethod
    def build_random(cls, model_dir):
        """Same pinned Moirai2 module config, fresh weights (attribution control)."""
        import json
        from uni2ts.model.moirai2 import Moirai2Module
        raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        return Moirai2Module(**raw)

    def encode(self, x, mask):
        values, observed, cycles = patchify(x, mask, self.native.patch_size)
        b, d, p, k = values.shape
        if p > self.native.max_seq_len:
            raise ValueError("Moirai context exceeds its native patch limit")
        values, observed = values.reshape(b*d, p, k), observed.reshape(b*d, p, k)
        active = observed.any(dim=(1, 2))
        hidden = x.new_zeros(b*d, p, self.width)
        if active.any():
            y, m = values[active], observed[active]
            times = torch.arange(p, device=x.device)[None].expand(y.shape[0], -1)
            samples = m.any(-1).long()
            variables = torch.zeros_like(times)  # each row is one independent variable
            # Native scaler mutates its statistics in place. Treat statistics as
            # constants, while keeping normalization and encoder input differentiable.
            loc, scale = self.native.scaler(y.detach(), m, samples, variables)
            z = torch.where(m, (y - loc) / scale, 0)
            z = self.native.in_proj(torch.cat([z, m.to(z.dtype)], -1))
            attention = (samples[:, :, None] == samples[:, None, :]) & (times[:, :, None] >= times[:, None, :])
            h = self.native.encoder(z, attention, time_id=times, var_id=variables)
            hidden[active] = h.to(x.dtype)  # stop before out_proj
        return pack(hidden.reshape(b, d, p, self.width), observed.any(-1).reshape(b, d, p), cycles)
