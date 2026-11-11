import torch
from pathlib import Path
from torch import nn
from backbones.base import FrozenBackbone, patchify, pack


class HistoryNorm(nn.Module):
    """Native statistics with a differentiable, NaN-safe application to observations.

    Native variance.sqrt backward is undefined for constant rows and missing
    points. Freezing statistics preserves its finite forward values and the
    gradient from history features back to observed input values.
    """
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x, loc_scale=None):
        _, (loc, scale) = self.original(x.detach(), loc_scale)
        clean = torch.where(torch.isfinite(x), x.float(), loc)
        result = (clean-loc)/scale
        if self.original.use_arcsinh:
            result = torch.arcsinh(result)
        return result.to(x.dtype), (loc, scale)


class Backbone(FrozenBackbone):
    def __init__(self, native, trainable=False):
        native.instance_norm = HistoryNorm(native.instance_norm)
        super().__init__(native, native.config.d_model, trainable=trainable)

    @classmethod
    def load(cls, model_id, revision, trainable=False):
        from chronos import Chronos2Pipeline
        return cls(Chronos2Pipeline.from_pretrained(model_id, revision=revision, device_map="cpu").model,
                   trainable=trainable)

    @classmethod
    def build_random(cls, model_dir):
        """Same architecture and chronos config, fresh weights (attribution control)."""
        import json
        from chronos.chronos2.config import Chronos2CoreConfig
        from chronos.chronos2.model import Chronos2Model
        path = Path(model_dir) / "config.json"
        if not path.is_file():
            raise RuntimeError(f"random init needs the pinned config.json under {model_dir}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        config = Chronos2CoreConfig(**{k: v for k, v in raw.items() if k != "chronos_config"})
        config.chronos_config = dict(raw["chronos_config"])
        return Chronos2Model(config)

    def encode(self, x, mask):
        b, s, d = x.shape
        if s > self.native.chronos_config.context_length:
            raise ValueError("observation window exceeds Chronos context limit")
        values, valid, cycles = patchify(x, mask, self.native.chronos_config.input_patch_size)
        p = values.shape[-2]
        # Native instance_norm uses NaNs, not context_mask, to omit missing data.
        context = torch.where(mask, x, torch.nan).transpose(1, 2).reshape(b*d, s)
        context_mask = mask.transpose(1, 2).reshape(b*d, s)
        active = context_mask.any(-1)
        hidden = x.new_zeros(b*d, p, self.width)
        if active.any():
            groups = torch.arange(b, device=x.device).repeat_interleave(d)[active]
            encoded, _, _, history_patches = self.native.encode(
                context=context[active], context_mask=context_mask[active], group_ids=groups,
                num_output_patches=1)  # fixed; no targets and no query-dependent placeholders
            if history_patches != p:
                raise ValueError("Chronos history patch contract changed")
            hidden[active] = encoded.last_hidden_state[:, :p].to(x.dtype)
        return pack(hidden.reshape(b, d, p, self.width), valid.any(-1), cycles)
