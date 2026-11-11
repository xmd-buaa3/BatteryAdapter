import torch
from pathlib import Path
from backbones.base import FrozenBackbone, patchify, pack


class Backbone(FrozenBackbone):
    def __init__(self, native, trainable=False):
        super().__init__(native, native.embed_dim, trainable=trainable)
        if native.patch_embed.stride != native.patch_embed.patch_size:
            raise ValueError("this wrapper requires the pinned non-overlapping Toto patches")

    @classmethod
    def load(cls, model_id, revision, trainable=False):
        from toto.model.toto import Toto
        return cls(Toto.from_pretrained(model_id, revision=revision).model, trainable=trainable)

    @classmethod
    def build_random(cls, model_dir):
        """Same pinned Toto config, fresh weights (attribution control)."""
        import json
        from toto.model.toto import TotoBackbone
        path = Path(model_dir) / "config.json"
        if not path.is_file():
            raise RuntimeError(f"random init needs the pinned config.json under {model_dir}")
        kwargs = json.loads(path.read_text(encoding="utf-8"))
        # identical math without the optional xformers dependency
        kwargs["use_memory_efficient_attention"] = False
        return TotoBackbone(**kwargs)

    def encode(self, x, mask):
        values, observed, cycles = patchify(x, mask, self.native.patch_embed.patch_size, side="right")
        b, d, p, k = values.shape
        values, observed = values.flatten(-2), observed.flatten(-2)
        _, loc, scale = self.native.scaler(values, weights=torch.ones_like(values),
                                           padding_mask=observed, prefix_length=None)
        # Some native scalers return their normalized tensor inside no_grad.
        # Reapply those exact statistics here, retaining the input gradient.
        scaled = torch.where(observed, (values-loc)/scale, 0)
        # Isolate wholly missing patches from real patches using native dataset IDs.
        valid = observed.reshape(b, d, p, k).any(-1)
        ids = torch.arange(1, d*p+1, device=x.device).reshape(1, d, p).expand(b, -1, -1)
        ids = torch.where(valid, 0, ids).repeat_interleave(k, -1)
        embeddings, reduced_ids = self.native.patch_embed(scaled, ids)
        # The pinned 1.0 path needs neither forecasts nor unembed.
        hidden = self.native.transformer(embeddings, reduced_ids, None)
        if hidden.shape != (b, d, p, self.width):
            raise ValueError("Toto transformer history shape changed")
        return pack(hidden.to(x.dtype), valid, cycles)
