import json
from pathlib import Path

import torch
from backbones.base import FrozenBackbone, patchify, pack


class Backbone(FrozenBackbone):
    def __init__(self, native, trainable=False):
        super().__init__(native, native.config.d_model, trainable=trainable)

    @classmethod
    def load(cls, model_id, revision, trainable=False):
        from momentfm import MOMENTPipeline
        native = MOMENTPipeline.from_pretrained(model_id, revision=revision, model_kwargs={"task_name": "embedding"})
        native.init()
        return cls(native, trainable=trainable)

    @classmethod
    def build_random(cls, model_dir):
        """Same pinned MOMENT architecture, fresh weights.

        momentfm's hub constructor cannot build weight-less from a local config,
        so the honest random path loads the structure then re-applies every
        module's own default reset_parameters (the exact initialization fresh
        training would start from)."""
        import torch
        from momentfm import MOMENTPipeline
        native = MOMENTPipeline.from_pretrained(str(model_dir), model_kwargs={"task_name": "embedding"})
        native.init()
        loaded = {k: v.detach().clone() for k, v in native.state_dict().items()}
        with torch.no_grad():
            for module in native.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
                elif type(module).__name__.endswith(("LayerNorm", "RMSNorm")) \
                        and hasattr(module, "weight"):
                    # T5LayerNorm-style norms define no reset_parameters; their
                    # own construction scheme is weight=ones (bias=zeros).
                    module.weight.fill_(1.0)
                    if getattr(module, "bias", None) is not None:
                        module.bias.zero_()
            # nothing pretrained may survive into the random init
            for name, param in native.named_parameters():
                if name in loaded and torch.equal(param.data, loaded[name]):
                    param.normal_(0.0, 0.02)
        return native

    def encode(self, x, mask):
        if x.shape[1] > self.native.seq_len:
            raise ValueError("MOMENT input exceeds the pinned sequence length")
        values, observed, cycles = patchify(x, mask, self.native.patch_len,
                                            minimum=self.native.seq_len, side="right")
        b, d, p, k = values.shape
        values, observed = values.reshape(b*d, p*k), observed.reshape(b*d, p*k)
        valid = observed.reshape(b*d, p, k).all(-1)
        active = valid.any(-1)
        hidden = x.new_zeros(b*d, p, self.width)
        if active.any():
            output = self.native.embed(x_enc=values[active, None], input_mask=observed[active].long(), reduction="none")
            hidden[active] = output.embeddings[:, 0].to(x.dtype)
        return pack(hidden.reshape(b, d, p, self.width), valid.reshape(b, d, p), cycles)
