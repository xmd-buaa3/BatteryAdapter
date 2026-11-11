import torch
from backbones.base import FrozenBackbone, patchify, pack


class Backbone(FrozenBackbone):
    def __init__(self, native, trainable=False):
        super().__init__(native, native.config.hidden_size, trainable=trainable)

    @classmethod
    def load(cls, model_id, revision, trainable=False):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                     code_revision=revision, trust_remote_code=True)
        return cls(model.model, trainable=trainable)

    @classmethod
    def build_random(cls, model_dir):
        """Same pinned Sundial config (remote code), fresh weights (attribution control)."""
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        return model.model

    def encode(self, x, mask):
        values, observed, cycles = patchify(x, mask, self.native.config.input_token_len)
        b, d, p, k = values.shape
        values, observed = values.reshape(b*d, p, k), observed.reshape(b*d, p, k)
        valid = observed.any(-1)
        active = valid.any(-1)
        hidden = x.new_zeros(b*d, p, self.width)
        if active.any():
            # Use the pinned embedder's weights with explicit per-point validity;
            # its input_ids path assumes every supplied point is observed.
            embedding = self.native.embed_layer
            z = torch.cat([values[active], observed[active].to(x.dtype)], -1)
            z = embedding.dropout(embedding.output_layer(embedding.act(embedding.hidden_layer(z)))) + embedding.residual_layer(z)
            result = self.native(inputs_embeds=z, attention_mask=valid[active], use_cache=False, return_dict=True)
            hidden[active] = result.last_hidden_state.to(x.dtype)
        return pack(hidden.reshape(b, d, p, self.width), valid.reshape(b, d, p), cycles)
