"""One backbone, three information levels, grouped MoE, Scheme B (plan §§4–8)."""
import torch
from torch import nn
from config import Config
from data.feature_provenance import select_inputs
from data.feature_provenance import allowed_metadata_fields, audit_metadata, metadata_whitelists
from backbones import load_backbone
from modules.curve_compressor import build_compressor
from modules.feature_bridge import FeatureBridge
from modules.early_input import EarlyFusionInputAdapter
from modules.grouped_moe import GroupedMoE
from modules.task_interaction import TaskInteraction
from heads.lifetime_shape_heads import LifetimeShapeHeads


def validate_inputs(inputs, level, c, category_sizes):
    x = select_inputs(inputs, level)
    b = inputs["curves"].shape[0]
    shapes = {"curves": (b, c.observed_cycles, 2, c.curve_channels, c.points),
              "curve_mask": (b, c.observed_cycles, 2, c.curve_channels, c.points),
              "capacity": (b, c.observed_cycles, 1), "capacity_mask": (b, c.observed_cycles, 1),
              "cycle_mask": (b, c.observed_cycles), "soc": (b, c.observed_cycles, 2, c.points),
              "soc_mask": (b, c.observed_cycles, 2, c.points), "soh": (b, c.observed_cycles, 1),
              "soh_mask": (b, c.observed_cycles, 1), "cycle_features": (b, c.observed_cycles, 2),
              "cycle_features_mask": (b, c.observed_cycles, 2)}
    for key, shape in shapes.items():
        if key not in x and key in ("soc", "soc_mask", "soh", "soh_mask", "cycle_features", "cycle_features_mask"):
            continue
        if key not in x or tuple(x[key].shape) != shape:
            raise ValueError(f"{key} must have shape {shape}")
        if key.endswith("mask") and x[key].dtype != torch.bool:
            raise ValueError(f"{key} must be boolean")
    if not x["cycle_mask"].any(-1).all():
        raise ValueError("each battery needs valid observed cycles")
    for key, mask_key in (("curves", "curve_mask"), ("capacity", "capacity_mask"), ("soc", "soc_mask"),
                          ("soh", "soh_mask"), ("cycle_features", "cycle_features_mask")):
        if key in x:
            mask = x[mask_key]
            cycles = x["cycle_mask"].reshape(*x["cycle_mask"].shape, *([1]*(mask.ndim-2)))
            mask = mask & cycles
            if not torch.isfinite(x[key][mask]).all():
                raise ValueError(f"{key} contains nonfinite values marked observed")
            x[mask_key], x[key] = mask, torch.where(mask, x[key], 0)
    if level >= 2:
        metadata, masks = x.get("metadata", {}), x.get("metadata_mask", {})
        allowed_fields = allowed_metadata_fields(c.metadata_profile)
        audit_metadata(metadata, allowed=allowed_fields); audit_metadata(masks, allowed=allowed_fields)
        if set(metadata) != set(masks):
            raise ValueError("each metadata field requires an explicit validity mask")
        numeric_names, categorical_names = metadata_whitelists(c.metadata_profile)
        for key, value in metadata.items():
            expected = (b, c.observed_cycles) if key in numeric_names else (b,)
            mask = masks[key]
            if tuple(value.shape) != expected or mask.shape != value.shape or mask.dtype != torch.bool:
                raise ValueError(f"metadata {key} and boolean mask must have shape {expected}")
            if key in numeric_names and not torch.isfinite(value[mask]).all():
                raise ValueError(f"observed metadata {key} must be finite")
            if key in categorical_names:
                if value.dtype != torch.long or (value < 0).any() or (value >= category_sizes[key]).any():
                    raise ValueError(f"{key} contains an invalid training-vocabulary index")
    if level == 4 and "protocol_features" not in x:
        raise ValueError("Level 4 requires extracted BatteryMFormer protocol_features")
    if level in (4, 5) and "protocol_features" in x:
        features = x.get("protocol_features", {})
        if level == 4:
            required = ("soc_input", "soc_mask", "soh_input",
                        "cycle_level_features", "aging_condition_embedding")
            missing = [name for name in required if name not in features]
            if missing:
                raise ValueError(f"Level 4 protocol_features missing: {missing}")
            soc_shape = tuple(features["soc_input"].shape)
            if soc_shape not in ((b, c.observed_cycles, 2, c.points),
                                 (b, c.observed_cycles, 2 * c.points),
                                 (b, c.observed_cycles, 300)):
                raise ValueError("Level 4 soc_input shape mismatch")
            if tuple(features["soc_mask"].shape) != soc_shape or features["soc_mask"].dtype != torch.bool:
                raise ValueError("Level 4 soc_mask shape/dtype mismatch")
            if tuple(features["soh_input"].shape) != (b, c.observed_cycles, 1):
                raise ValueError("Level 4 soh_input shape mismatch")
            if "soh_mask" in features and (tuple(features["soh_mask"].shape) != tuple(features["soh_input"].shape) or features["soh_mask"].dtype != torch.bool):
                raise ValueError("Level 4 soh_mask shape/dtype mismatch")
            if tuple(features["cycle_level_features"].shape) != (b, c.observed_cycles, 2):
                raise ValueError("Level 4 cycle_level_features shape mismatch")
            if "cycle_level_features_mask" in features and (tuple(features["cycle_level_features_mask"].shape) != tuple(features["cycle_level_features"].shape) or features["cycle_level_features_mask"].dtype != torch.bool):
                raise ValueError("Level 4 cycle_level_features_mask shape/dtype mismatch")
            if tuple(features["aging_condition_embedding"].shape) != (b, 1, c.reference_embedding_dim):
                raise ValueError("Level 4 aging_condition_embedding shape mismatch")
            if "aging_condition_mask" in features and (tuple(features["aging_condition_mask"].shape) != tuple(features["aging_condition_embedding"].shape) or features["aging_condition_mask"].dtype != torch.bool):
                raise ValueError("Level 4 aging_condition_mask shape/dtype mismatch")
    if level == 3 and "soh" in x:
        if (x["soh"][x["soh_mask"]] <= c.eta).any():
            raise ValueError("battery reached EOL within its observed SOH history")
    return x


class BatteryAdapter(nn.Module):
    def __init__(self, config=None, *, backbone=None, vocab_sizes=None):
        super().__init__()
        self.config = c = config or Config()
        if max(c.levels) >= 4 and not c.use_tsfm:
            raise ValueError("Level 4/5 require the shared TSFM + BatteryAdapter model")
        if not c.use_tsfm and max(c.levels) >= 2:
            raise ValueError("early-fusion Level 2/3 requires a TSFM; no_tsfm is not an information-matched arm")
        self.compressor = build_compressor(c)
        trainable = (c.backbone_train == "full" or (c.backbone == "tiny" and c.tiny_trainable)
                     or c.backbone_lora)
        self.backbone = (backbone if backbone is not None
                         else load_backbone(c.backbone, init=c.backbone_init, trainable=trainable,
                                            frozen_scale=c.timestral_frozen_scale)) \
            if c.use_tsfm else None
        if self.backbone is not None and c.backbone_lora:
            # LoRA on attention projections only; base weights stay frozen and the
            # adapter params join the backbone lr group via the trainer grouping.
            # Module-name suffixes (PEFT matches by name suffix): chronos2 encoders
            # expose q/k/v/o linears, Toto attention exposes wQKV/wO.
            from peft import LoraConfig, get_peft_model
            targets = {"chronos2": ["q", "k", "v", "o"], "toto1": ["wQKV", "wO"]}
            lora = LoraConfig(r=c.lora_r, lora_alpha=c.lora_alpha,
                              lora_dropout=c.lora_dropout, bias="none",
                              target_modules=targets[c.backbone])
            self.backbone.native = get_peft_model(self.backbone.native, lora)
            self.backbone.trainable = True  # keep train()/eval() under trainer control
        if self.backbone is not None and not self.backbone.trainable:
            self.backbone.requires_grad_(False)
        width = self.backbone.width if self.backbone is not None else c.hidden_dim
        self.input_adapter = EarlyFusionInputAdapter(c, vocab_sizes or {})
        self.bridge = FeatureBridge(c, width, input_channels=self.input_adapter.total_channels)
        self.moe = GroupedMoE(c) if c.use_moe else None
        self.interaction = TaskInteraction(c)
        self.heads = LifetimeShapeHeads(c)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
            if mode and self.config.backbone_lora:
                for module in self.backbone.native.modules():
                    dropout = getattr(module, "lora_dropout", None)
                    if isinstance(dropout, nn.ModuleDict):
                        dropout.train()
        return self

    def validate(self, inputs, level):
        return validate_inputs(inputs, level, self.config,
                              {k: v.num_embeddings for k, v in self.input_adapter.static.category.items()})

    def encode_base(self, inputs, level=1):
        if self.config.compressor == "capacity_only" and not inputs["capacity_mask"].flatten(1).any(-1).all():
            raise ValueError("capacity_only requires recorded per-cycle capacity; zero-filled strict cache is insufficient")
        compressed, local, local_mask, features_valid = self.compressor(inputs["curves"], inputs["curve_mask"])
        if "curve" in self.config.input_drop:
            compressed = torch.zeros_like(compressed)
            features_valid = torch.zeros_like(features_valid, dtype=torch.bool)
        values, mask, layout = self.input_adapter(inputs["capacity"], inputs["capacity_mask"],
                                                  compressed, features_valid, inputs, level)
        encoded = self.backbone(values, mask) if self.backbone is not None else None
        return dict(compressed=compressed, local=local, local_mask=local_mask,
                    input_values=values, input_mask=mask, layout=layout,
                    encoded=encoded, level=level)

    def predict_encoded(self, inputs, level, query, base, return_embedding=False):
        compressed, mask, encoded = base["compressed"], base["input_mask"], base["encoded"]
        capacity = inputs["capacity"] if level in (2, 3) else torch.zeros_like(inputs["capacity"])
        h = self.bridge(encoded, capacity, compressed, mask, inputs["cycle_mask"])
        task = self.config.prediction_task
        # "joint" needs both expert groups; the single-task protocols route
        # only their own group through the MoE.
        moe_task = task if task in ("scalar", "curve") else None
        u, v, gates = (self.moe(h, inputs["cycle_mask"], moe_task)
                       if self.moe is not None else (h, h, {}))
        if task == "scalar":
            u, v = u, u
        elif task == "curve":
            u, v = v, v
        elif task == "joint":
            u, v = self.interaction(u, v, inputs["cycle_mask"], self.config.use_interaction)
        else:
            raise ValueError(f"unsupported prediction task {task!r}")
        prediction = self.heads(u, v, inputs, level, query)
        if return_embedding:
            weights = inputs["cycle_mask"][..., None].to(v.dtype)
            prediction["embedding"] = (v * weights).sum(1) / weights.sum(1).clamp_min(1)
        return {**prediction, "gate_weights": gates}

    def query_grid(self, device):
        c = self.config
        if c.protocol == "batterymformer":
            return torch.arange(1, c.horizon + 1, device=device, dtype=torch.float32)
        return torch.arange(c.observed_cycles+1, c.observed_cycles+c.horizon+1, device=device, dtype=torch.float32)

    def forward(self, inputs, information_level=1, query_cycles=None, return_embedding=False):
        x = self.validate(inputs, information_level)
        query = self.query_grid(x["curves"].device) if query_cycles is None else query_cycles.to(x["curves"].device)
        return self.predict_encoded(x, information_level, query,
                                    self.encode_base(x, information_level), return_embedding=return_embedding)
