"""Level-controlled inputs that are packed before the TSFM.

The native TSFM wrappers normalize each variable independently.  Static fields
therefore cannot be broadcast as a constant sequence: the native normalizer
would remove the information.  ``StaticPatternEncoder`` maps each static field
to a small, deterministic non-constant cycle pattern before the wrapper sees
it.  This keeps the public wrapper contract ``[B, S, D] + mask`` unchanged
while making the level switch an actual TSFM input switch.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from data.feature_provenance import metadata_whitelists


class StaticPatternEncoder(nn.Module):
    """Encode a static value into a non-constant, cycle-aligned carrier."""

    def __init__(self, config, vocab_sizes):
        super().__init__()
        s, k, d = config.observed_cycles, config.static_basis_dim, config.hidden_dim
        t = torch.linspace(-1.0, 1.0, s)
        columns = [t, t.square(), torch.sin(math.pi * t), torch.cos(math.pi * t)]
        for frequency in range(2, k + 1):
            columns.extend((torch.sin(frequency * math.pi * t), torch.cos(frequency * math.pi * t)))
        basis = torch.stack(columns[:k], -1)
        basis = (basis - basis.mean(0, keepdim=True)) / basis.std(0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("basis", basis, persistent=True)
        numeric_names, category_names = metadata_whitelists(config.metadata_profile)
        self.numeric = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, k))
            for name in numeric_names
        })
        self.category = nn.ModuleDict({
            name: nn.Embedding(int(vocab_sizes.get(name, 1)), k)
            for name in category_names
        })

    def forward(self, name, value, valid, cycle_mask):
        # value is [B, S] for numeric metadata and [B] for categorical metadata.
        if name in self.numeric:
            if value.ndim == 2:
                effective = valid.bool() & torch.isfinite(value)
                clean = torch.where(effective, value, torch.zeros_like(value))
                weights = effective.to(value.dtype)
                scalar = clean.sum(-1) / weights.sum(-1).clamp_min(1)
            else:
                scalar = value
            coeff = self.numeric[name](torch.nan_to_num(scalar)[..., None])
            present = valid.any(-1) if valid.ndim == 2 else valid
        else:
            coeff = self.category[name](value.long())
            present = valid.bool()
        pattern = torch.einsum("bk,sk->bs", coeff, self.basis)
        mask = present[:, None] & cycle_mask
        return torch.where(mask, pattern, torch.zeros_like(pattern))[..., None], mask[..., None]


class EarlyFusionInputAdapter(nn.Module):
    """Pack all permitted level fields into one fixed-width TSFM input."""

    dynamic_metadata = ("temperature", "charge_rate_summary", "discharge_rate_summary")

    def __init__(self, config, vocab_sizes=None):
        super().__init__()
        self.config = config
        self.vocab_sizes = dict(vocab_sizes or {})
        self.static = StaticPatternEncoder(config, self.vocab_sizes)
        numeric_names, category_names = metadata_whitelists(config.metadata_profile)
        self.base_width = 1 + config.compressed_dim
        # Static slots: every whitelist numeric that is not a measured per-cycle
        # series (registered numerics + nominal capacity) plus all categoricals.
        self.static_metadata = (tuple(name for name in numeric_names if name not in self.dynamic_metadata)
                                + tuple(category_names))
        self.metadata_width = len(self.dynamic_metadata) + len(self.static_metadata)
        # The CE/EE state slot belongs to the Level 1-3 information ladder only:
        # protocol runs (4/5) carry cycle-level features through ``protocol_cycle``
        # and keep the original 43-channel layout.
        self.cycle_slot = max(config.levels) <= 3 and config.use_cycle_features
        self.state_width = 2 * config.state_bins + 1 + (2 if self.cycle_slot else 0)
        self.protocol_width = 2 + 2 * config.state_bins + 1
        self.protocol_condition = nn.Sequential(nn.Linear(2, 2), nn.GELU())
        self.cycle_condition = nn.Sequential(nn.Linear(2, 2), nn.GELU())
        self.protocol_aging = nn.Sequential(nn.Linear(config.reference_embedding_dim, 1), nn.Tanh())
        self.total_channels = self.base_width + self.metadata_width + self.state_width + self.protocol_width
        self.layout = self._make_layout()

    def _make_layout(self):
        index = 0
        layout = {}
        for name, width in (
            ("capacity", 1), ("compressed", self.config.compressed_dim),
            *[(name, 1) for name in self.dynamic_metadata],
            *[(name, 1) for name in self.static_metadata],
            ("soc", 2 * self.config.state_bins), ("soh", 1),
            *([("cycle_features", 2)] if self.cycle_slot else []),
            ("protocol_cycle", 2), ("protocol_soc", 2 * self.config.state_bins),
            ("protocol_aging", 1),
        ):
            layout[name] = (index, index + width)
            index += width
        return layout

    def _slot(self, value, mask, cycle_mask):
        value = torch.nan_to_num(value)
        mask = mask.bool() & cycle_mask
        return torch.where(mask, value, torch.zeros_like(value)), mask

    def forward(self, capacity, capacity_mask, compressed, compressed_mask, inputs, level):
        if level not in (1, 2, 3, 4, 5):
            raise ValueError("information_level must be 1 through 5")
        b, s = capacity.shape[:2]
        cycle_mask = inputs["cycle_mask"].bool()
        values, masks = [], []

        # Level 1 receives the physical capacity curve through ``compressed``.
        # The cycle-level scalar is deliberately gated off until Level 2.
        if level >= 2 and not (level == 4 and self.config.information_ablation == 1):
            values.append(torch.nan_to_num(capacity))
            masks.append(capacity_mask.bool() & cycle_mask[..., None])
        else:
            values.append(torch.zeros_like(capacity))
            masks.append(torch.zeros_like(capacity_mask, dtype=torch.bool))
        values.append(torch.nan_to_num(compressed))
        masks.append(compressed_mask.bool() & cycle_mask[..., None])

        metadata = inputs.get("metadata", {}) if level >= 2 else {}
        metadata_mask = inputs.get("metadata_mask", {}) if level >= 2 else {}
        for name in self.dynamic_metadata:
            if name in metadata:
                value = metadata[name]
                valid = metadata_mask[name]
                value, valid = self._slot(value[..., None], valid[..., None], cycle_mask[..., None])
                # Native per-variable normalizers erase a constant sequence.
                # Replace only constant/empty numeric rows with the same static
                # carrier used for nominal capacity; changing metadata still
                # changes the pre-TSFM input.
                varying = (value.squeeze(-1).masked_fill(~valid.squeeze(-1), 0).var(-1, unbiased=False) > 1e-8)
                if (~varying).any():
                    pattern, pattern_mask = self.static(name, metadata[name], metadata_mask[name], cycle_mask)
                    value = torch.where(varying[:, None, None], value, pattern)
                    valid = torch.where(varying[:, None, None], valid, pattern_mask)
            else:
                value = capacity.new_zeros(b, s, 1)
                valid = torch.zeros(b, s, 1, dtype=torch.bool, device=capacity.device)
            values.append(value)
            masks.append(valid)

        for name in self.static_metadata:
            if name in metadata:
                value = metadata[name]
                valid = metadata_mask[name]
                value, valid = self.static(name, value, valid, cycle_mask)
            else:
                value = capacity.new_zeros(b, s, 1)
                valid = torch.zeros(b, s, 1, dtype=torch.bool, device=capacity.device)
            values.append(value)
            masks.append(valid)

        if level in (3,) and self.config.use_soc and "soc" in inputs:
            soc, soc_mask = inputs["soc"], inputs["soc_mask"]
            soc_mask = soc_mask.bool() & torch.isfinite(soc)
            numerator = F.adaptive_avg_pool1d(
                torch.where(soc_mask, soc, 0).reshape(b * s * 2, 1, -1), self.config.state_bins)
            denominator = F.adaptive_avg_pool1d(
                soc_mask.to(capacity.dtype).reshape(b * s * 2, 1, -1), self.config.state_bins)
            pooled = (numerator / denominator.clamp_min(1e-6)).reshape(b, s, 2 * self.config.state_bins)
            pooled_mask = (denominator.reshape(b, s, 2 * self.config.state_bins) > 0) & cycle_mask[..., None]
        else:
            pooled = capacity.new_zeros(b, s, 2 * self.config.state_bins)
            pooled_mask = torch.zeros(b, s, 2 * self.config.state_bins, dtype=torch.bool, device=capacity.device)
        values.append(torch.nan_to_num(pooled))
        masks.append(pooled_mask)

        if level == 3 and self.config.use_soh and "soh" in inputs:
            soh, soh_mask = self._slot(inputs["soh"], inputs["soh_mask"], cycle_mask[..., None])
        elif (level == 4 and self.config.information_ablation is None and self.config.use_soh and
              "observed_end" in inputs.get("protocol_features", {})):
            strict_features = inputs["protocol_features"]
            soh, soh_mask = self._slot(strict_features["soh_input"], strict_features["soh_mask"],
                                       cycle_mask[..., None])
        else:
            soh = capacity.new_zeros(b, s, 1)
            soh_mask = torch.zeros(b, s, 1, dtype=torch.bool, device=capacity.device)
        values.append(soh)
        masks.append(soh_mask)

        if self.cycle_slot:
            if level == 3 and "cycle_features" in inputs:
                features = inputs["cycle_features"]
                if tuple(features.shape) != (b, s, 2):
                    raise ValueError("Level 3 cycle_features must have shape [B,S,2]")
                feature_valid = (inputs.get("cycle_features_mask", torch.isfinite(features)).bool()
                                 & torch.isfinite(features) & cycle_mask[..., None])
                feature_values = torch.nan_to_num(self.cycle_condition(features))
                feature_values = torch.where(feature_valid, feature_values, 0)
            else:
                feature_values = capacity.new_zeros(b, s, 2)
                feature_valid = torch.zeros_like(feature_values, dtype=torch.bool)
            values.append(feature_values)
            masks.append(feature_valid)

        protocol = inputs.get("protocol_features", {}) if level in (4, 5) else {}
        cycle_features = (protocol.get("cycle_level_features")
                          if self.config.information_ablation is None and "cyc" not in self.config.input_drop else None)
        if cycle_features is None:
            cycle_values = capacity.new_zeros(b, s, 2)
            cycle_valid = torch.zeros_like(cycle_values, dtype=torch.bool)
        else:
            if tuple(cycle_features.shape) != (b, s, 2):
                raise ValueError("Level 4 cycle_level_features must have shape [B,S,2]")
            cycle_valid = protocol.get("cycle_level_features_mask",
                                       torch.isfinite(cycle_features)).bool()
            cycle_valid = cycle_valid & torch.isfinite(cycle_features) & cycle_mask[..., None]
            cycle_values = torch.nan_to_num(self.protocol_condition(cycle_features))
            cycle_values = torch.where(cycle_valid, cycle_values, 0)
        values.append(cycle_values)
        masks.append(cycle_valid)

        soc_input, soc_mask = ((protocol.get("soc_input"), protocol.get("soc_mask"))
                               if self.config.information_ablation is None and "cyc" not in self.config.input_drop else (None, None))
        if soc_input is None or soc_mask is None:
            pooled = capacity.new_zeros(b, s, 2 * self.config.state_bins)
            pooled_mask = torch.zeros_like(pooled, dtype=torch.bool)
        else:
            if soc_input.ndim == 3 and soc_input.shape[-1] in (2 * self.config.points, 300):
                points = soc_input.shape[-1] // 2
                soc_input = soc_input.reshape(b, s, 2, points)
                soc_mask = soc_mask.reshape(b, s, 2, points)
            if tuple(soc_input.shape) != (b, s, 2, self.config.points):
                raise ValueError("protocol soc_input must have shape [B,S,2,P] or [B,S,2P]")
            soc_mask = soc_mask.bool() & torch.isfinite(soc_input)
            numerator = F.adaptive_avg_pool1d(torch.where(soc_mask, soc_input, 0).reshape(b*s*2, 1, -1), self.config.state_bins)
            denominator = F.adaptive_avg_pool1d(soc_mask.to(capacity.dtype).reshape(b*s*2, 1, -1), self.config.state_bins)
            pooled = (numerator / denominator.clamp_min(1e-6)).reshape(b, s, 2*self.config.state_bins)
            pooled_mask = (denominator.reshape(b, s, 2*self.config.state_bins) > 0) & cycle_mask[..., None]
            pooled = torch.where(pooled_mask, pooled, 0)
        values.append(pooled)
        masks.append(pooled_mask)

        aging = (protocol.get("aging_condition_embedding")
                 if self.config.information_ablation != 1 and "aging" not in self.config.input_drop else None)
        if aging is None:
            aging_value = capacity.new_zeros(b, s, 1)
            aging_mask = torch.zeros_like(aging_value, dtype=torch.bool)
        else:
            if tuple(aging.shape) != (b, 1, self.config.reference_embedding_dim):
                raise ValueError("Level 4 aging_condition_embedding must have shape [B,1,D]")
            aging_value = self.protocol_aging(torch.nan_to_num(aging[:, 0])).unsqueeze(1).expand(-1, s, -1)
            aging_mask = protocol.get("aging_condition_mask")
            if aging_mask is not None:
                present = (aging_mask.bool() & torch.isfinite(aging)).any(dim=-1).squeeze(1)
                aging_mask = present[:, None, None] & cycle_mask[..., None]
            else:
                feature_mask = protocol.get("protocol_feature_mask")
                if feature_mask is None:
                    present = torch.isfinite(aging).all(-1).squeeze(1)
                else:
                    present = feature_mask.bool()
                    if present.ndim > 1:
                        present = present[..., 0]
                    if present.ndim == 1 and present.shape[0] != b:
                        present = present[-1].expand(b)
                    if present.ndim == 0:
                        present = present.expand(b)
                aging_mask = present[:, None, None] & cycle_mask[..., None]
            aging_value = torch.where(aging_mask, aging_value, 0)
        values.append(aging_value)
        masks.append(aging_mask)

        packed_values = torch.cat(values, -1)
        packed_mask = torch.cat(masks, -1)
        if packed_values.shape[-1] != self.total_channels or packed_mask.shape != packed_values.shape:
            raise ValueError("early-fusion input layout has an unexpected width")
        if not torch.isfinite(packed_values).all():
            raise FloatingPointError("early-fusion input contains nonfinite values")
        packed_values = torch.where(packed_mask, packed_values, torch.zeros_like(packed_values))
        return packed_values, packed_mask, self.layout
