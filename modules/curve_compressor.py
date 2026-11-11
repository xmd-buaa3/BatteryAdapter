import torch
from torch import nn
from torch.nn import functional as F


class CurveCompressor(nn.Module):
    """One shared CNN; 2 phases x 4 local regions x 2 features by default."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        v, width, features = config.curve_channels, config.local_dim, config.compressed_dim // 8
        self.encoder = nn.Sequential(nn.Conv1d(2*v, width, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(width, width, 3, padding=1), nn.GELU())
        self.project = nn.Linear(width, features)
        self.reconstruct = nn.Sequential(nn.Linear(4*features, width), nn.GELU(),
                                          nn.Linear(width, v*config.points))

    def forward(self, curves, mask):
        b, s, p, v, l = curves.shape
        mask = mask & torch.isfinite(curves)
        values = torch.where(mask, curves, 0).reshape(b*s*p, v, l)
        point_mask = mask.reshape(b*s*p, v, l).any(1, keepdim=True)
        encoded = self.encoder(torch.cat([values, mask.reshape(b*s*p, v, l).to(values.dtype)], 1))
        counts = F.adaptive_avg_pool1d(point_mask.to(values.dtype), 4)
        local = F.adaptive_avg_pool1d(encoded * point_mask, 4) / counts.clamp_min(1e-6)
        local = local.transpose(1, 2).reshape(b, s, p, 4, -1)
        valid = counts.squeeze(1).reshape(b, s, p, 4) > 0
        local = torch.where(valid[..., None], local, 0)
        compressed = torch.where(valid[..., None], self.project(local), 0)
        feature_mask = valid[..., None].expand_as(compressed).reshape(b, s, -1)
        return compressed.reshape(b, s, -1), local, valid, feature_mask

    def decode(self, compressed):
        b, s, _ = compressed.shape
        return self.reconstruct(compressed.reshape(b, s, 2, -1)).reshape(
            b, s, 2, self.config.curve_channels, self.config.points)


class CPFlattenCompressor(nn.Module):
    """A2 control: whole-cycle flatten -> MLP (CPTransformer-style cycle embedding).

    Consumes the identical curves/mask contract and emits the identical
    [B, S, compressed_dim] channels, but replaces phase-wise CNN + regional
    pooling with one whole-cycle embedding per cycle.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        v, width, out = config.curve_channels, config.local_dim, config.compressed_dim
        # both phases x channels x points, values and validity packed side by side
        self.in_dim = 4 * v * config.points
        self.encoder = nn.Sequential(nn.Linear(self.in_dim, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU())
        self.project = nn.Linear(width, out)
        self.reconstruct = nn.Sequential(nn.Linear(out, width), nn.GELU(),
                                         nn.Linear(width, 2 * v * config.points))

    def forward(self, curves, mask):
        b, s, p, v, l = curves.shape
        mask = mask & torch.isfinite(curves)
        flat = torch.cat([torch.where(mask, curves, 0), mask.to(curves.dtype)], -1).reshape(b, s, -1)
        if flat.shape[-1] != self.in_dim:
            raise ValueError("curve layout disagrees with the cp_flatten contract")
        valid = mask.any(dim=(2, 3, 4))
        compressed = torch.where(valid[..., None], self.project(self.encoder(flat)), 0)
        local = compressed[:, :, None, None, :]
        cycle_valid = valid[..., None, None].expand(b, s, p, 1)
        feature_mask = valid[..., None].expand(b, s, self.config.compressed_dim)
        return compressed, local, cycle_valid, feature_mask

    def decode(self, compressed):
        b, s, _ = compressed.shape
        return self.reconstruct(compressed).reshape(
            b, s, 2, self.config.curve_channels, self.config.points)


class CapacityOnlyCompressor(nn.Module):
    """Do not read curves or create trainable curve features when CAF is removed."""
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, curves, mask):
        b, s = curves.shape[:2]
        compressed = curves.new_zeros(b, s, self.config.compressed_dim)
        local = curves.new_zeros(b, s, 2, 4, self.config.local_dim)
        valid = torch.zeros(b, s, 2, 4, dtype=torch.bool, device=curves.device)
        return compressed, local, valid, torch.zeros_like(compressed, dtype=torch.bool)


def build_compressor(config):
    if config.compressor == "phase_cnn":
        return CurveCompressor(config)
    if config.compressor == "cp_flatten":
        return CPFlattenCompressor(config)
    if config.compressor == "capacity_only":
        return CapacityOnlyCompressor(config)
    raise ValueError(f"unknown compressor {config.compressor!r}")
