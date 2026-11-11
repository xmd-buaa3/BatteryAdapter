import torch
from torch import nn
from torch.nn import functional as F
from heads.scheme_b import decode


def _last_observed_slot(mask):
    """Per-row index of the last observed slot (the observation end).

    Loaders emit prefix-contiguous masks, so with a truncated observation
    (manifest ``observed_used`` below ``observed_cycles``) this is cycle N-1
    and the start anchor reads the SOH the cell actually exposed; with the
    full window it is the storage end, matching the historical ``[:, -1]``
    readout bit for bit.  All-masked rows fall back to slot 0 with a False
    mask, i.e. the estimated start, as before.
    """
    flipped = mask.int().flip(-1)
    last = mask.shape[-1] - 1 - flipped.argmax(-1)
    return torch.where(mask.any(-1), last, torch.zeros_like(last))


class LifetimeShapeHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.lifetime = nn.Linear(d, 1)
        self.shape = nn.Linear(d, config.shape_bins)
        self.start = nn.Linear(d, 1)
        # Only the explicitly selected unconstrained control uses this head.
        self.free_curve = nn.Sequential(nn.Linear(d+1, d), nn.GELU(), nn.Linear(d, 1)) if not config.scheme_b else None

    def forward(self, u, v, inputs, level, query):
        c = self.config
        strict_end = inputs.get("protocol_features", {}).get("observed_end") if level == 4 else None
        observed_end = (strict_end.to(device=u.device, dtype=u.dtype) if strict_end is not None
                        else u.new_full((u.shape[0],), float(c.observed_cycles)))
        if observed_end.shape != (u.shape[0],) or not torch.isfinite(observed_end).all() or (observed_end < 1).any() or (observed_end > c.observed_cycles).any():
            raise ValueError("observation endpoints must be one valid prefix length per sample")
        remaining = c.remaining_min + c.remaining_scale * F.softplus(self.lifetime(u.mean(1)).squeeze(-1))
        state = v.mean(1)
        endpoint = 0.0 if c.protocol == "batterymformer" else c.eta
        estimate = endpoint + c.start_margin + F.softplus(self.start(state).squeeze(-1))
        start, observed = estimate, torch.zeros_like(estimate, dtype=torch.bool)
        if level == 4 and c.information_ablation is None and c.observed_start and "protocol_features" in inputs:
            features = inputs["protocol_features"]
            soh_valid = features.get("soh_mask", torch.isfinite(features["soh_input"]))[..., 0]
            index = _last_observed_slot(soh_valid)[:, None]
            value = features["soh_input"][..., 0].gather(1, index)[:, 0]
            mask = soh_valid.gather(1, index)[:, 0]
        elif level == 3 and c.use_soh and c.observed_start and "soh" in inputs:
            soh_valid = inputs["soh_mask"][..., 0]
            index = _last_observed_slot(soh_valid)[:, None]
            value = inputs["soh"][..., 0].gather(1, index)[:, 0]
            mask = soh_valid.gather(1, index)[:, 0]
        else:
            value, mask = None, None
        if value is not None:
            observed = mask & torch.isfinite(value)
            if strict_end is None and (observed & (value <= endpoint)).any():
                raise ValueError("observed SOH at the observation end has already reached EOL")
            start = torch.where(observed, value, estimate)
        total_life = observed_end + remaining
        total_life_scaled = (total_life - c.eol_mean) / c.eol_std
        # A scalar BatteryLife run routes only the lifetime expert.  Keep a
        # finite, detached curve placeholder for legacy consumers that inspect
        # the output dictionary, while excluding it from the loss and metrics.
        if c.prediction_task == "scalar":
            future = query.new_zeros((u.shape[0], query.shape[-1] if query.ndim else 1))
            logits, weights = None, None
        else:
            logits = self.shape(state)
            decode_eta = 1e-3 if c.protocol == "batterymformer" else c.eta
            decode_start = torch.clamp(start, min=decode_eta + c.start_margin)
            q = query[None].expand(u.shape[0], -1) if query.ndim == 1 else query
            decode_query = torch.maximum(q, observed_end[:, None])
            # protect_life_head: the decode consumes the lifetime/starting values
            # detached, so the curve loss cannot backprop into the heads that the
            # life label owns; the curve pathway only shapes encoder + shape weights.
            decode_remaining = remaining.detach() if c.protect_life_head else remaining
            decode_start_used = decode_start.detach() if c.protect_life_head else decode_start
            future, weights = decode(decode_remaining, decode_start_used, logits, decode_query,
                                     observed_cycles=observed_end,
                                     eta=decode_eta, floor=c.shape_floor)
            if self.free_curve is not None:
                q = query[None].expand(u.shape[0], -1) if query.ndim == 1 else query
                # Fixed scale: no true-life or query-maximum normalization.
                coordinate = (q - observed_end[:, None]) / c.remaining_scale
                future = self.free_curve(torch.cat([state[:, None].expand(-1, q.shape[1], -1), coordinate[..., None]], -1)).squeeze(-1)
        return dict(total_life=total_life, remaining_life=remaining,
                    total_life_scaled=total_life_scaled,
                    future_soh=future, start_soh_used=start, estimated_start=estimate,
                    start_is_observed=observed, shape_weights=weights, shape_logits=logits,
                    primary=future if c.prediction_task == "curve" else total_life,
                    task_head=c.prediction_task)
