"""Retraining controls vs inference information switching are distinct operations."""
from dataclasses import replace
import numpy as np
import torch
from evaluation.metrics import report, crossing_from_curve
from data.protocol_extractors.common import source_eta
from training.trainer import map_tensors


def _domain_eta(domains, reference):
    """Per-record eta of the benchmark's SOH normalization.

    The BatteryMFormer cache stores SOH as ``[SOH - eta] / [1 - eta]`` with the
    domain's official EOL threshold (0.9 for CALB, 0.8 for every other domain,
    per the reference ``data_loader_soh_optimized``); evaluation must invert
    exactly that mapping before reporting physical SOH units.
    """
    return torch.tensor([source_eta(str(d)) for d in domains],
                        device=reference.device, dtype=reference.dtype)


def configurations(config):
    return {
        "full": config,
        "plain_two_heads": replace(config, use_moe=False, use_interaction=False, scheme_b=False),
        "plain_two_heads_scheme_b": replace(config, use_moe=False, use_interaction=False),
        "moe_scheme_b": replace(config, use_interaction=False),
        "no_tsfm": replace(config, use_tsfm=False, levels=(1,)),
        "level1": replace(config, levels=(1,)),
        "level2": replace(config, levels=(2,)),
        "level3": replace(config, levels=(3,)),
        "soc_only": replace(config, levels=(3,), use_soh=False),
        "soh_only": replace(config, levels=(3,), use_soc=False),
        "estimated_start": replace(config, observed_start=False),
        "compressed8": replace(config, compressed_dim=8),
        "compressed32": replace(config, compressed_dim=32),
        "recorded_capacity_curve": replace(config, curve_channels=3),
    }


MAIN_TABLE = (
    "CPMLP (information matched)", "CPTransformer (information matched)",
    "BatteryMFormer-NoProtocol (information matched)",
    "TSFM + plain two heads", "TSFM + plain two heads + Scheme B",
    "TSFM + grouped MoE + Scheme B", "full BatteryAdapter",
)


@torch.no_grad()
def evaluate(model, loader, level, alpha=.15):
    model.eval()
    device = next(model.parameters()).device
    predictions, targets, domains, sources = [], [], [], []
    per_battery_curve_mae = []
    coverage = dict(batteries=0, soc_available=0, soh_available=0, observed_start_used=0)
    crossing_gaps = []
    for batch in loader:
        batch = map_tensors(batch, lambda t: t.to(device))
        expected = model.query_grid(device)[None].expand_as(batch["targets"]["query_cycles"])
        if not torch.equal(expected, batch["targets"]["query_cycles"]):
            raise ValueError("evaluation labels must use the configured query grid")
        prediction = model(batch["inputs"], level)
        predicted_curve = prediction["future_soh"]
        target_curve = batch["targets"]["future_soh"]
        if model.config.protocol == "batterymformer":
            # The cache stores SOH as [SOH-eta]/[1-eta] with the benchmark's
            # per-domain EOL threshold (CALB 0.9, others 0.8); report all
            # metrics and diagnostics in physical SOH units.
            etas = _domain_eta(batch["domain"], predicted_curve)
            scale = 1.0 - etas.unsqueeze(-1)
            predicted_curve = predicted_curve * scale + etas.unsqueeze(-1)
            target_curve = target_curve * scale + etas.unsqueeze(-1)
        if model.config.prediction_task in ("curve", "joint") and model.config.scheme_b:
            queries = torch.cat([expected[:, :1]-1, expected], -1)
            start = prediction["start_soh_used"]
            etas_per_record = etas if model.config.protocol == "batterymformer" \
                else torch.full((start.shape[0],), float(model.config.eta), device=start.device, dtype=start.dtype)
            if model.config.protocol == "batterymformer":
                start = start * (1.0 - etas_per_record) + etas_per_record
            curves = torch.cat([start[:, None], predicted_curve], -1)
            for eta_value in sorted({float(e) for e in etas_per_record.tolist()}):
                members = [i for i, e in enumerate(etas_per_record.tolist()) if float(e) == eta_value]
                crossing = crossing_from_curve(queries[members].cpu().numpy(),
                                               curves[members].cpu().numpy(), eta_value)
                valid = np.isfinite(crossing)
                reference_life = prediction["total_life"].cpu().numpy()[members]
                crossing_gaps.extend(np.abs(crossing[valid]-reference_life[valid]).tolist())
        coverage["batteries"] += len(batch["cell_id"])
        coverage["observed_start_used"] += int(prediction["start_is_observed"].sum())
        for field in ("soc", "soh"):
            mask = batch["inputs"].get(field+"_mask")
            if level == 3 and getattr(model.config, "use_"+field) and mask is not None:
                coverage[field+"_available"] += int(mask.flatten(1).any(-1).sum())
        # Metadata availability is a data property, reported at every inference
        # level so empty whitelist fields (chemistry/format) stay visible.
        for name, mask in (batch["inputs"].get("metadata_mask") or {}).items():
            per_battery = mask.flatten(1).any(-1) if mask.ndim > 1 else mask
            coverage.setdefault("metadata_available", {})[name] = \
                coverage.get("metadata_available", {}).get(name, 0) + int(per_battery.sum())
        predictions.append({"total_life": prediction["total_life"].cpu(), "future_soh": predicted_curve.cpu()})
        target_mask = batch["targets"]["trajectory_mask"] if "trajectory_mask" in batch["targets"] else batch["targets"]["future_mask"]
        if model.config.prediction_task in ("curve", "joint"):
            errors = (predicted_curve - target_curve).abs()
            for i, cell_id in enumerate(batch["cell_id"]):
                valid = target_mask[i] & torch.isfinite(target_curve[i])
                if valid.any():
                    per_battery_curve_mae.append({"domain": str(batch["domain"][i]),
                                                  "cell_id": str(cell_id),
                                                  "n": int(valid.sum()),
                                                  "mae": float(errors[i, valid].mean())})
        targets.append({"total_life": batch["targets"]["total_life"].cpu(), "future_soh": target_curve.cpu(),
                        "future_mask": target_mask.cpu()})
        domains.extend(batch["domain"]); sources.extend(batch["eol_source"])
    if not predictions:
        raise ValueError("empty evaluation split")
    collect = lambda rows: {k: torch.cat([r[k] for r in rows]).numpy() for k in rows[0]}
    return {**report(collect(predictions), collect(targets), domains, sources, alpha,
                      primary_task=model.config.prediction_task),
            "primary_task": model.config.prediction_task, "input_protocol": model.config.protocol,
            "input_coverage": coverage,
            "per_battery_curve_mae": per_battery_curve_mae,
            "sampled_crossing_diagnostic": {"evaluated": len(crossing_gaps), "max_gap": max(crossing_gaps, default=None),
                                             "meaning": "linear grid approximation; not a second accuracy metric"}}


def match_input_encoder(reference, candidate):
    """Freeze the same early input encoder for task-head controls."""
    for name in ("compressor", "backbone", "input_adapter", "bridge"):
        src, dest = getattr(reference, name), getattr(candidate, name)
        if src is not None and dest is not None:
            dest.load_state_dict(src.state_dict(), strict=True)
            dest.requires_grad_(False)
