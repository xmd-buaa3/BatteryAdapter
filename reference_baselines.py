"""Strict-cache input interface for the BatteryMFormer cross-domain reference."""
from pathlib import Path
from types import SimpleNamespace
import importlib
import sys

import torch

MODELS = ("BatteryMFormer",)


def load_model(name, model_config, reference_root):
    if name not in MODELS:
        raise ValueError(f"unknown reference model: {name}")
    root = Path(reference_root).resolve()
    if not (root / "models" / f"{name}.py").is_file():
        raise FileNotFoundError(f"reference model source missing: {root / 'models' / (name + '.py')}")
    sys.path.insert(0, str(root))
    module = importlib.import_module(f"models.{name}")
    if Path(module.__file__).resolve() != root / "models" / f"{name}.py":
        raise RuntimeError("another reference models package is already loaded")
    return module.Model(SimpleNamespace(**model_config))


def prepare_batch(batch, name, device):
    if name not in MODELS:
        raise ValueError(f"unknown reference model: {name}")
    inputs, targets = batch["inputs"], batch["targets"]
    curves = inputs["curves"].to(device)
    mask = inputs["curve_mask"].to(device)
    curves = torch.where(mask, curves, 0)
    curves = torch.cat((curves[:, :, 0], curves[:, :, 1]), dim=-1)
    kwargs = {"cycle_curve_data": curves, "curve_attn_mask": inputs["cycle_mask"].to(device)}
    target = targets["future_soh"].to(device)
    target_mask = targets["trajectory_mask"].to(device).bool()
    features = inputs["protocol_features"]
    soc = features["soc_input"].to(device)
    if soc.ndim == 4 and soc.shape[-2:] == (2, 150):
        soc = soc.flatten(-2)
    if soc.shape != (curves.shape[0], curves.shape[1], curves.shape[-1]):
        raise ValueError("BatteryMFormer SOC must have shape [batch, cycles, 300]")
    kwargs.update(aging_condition_embedding=features["aging_condition_embedding"].to(device),
                  soc_input=soc, soh_input=features["soh_input"].to(device),
                  cycle_level_features=features["cycle_level_features"].to(device),
                  soh_trajectory=target, trajectory_mask=target_mask,
                  life_labels=targets["total_life"].to(device))
    if not torch.isfinite(kwargs["life_labels"]).all():
        raise ValueError("BatteryMFormer requires finite life labels")
    return kwargs, target, target_mask


def loss_for_batch(model, batch, name, device, recovery_weight, memory_weight):
    kwargs, target, mask = prepare_batch(batch, name, device)
    result = model(**kwargs)
    prediction = result[0] if model.training else result
    if prediction.shape != target.shape or not mask.any():
        raise ValueError("reference prediction shape or trajectory mask is invalid")
    loss = (prediction[mask] - target[mask]).square().mean()
    if model.training:
        loss = loss + recovery_weight * result[1] + memory_weight * result[3]
    return prediction, target, mask, loss
