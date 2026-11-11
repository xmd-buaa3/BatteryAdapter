"""Streaming reference-style point metrics over all cell/prefix samples."""
import math
import torch

from data.protocol_extractors.bmf_strict import STRICT_CACHE_VERSION
from data.protocol_extractors.common import source_eta
from training.trainer import map_tensors


class StrictMetricFailure(ValueError):
    """The candidate's own predictions made the metric undefined.

    Distinct from protocol errors (wrong level, wrong query grid, missing
    samples): those must fail closed, while this one only disqualifies the
    candidate and lets a search record it and move on.
    """


@torch.no_grad()
def evaluate_bmf_strict(model, loader, level=4):
    if level != 4:
        raise ValueError("strict reference evaluation requires Level 4")
    training = model.training
    model.eval()
    device = next(model.parameters()).device
    abs_sum = rel_sum = 0.0
    n_mae = n_mape = 0
    seen = set()
    try:
        for original in loader:
            for domain, cell, prefix in zip(original["domain"], original["cell_id"], original["prefix"]):
                key = (str(domain), str(cell), int(prefix))
                if key in seen:
                    raise ValueError(f"duplicate strict evaluation sample: {key}")
                seen.add(key)
            batch = map_tensors(original, lambda t: t.to(device))
            query = model.query_grid(device)[None].expand_as(batch["targets"]["query_cycles"])
            if not torch.equal(query, batch["targets"]["query_cycles"]):
                raise ValueError("strict evaluation requires the registered absolute query grid")
            predicted = model(batch["inputs"], level)["future_soh"]
            target = batch["targets"]["future_soh"]
            mask = batch["targets"]["trajectory_mask"].bool()
            # Invert [SOH - eta] / [1 - eta] in float64 with the exact threshold.
            # A float32 eta makes 1-eta 0.19999999 and shifts a MAPE near 11 by
            # ~7e-6, which is enough to blur equal-score checkpoint comparison.
            eta = torch.tensor([source_eta(str(d)) for d in batch["domain"]],
                               dtype=torch.float64, device=predicted.device)[:, None]
            predicted = predicted.double() * (1 - eta) + eta
            target = target.double() * (1 - eta) + eta
            if not torch.isfinite(target[mask]).all() or not torch.isfinite(predicted[mask]).all():
                raise StrictMetricFailure("nonfinite value in strict evaluation's valid points")
            err = (predicted[mask] - target[mask]).abs()
            truth = target[mask]
            relative = truth > .01
            abs_sum += float(err.sum())
            rel_sum += float((err[relative] / truth[relative]).sum())
            n_mae += int(err.numel())
            n_mape += int(relative.sum())
    finally:
        model.train(training)
    dataset = loader.dataset
    expected = {(str(dataset.records[i].domain), str(dataset.records[i].cell_id), int(k))
                for i, k in dataset.index}
    if seen != expected:
        raise ValueError("strict evaluation did not cover every registered (cell, prefix) sample")
    if not n_mae or not n_mape:
        raise StrictMetricFailure("strict evaluation has no valid MAE/MAPE points")
    mae, mape = abs_sum / n_mae, 100 * rel_sum / n_mape
    if not math.isfinite(mae) or not math.isfinite(mape):
        raise StrictMetricFailure("nonfinite strict pooled metric")
    return {"mae": mae, "mape": mape, "mae_abs_sum": abs_sum, "mae_count": n_mae,
            "mape_relative_sum": rel_sum, "mape_count": n_mape, "sample_count": len(seen),
            "aggregation": "all-valid-points", "protocol": STRICT_CACHE_VERSION}
