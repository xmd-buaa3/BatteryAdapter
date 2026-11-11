import torch
from torch.nn import functional as F


def _mean_valid(error, valid):
    return torch.where(valid, error, 0).sum() / valid.sum().clamp_min(1)


def _life_term(prediction, targets, config):
    """Total-life loss in the scale selected by ``config.life_loss``.

    ``scaled_huber`` is the BatteryLife protocol loss on the training-split
    z-scored life; ``log_huber`` compares log lifetimes so small errors are
    relative, matching MAPE-style benchmarks.
    """
    if config.life_loss == "log_huber":
        target = targets["total_life"]
        pred = prediction["total_life"].clamp_min(1e-3)
        valid = torch.isfinite(target) & (target > 0)
        safe_target = torch.where(valid, target, pred.detach()).clamp_min(1e-3)
        error = F.huber_loss(pred.log() - safe_target.log(),
                             torch.zeros_like(pred), reduction="none")
        return _mean_valid(error, valid)
    target = targets["total_life_scaled"] if "total_life_scaled" in targets else targets["total_life"]
    pred = prediction["total_life_scaled"] if "total_life_scaled" in prediction else prediction["total_life"]
    valid = torch.isfinite(target)
    safe = torch.where(valid, target, pred.detach())
    value = _mean_valid(F.huber_loss(pred, safe, reduction="none"), valid)
    return value


def _curve_terms(prediction, targets, config):
    """Curve and start losses on the fixed query grid (BatteryMFormer labels,
    or the optional BatteryLife auxiliary SOH labels)."""
    target = targets["future_soh"]
    mask = targets["trajectory_mask"] if "trajectory_mask" in targets else targets["future_mask"]
    pred = prediction["future_soh"]
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError("curve predictions, labels and masks must use the same fixed grid")
    mask = mask.bool() & torch.isfinite(target)
    safe_target = torch.where(mask, target, pred.detach())
    normalized = (pred - safe_target) / config.sigma_soh
    if getattr(config, "curve_loss", "huber") == "mse":
        error = normalized ** 2
    else:
        error = F.huber_loss(normalized,
                             torch.zeros_like(target), reduction="none")
    counts = mask.sum(-1)
    per_battery = torch.where(mask, error, 0).sum(-1) / counts.clamp_min(1)
    curve_loss = _mean_valid(per_battery, counts > 0)
    start = targets.get("start_soh", torch.full_like(pred[:, 0], float("nan")))
    valid = torch.isfinite(start)
    estimate = prediction["estimated_start"]
    safe_start = torch.where(valid, start, estimate.detach())
    start_error = F.huber_loss((estimate-safe_start)/config.sigma_soh,
                               torch.zeros_like(start), reduction="none")
    start_loss = _mean_valid(start_error, valid)
    return curve_loss, start_loss


def losses(prediction, targets, config):
    """Compute the task selected by ``task_head``.

    ``scalar`` optimizes only the total life; ``curve`` only the trajectory
    plus its start; ``joint`` adds the life term to the curve terms so the
    BatteryLife protocol can exploit auxiliary SOH-trajectory labels.
    """
    task = config.prediction_task
    if task == "scalar":
        value = _life_term(prediction, targets, config)
        zero = value.new_zeros(())
        return dict(total=value, lifetime=value, curve=zero, start=zero)

    curve_loss, start_loss = _curve_terms(prediction, targets, config)
    if task == "joint":
        life_loss = _life_term(prediction, targets, config)
        total = (config.loss_weights[0] * life_loss + config.loss_weights[1] * curve_loss
                 + config.loss_weights[2] * start_loss)
        return dict(total=total, lifetime=life_loss, curve=curve_loss, start=start_loss)

    total = config.loss_weights[1] * curve_loss + config.loss_weights[2] * start_loss
    zero = total.new_zeros(())
    return dict(total=total, lifetime=zero, curve=curve_loss, start=start_loss)
