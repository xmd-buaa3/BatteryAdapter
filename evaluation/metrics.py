import numpy as np


def _metrics(pred, target, alpha):
    if not target.size:
        return dict(n=0, mae=None, rmse=None, mape=None, alpha_accuracy=None)
    if not np.isfinite(pred).all():
        raise ValueError("nonfinite predictions cannot be omitted from evaluation")
    error = np.abs(pred-target)
    relative = error[np.abs(target) > 1e-12]/np.abs(target[np.abs(target) > 1e-12])
    return dict(n=int(target.size), mae=float(error.mean()), rmse=float(np.sqrt((error**2).mean())),
                mape=float(relative.mean()) if relative.size else None,
                alpha_accuracy=float((relative <= alpha).mean()) if relative.size else None)


def lifetime_metrics(pred, target, alpha=.15):
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite relative-error tolerance in (0,1)")
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    if pred.shape != target.shape:
        raise ValueError("lifetime shape mismatch")
    valid = np.isfinite(target) & (target > 0)
    return _metrics(pred[valid], target[valid], alpha)


def curve_metrics(pred, target, mask, domains=None):
    pred, target, mask = np.asarray(pred, float), np.asarray(target, float), np.asarray(mask, bool)
    if pred.shape != target.shape or mask.shape != target.shape or target.ndim != 2:
        raise ValueError("curve scores need matching [B,H] predictions, labels and masks")
    if domains is not None and len(domains) != target.shape[0]:
        raise ValueError("one domain per battery required")
    rows = []
    for i in range(target.shape[0]):
        valid = mask[i] & np.isfinite(target[i])
        row = _metrics(pred[i, valid], target[i, valid], .15)
        if row["n"]:
            rows.append((i, row))
    def aggregate(selected):
        return {**{key: float(np.mean([r[key] for _, r in selected if r[key] is not None]))
                   if any(r[key] is not None for _, r in selected) else None for key in ("mae", "rmse", "mape")},
                "n_batteries": len(selected)}
    result = aggregate(rows)
    if domains is not None:
        result["by_domain"] = {domain: aggregate([(i, r) for i, r in rows if domains[i] == domain])
                               for domain in sorted(set(domains))}
    return result


def report(predictions, targets, domains, eol_sources, alpha=.15, primary_task="both"):
    alphas = [alpha] if np.isscalar(alpha) else list(alpha)
    if not alphas:
        raise ValueError("at least one benchmark alpha tolerance is required")
    groups = {}
    for source in sorted(set(eol_sources)):
        ix = np.array([s == source for s in eol_sources])
        life = None
        curve = None
        if primary_task in ("both", "scalar", "joint"):
            life = lifetime_metrics(predictions["total_life"][ix], targets["total_life"][ix], alphas[0])
            life["accuracy_by_alpha"] = {str(a): lifetime_metrics(predictions["total_life"][ix], targets["total_life"][ix], a)["alpha_accuracy"] for a in alphas}
        if primary_task in ("both", "curve", "joint"):
            target_mask = targets["trajectory_mask"] if "trajectory_mask" in targets else targets["future_mask"]
            curve = curve_metrics(predictions["future_soh"][ix], targets["future_soh"][ix],
                                  target_mask[ix], np.array(domains)[ix])
        groups[source] = dict(lifetime=life, curve=curve)
    return groups


def crossing_from_curve(query_cycles, soh_curve, eta=.8, *, mask=None, inclusive=True, interpolate=True):
    """First sampled crossing or a linearly interpolated diagnostic (adapted from dsv41).

    Missing points before a crossing make its first occurrence unknown. A crossing
    at the first grid point is left-censored for interpolation; return NaN. A
    sampled/interpolated diagnostic is not the analytic or integer Scheme B EOL.
    """
    s = np.asarray(soh_curve, float)
    if s.ndim != 2 or not 0 < eta < 1:
        raise ValueError("curve must be [B,H] and eta in (0,1)")
    q = np.asarray(query_cycles, float)
    if q.ndim == 1:
        q = np.broadcast_to(q, s.shape)
    if q.shape != s.shape or not np.isfinite(q).all() or np.any(np.diff(q, axis=-1) <= 0):
        raise ValueError("one finite strictly increasing query grid per curve is required")
    valid = np.isfinite(s)
    if mask is not None:
        if np.shape(mask) != s.shape:
            raise ValueError("crossing mask must match the curves")
        valid &= np.asarray(mask, bool)
    result = np.full(s.shape[0], np.nan)
    for i in range(s.shape[0]):
        hits = np.flatnonzero((s[i] <= eta if inclusive else s[i] < eta) & valid[i])
        if not hits.size:
            continue
        j = hits[0]
        if not valid[i, :j+1].all():
            continue
        if not interpolate:
            result[i] = q[i, j]
        elif j > 0:
            fraction = (eta-s[i, j-1])/(s[i, j]-s[i, j-1])
            result[i] = q[i, j-1] + fraction*(q[i, j]-q[i, j-1])
    return result
