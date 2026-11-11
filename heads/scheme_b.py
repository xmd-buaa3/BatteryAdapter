import torch


def decode(remaining, start, logits, query_cycles, *, observed_cycles=100, eta=.8, floor=1e-3):
    """Plan §8: positive increments, piecewise linear A(u), positive tail slope."""
    b, k = logits.shape
    if k < 1 or not 0 < floor < 1 or not 0 < eta < 1:
        raise ValueError("nonempty shape bins and eta/floor in (0,1) required")
    if remaining.shape != (b,) or start.shape != (b,):
        raise ValueError("remaining and start must have shape [B]")
    if not all(torch.isfinite(x).all() for x in (remaining, start, logits, query_cycles)):
        raise ValueError("decoder inputs must be finite")
    if (remaining <= 0).any() or (start <= eta).any():
        raise ValueError("positive remaining life and a start strictly above EOL required")
    q = query_cycles.to(remaining)
    if q.ndim == 1:
        q = q[None].expand(b, -1)
    endpoint = torch.as_tensor(observed_cycles, device=remaining.device, dtype=remaining.dtype)
    if endpoint.ndim == 0:
        endpoint = endpoint.expand(b)
    if endpoint.shape != (b,) or not torch.isfinite(endpoint).all():
        raise ValueError("observation endpoints must be finite scalars or [B]")
    if q.ndim != 2 or q.shape[0] != b or (q < endpoint[:, None]).any():
        raise ValueError("query cycles must be [H] or [B,H], at or after the window endpoint")
    weights = (1 - floor) * logits.softmax(-1) + floor/k
    prefix = torch.cat([weights.new_zeros(b, 1), weights.cumsum(-1)], -1)
    u = (q - endpoint[:, None]) / remaining[:, None]
    position = (u*k).floor().long().clamp(0, k-1)
    a = prefix.gather(1, position) + (u*k - position) * weights.gather(1, position)
    a = torch.where(u >= 1, 1 + k * weights[:, -1:] * (u-1), a)
    soh = start[:, None] * torch.exp(-torch.log(start[:, None]/eta) * a)
    return soh, weights


def integer_eol(total_life, inclusive=True):
    """First integer n with SOH(n) <= eta (or < eta for strict benchmarks)."""
    return torch.ceil(total_life) if inclusive else torch.floor(total_life) + 1
