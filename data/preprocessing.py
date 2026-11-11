"""Physical-unit preprocessing. No nominal capacity, protocols or labels enter here."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Cycle:
    number: int
    voltage: np.ndarray
    current: np.ndarray
    capacity: float
    capacity_basis: str
    time: np.ndarray | None = None
    time_unit: str = "s"
    current_sign: int = 1
    charge_q: np.ndarray | None = None
    discharge_q: np.ndarray | None = None
    coordinate: np.ndarray | None = None
    coordinate_name: str | None = None
    soc: np.ndarray | None = None


def _runs(mask):
    indices = np.flatnonzero(mask)
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)


def resample_cycle(cycle, points=128, channels=2, current_threshold=1e-5, *, return_coverage=False):
    """Select the longest valid contiguous segment per phase; retain its original axis.

    No integration crosses a rest, timestamp reset, invalid point or phase change.
    Relative progress is a coordinate only; the third channel requires a real counter.
    """
    v, i = (np.asarray(x, dtype=float) for x in (cycle.voltage, cycle.current))
    if v.ndim != 1 or v.shape != i.shape or cycle.current_sign not in (-1, 1):
        raise ValueError("voltage/current must be aligned 1D arrays with a declared sign")
    if not cycle.capacity_basis or not np.isfinite(cycle.capacity) or cycle.capacity <= 0:
        raise ValueError("a positive capacity scalar and measurement basis are required")
    times = None
    if cycle.time is not None:
        units = {"s": 1, "ms": 1e-3, "min": 60, "h": 3600}
        if cycle.time_unit not in units:
            raise ValueError("time_unit must be s, ms, min or h")
        times = np.asarray(cycle.time, dtype=float) * units[cycle.time_unit]
    arrays = (times, cycle.charge_q, cycle.discharge_q, cycle.coordinate, cycle.soc)
    if any(x is not None and np.shape(x) != v.shape for x in arrays):
        raise ValueError("all sample arrays must have equal lengths")
    if cycle.coordinate is not None and not cycle.coordinate_name:
        raise ValueError("fallback coordinate requires an explicit name")
    curves = np.zeros((2, channels, points), np.float32)
    mask = np.zeros_like(curves, dtype=bool)
    soc = np.zeros((2, points), np.float32)
    soc_mask = np.zeros_like(soc, dtype=bool)
    provenance = []
    coverage = []
    for phase, sign in enumerate((1, -1)):
        counter = cycle.charge_q if phase == 0 else cycle.discharge_q
        valid = np.isfinite(v) & np.isfinite(i) & (sign * cycle.current_sign * i > current_threshold)
        if times is not None:
            valid &= np.isfinite(times)
        candidates = []
        for run in _runs(valid):
            # Duplicate/reset timestamps separate acquisition segments; never sort them.
            cuts = np.flatnonzero(np.diff(times[run]) <= 0) + 1 if times is not None else []
            for ix in np.split(run, cuts):
                if ix.size < 2:
                    continue
                q = None if counter is None else np.asarray(counter, float)[ix]
                recorded = q is not None and np.isfinite(q).all() and np.all(np.diff(q) >= 0) and np.ptp(q) > 1e-9
                if recorded:
                    coordinate, source = q, "recorded_capacity_progress"
                elif times is not None:
                    area = (np.abs(i[ix][1:]) + np.abs(i[ix][:-1])) * .5 * np.diff(times[ix]) / 3600
                    coordinate, source = np.r_[0, np.cumsum(area)], "integrated_capacity_progress"
                elif cycle.coordinate is not None:
                    coordinate = np.asarray(cycle.coordinate, float)[ix]
                    source = cycle.coordinate_name
                else:
                    raise ValueError("resampling needs a counter, timestamps or declared coordinate")
                if not np.isfinite(coordinate).all() or np.any(np.diff(coordinate) < 0) or np.ptp(coordinate) <= 1e-9:
                    continue
                candidates.append((ix, coordinate, source, recorded))
        if not candidates:
            provenance.append("missing")
            coverage.append(dict(valid_samples=int(valid.sum()), selected_samples=0, usable_segments=0))
            continue
        # ponytail: longest segment only; use a multi-segment representation for pulse protocols.
        ix, q, source, recorded = max(candidates, key=lambda r: len(r[0]))
        coverage.append(dict(valid_samples=int(valid.sum()), selected_samples=len(ix), usable_segments=len(candidates)))
        progress = (q - q[0]) / (q[-1] - q[0])
        progress, unique = np.unique(progress, return_index=True)
        ix = ix[unique]
        grid = np.linspace(0, 1, points)
        for variable, values in enumerate((v, i)):
            curves[phase, variable] = np.interp(grid, progress, values[ix])
            mask[phase, variable] = True
        if channels == 3 and recorded:
            # Keep the recorded within-cycle capacity curve.  The scalar
            # discharge capacity is a separate cycle-level feature; it must
            # never be broadcast back into this channel.
            curves[phase, 2] = np.interp(grid, progress, np.asarray(counter)[ix])
            mask[phase, 2] = True
        if cycle.soc is not None:
            state = np.asarray(cycle.soc, float)[ix]
            if np.isfinite(state).all() and ((state >= 0) & (state <= 1)).all():
                soc[phase] = np.interp(grid, progress, state)
                soc_mask[phase] = True
        provenance.append(source)
    result = (curves, mask, soc, soc_mask, provenance)
    return (*result, coverage) if return_coverage else result


def preprocess(cycles, config, include_soc=False, require_full_window=True):
    """Physical-unit preprocessing. No nominal capacity, protocols or labels enter here.

    ``require_full_window=False`` is the dataset-root extraction escape hatch for
    sources whose recorded life stops before the endpoint: missing slots keep
    their masks False and zero values, so absence stays explicit.
    """
    from dataclasses import replace
    s, l, d = config.observed_cycles, config.points, config.curve_channels
    values = dict(curves=np.zeros((s, 2, d, l), np.float32),
                  curve_mask=np.zeros((s, 2, d, l), bool),
                  capacity=np.zeros((s, 1), np.float32), capacity_mask=np.zeros((s, 1), bool),
                  cycle_mask=np.zeros(s, bool), soc=np.zeros((s, 2, l), np.float32),
                  soc_mask=np.zeros((s, 2, l), bool))
    seen, basis, coordinates, coverage = set(), {}, {}, {}
    for raw in cycles:
        if raw.number < 1 or raw.number != int(raw.number):
            raise ValueError("cycle numbers must be positive integers")
        if raw.number > s:
            continue
        if raw.number in seen:
            raise ValueError(f"duplicate cycle number {raw.number}")
        seen.add(raw.number)
        n = int(raw.number) - 1
        result = resample_cycle(raw if include_soc else replace(raw, soc=None), l, d, config.current_threshold, return_coverage=True)
        for key, result_value in zip(("curves", "curve_mask", "soc", "soc_mask"), result[:4]):
            values[key][n] = result_value
        values["capacity"][n, 0] = raw.capacity
        values["capacity_mask"][n, 0] = True
        values["cycle_mask"][n] = True
        basis[raw.number], coordinates[raw.number] = raw.capacity_basis, result[4]
        coverage[raw.number] = result[5]
    if require_full_window and (s not in seen or 1 not in seen):
        raise ValueError("the observation window must reach both cycle 1 and its endpoint")
    return values, {"capacity_basis": basis, "resampling_coordinates": coordinates, "phase_coverage": coverage,
                    "multiple_segments": any(p["usable_segments"] > 1 for phases in coverage.values() for p in phases)}


def _reference_cycle_features_and_mask(cycles, observed_cycles):
    """Per-cycle CE/EE features plus an explicit computability mask.

    Mirrors the reference loader's cycle-level features: CE = max discharge /
    max charge counter, EE = discharge/charge energy integral (V·|I| over
    time).  A slot is masked False when its raw ingredients are missing (no
    charge/discharge counters, no usable time axis, no discharge energy), so
    absence stays explicit instead of reading as a zero measurement.
    """
    features = np.zeros((observed_cycles, 2), np.float32)
    mask = np.zeros((observed_cycles, 2), bool)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for cycle in cycles:
        if not 1 <= cycle.number <= observed_cycles:
            continue
        slot = cycle.number - 1
        charge = np.asarray(cycle.charge_q, float) if cycle.charge_q is not None else np.array([])
        discharge = np.asarray(cycle.discharge_q, float) if cycle.discharge_q is not None else np.array([])
        if charge.size and discharge.size and np.nanmax(charge) > 0:
            features[slot, 0] = np.float32(np.nanmax(discharge) / max(np.nanmax(charge), 1e-6))
            mask[slot, 0] = True
        if cycle.time is not None and charge.size and discharge.size:
            time = np.asarray(cycle.time, float)
            voltage, current = np.asarray(cycle.voltage, float), np.asarray(cycle.current, float)
            valid = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current)
            if valid.sum() > 2:
                hours = time[valid] / (3600. if cycle.time_unit == "s" else 1.)
                power = voltage[valid] * np.abs(current[valid])
                positive = current[valid] > 0
                negative = current[valid] < 0
                e_chg = integrate(power[positive], hours[positive]) if positive.sum() > 1 else 0.
                e_dis = integrate(power[negative], hours[negative]) if negative.sum() > 1 else 0.
                if e_chg > 0:
                    features[slot, 1] = np.float32(e_dis / max(e_chg, 1e-6))
                    mask[slot, 1] = True
    return features, mask


def reference_cycle_features(cycles, observed_cycles):
    """Approximate the reference loader's per-cycle CE/EE features from raw traces."""
    return _reference_cycle_features_and_mask(cycles, observed_cycles)[0]
