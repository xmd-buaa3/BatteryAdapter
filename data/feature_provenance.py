"""A single whitelist, applied before encoding, scaling or caching."""
import numpy as np

NUMERIC = ("temperature", "charge_rate_summary", "discharge_rate_summary", "nominal_capacity")
CATEGORICAL = ("battery_chemistry", "battery_format")
ALLOWED = NUMERIC + CATEGORICAL
# 2026-10-02 information-registration completion: the Level-2 whitelist can be
# extended to the full BatteryMFormer registered aging-condition tuple
# (positive/negative electrode, electrolyte, package structure, manufacturer,
# formation/charge/discharge protocol, operating temperature, SOC window).
# The extension is opt-in through ``metadata_profile="bmf"`` so every existing
# legacy run keeps exactly these six fields and the 43-channel layout.
BMF_NUMERIC = ("ambient_temperature", "charge_rate_max", "charge_cutoff_voltage",
               "has_cv_stage", "discharge_rate", "discharge_cutoff_voltage",
               "n_charge_stages", "soc_range_lo", "soc_range_hi")
BMF_CATEGORICAL = ("battery_anode", "battery_electrolyte", "battery_manufacturer",
                   "battery_formation", "charge_protocol_id", "discharge_protocol_id")
METADATA_PROFILES = ("legacy", "bmf")
STATE_METHODS = {"measured", "early_calibrated"}


def check_profile(profile):
    if profile not in METADATA_PROFILES:
        raise ValueError(f"metadata_profile must be one of {METADATA_PROFILES}, got {profile!r}")


def numeric_fields(profile):
    """Measured plus registered numeric metadata names for the active profile."""
    check_profile(profile)
    return NUMERIC + BMF_NUMERIC if profile == "bmf" else NUMERIC


def categorical_fields(profile):
    """Measured plus registered categorical metadata names for the active profile."""
    check_profile(profile)
    return CATEGORICAL + BMF_CATEGORICAL if profile == "bmf" else CATEGORICAL


def metadata_whitelists(profile):
    return numeric_fields(profile), categorical_fields(profile)


def allowed_metadata_fields(profile):
    return numeric_fields(profile) + categorical_fields(profile)


def audit_metadata(metadata, allowed=None):
    extra = set(metadata) - set(ALLOWED if allowed is None else allowed)
    if extra:
        raise ValueError(f"metadata outside whitelist: {sorted(extra)}")
    return metadata


def audit_state(provenance, field, window):
    source = provenance.get(field, {})
    if source.get("method") not in STATE_METHODS:
        raise ValueError(f"{field} needs measured or early_calibrated provenance")
    if not source.get("reference") or not 0 < source.get("observed_through", 0) <= window:
        raise ValueError(f"{field} needs a reference and an early-only observation limit")
    if source.get("uses_protocol", False) or source.get("uses_future", False):
        raise ValueError(f"{field} cannot use protocol endpoints or future observations")


def select_inputs(inputs, level):
    if level not in (1, 2, 3, 4, 5):
        raise ValueError("information_level must be 1 through 5")
    allowed = {"curves", "curve_mask", "capacity", "capacity_mask", "cycle_mask"}
    if level in (2, 3):
        allowed |= {"metadata", "metadata_mask"}
    if level == 3:
        allowed |= {"soc", "soc_mask", "soh", "soh_mask", "cycle_features", "cycle_features_mask"}
    if level in (4, 5):
        allowed.add("protocol_features")
        if level == 4 and "protocol_features" not in inputs:
            raise ValueError("Level 4 requires the extracted BatteryMFormer protocol fields")
    # Target fields are never accepted even if a caller bypasses the dataset.
    if set(inputs) & {"targets", "total_life", "future_soh", "future_mask", "s100_label"}:
        raise ValueError("supervision must be passed to the loss, separately from inputs")
    known = allowed | {"metadata", "metadata_mask", "soc", "soc_mask", "soh", "soh_mask",
                       "cycle_features", "cycle_features_mask"}
    if set(inputs) - known:
        raise ValueError(f"unknown model input fields: {sorted(set(inputs) - known)}")
    return {k: v for k, v in inputs.items() if k in allowed}


def finite_values(value, shape):
    a = np.broadcast_to(np.asarray(value, dtype=np.float32), shape).copy()
    return np.where(np.isfinite(a), a, 0), np.isfinite(a)
