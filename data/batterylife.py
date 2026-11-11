"""Explicit raw-field mapping, early-only metadata, and a reproducible data protocol."""
import hashlib
import json
import os
import pickle
from pathlib import Path
import numpy as np
from data.dataset import make_record, check_splits
from data.preprocessing import Cycle
from data.feature_provenance import allowed_metadata_fields, audit_metadata


def _reference(value, name):
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in ("main", "head", "latest") or "REPLACE" in value.upper():
        raise ValueError(f"{name} needs a concrete, nonempty reference/definition")


def _reference_aging_embedding(raw_path, dimension):
    """Load the pinned BMF prompt embedding when it is available locally."""
    embedding_path = Path(__file__).resolve().parents[1] / "reference" / "BatteryMFormer" / \
        "data_provider" / "prompt_embeddings" / "Qwen3_total.pkl"
    if not embedding_path.is_file():
        return None
    try:
        with embedding_path.open("rb") as stream:
            table = pickle.load(stream)
        value = table.get(raw_path.name, table.get(raw_path.stem)) if isinstance(table, dict) else None
        value = np.asarray(value, np.float32) if value is not None else None
        return value.reshape(1, -1) if value is not None and value.size == dimension else None
    except (OSError, ValueError, TypeError, pickle.PickleError):
        return None


def observed_rate_summary(current, nominal_capacity, threshold=1e-5):
    """Median observed |I|/rated Ah (Level 2 only); adapted from dsv41."""
    if nominal_capacity is None or not np.isfinite(nominal_capacity) or nominal_capacity <= 0:
        return np.nan
    current = np.asarray(current, float)
    observed = np.isfinite(current) & (np.abs(current) > threshold)
    return float(np.median(np.abs(current[observed])) / nominal_capacity) if observed.any() else np.nan


def early_metadata(raw, early_cycles, source, config):
    """Read explicitly selected raw fields, retaining one value per original cycle."""
    metadata = dict(audit_metadata(source.get("metadata", {}),
                                   allowed=allowed_metadata_fields(config.metadata_profile)))
    selected = source.get("metadata_from_raw", [])
    allowed = {"temperature", "charge_rate_summary", "discharge_rate_summary", "nominal_capacity"}
    if not isinstance(selected, list) or any(name not in allowed for name in selected) or len(set(selected)) != len(selected):
        raise ValueError("metadata_from_raw must list distinct temperature/rate/nominal-capacity fields")
    if set(selected) & set(metadata):
        raise ValueError("choose one declared source per metadata field")
    audit = {name: "manifest" for name in metadata}
    nominal = metadata.get("nominal_capacity", raw.get("nominal_capacity_in_Ah"))
    if any("rate" in name for name in selected) or "nominal_capacity" in selected:
        if nominal is None or np.ndim(nominal) != 0 or not np.isfinite(nominal) or nominal <= 0:
            raise ValueError("derived rates require a positive scalar nominal capacity at Level 2")
    offset, _ = _record_window(source)
    for name in selected:
        if name == "nominal_capacity":
            metadata[name] = float(nominal)
        else:
            values = np.full(config.observed_cycles, np.nan)
            for item in early_cycles:
                if name == "temperature":
                    samples = np.asarray(item.get("temperature_in_C", []), float)
                    samples = samples[np.isfinite(samples)]
                    value = float(samples.mean()) if samples.size else np.nan
                else:
                    current = np.asarray(item["current_in_A"], float) * source.get("current_sign", 1)
                    sign = 1 if name == "charge_rate_summary" else -1
                    value = observed_rate_summary(current[sign * current > config.current_threshold], nominal, config.current_threshold)
                slot = int(item["cycle_number"]) - offset - 1
                if not 0 <= slot < config.observed_cycles:
                    raise ValueError("early metadata slot outside the observation window")
                values[slot] = value
            metadata[name] = values
        audit[name] = {"reference": source["path"], "observed_through": config.observed_cycles,
                       "method": "early_raw_measurements", "nominal_capacity_Ah": float(nominal) if "rate" in name else None}
    return metadata, audit


def _record_window(source):
    """Declared per-record window contract: rebase offset and strict/positional policy.

    ``cycle_number_offset`` rebases raw cycle numbers whose first recorded life cycle is
    not 1 (the BatteryMFormer positional convention the multicurve extractor applies).
    ``window_policy: "positional"`` relaxes the cycle-1-and-endpoint requirement for
    sources with short recorded lives, mirroring the unified-eval positional semantics;
    missing slots stay explicitly masked.
    """
    offset = source.get("cycle_number_offset", 0)
    policy = source.get("window_policy", "strict")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("cycle_number_offset must be a nonnegative integer")
    if policy not in ("strict", "positional"):
        raise ValueError("window_policy must be strict or positional")
    return offset, policy


def _read_protocol(path, config):
    """Parse and validate the declared data protocol, independent of per-record loading."""
    path = Path(path).resolve()
    content = path.read_bytes()
    protocol = json.loads(content)
    required = ("schema_version", "dataset_version", "snapshot", "split_reference", "capacity_definition", "soh_definition",
                "smoothing", "eol_rule", "pretraining_overlap", "eta", "soh_scale", "input_layout_version",
                "horizon", "observed_cycles", "segment_policy", "alpha_tolerances", "records")
    if not isinstance(protocol, dict) or any(key not in protocol for key in required):
        raise ValueError(f"manifest requires {required}")
    if type(protocol["schema_version"]) is not int or protocol["schema_version"] != 3:
        raise ValueError("manifest schema_version must be 3; migrate the declared early-fusion data protocol")
    for name in ("dataset_version", "snapshot", "split_reference", "capacity_definition", "soh_definition", "smoothing", "pretraining_overlap"):
        _reference(protocol[name], name)
    for name in ("horizon", "observed_cycles"):
        if type(protocol[name]) is not int or protocol[name] != getattr(config, name):
            raise ValueError(f"manifest {name} differs from experiment configuration")
    if isinstance(protocol["eta"], bool) or protocol["eta"] != config.eta or protocol["eol_rule"] != "<=":
        raise ValueError("manifest must match the configured threshold and inclusive integer EOL rule")
    if protocol["soh_scale"] != config.soh_scale or int(protocol["input_layout_version"]) != config.input_layout_version:
        raise ValueError("manifest SOH scale/layout version differs from the early-fusion configuration")
    profile = protocol.get("metadata_profile", "legacy")
    if profile != config.metadata_profile:
        raise ValueError(f"manifest metadata_profile {profile!r} differs from config "
                         f"{config.metadata_profile!r}; use the manifest built for this profile")
    if protocol["segment_policy"] not in ("single", "longest"):
        raise ValueError("segment_policy must be single (exclude multi-segment phases) or longest (report discarded segments)")
    alphas = protocol["alpha_tolerances"]
    if not isinstance(alphas, list) or not alphas or any(isinstance(a, bool) or not isinstance(a, (int, float)) or not 0 < a < 1 for a in alphas) or len(set(alphas)) != len(alphas):
        raise ValueError("alpha_tolerances must list distinct tolerances in (0,1)")
    tolerance = protocol.get("eol_convention_tolerance", 0)
    if isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0:
        raise ValueError("eol_convention_tolerance must be a nonnegative integer")
    if not isinstance(protocol["records"], list) or not protocol["records"]:
        raise ValueError("manifest records must be a nonempty list")
    return path, content, protocol


def _unique_sources(path, protocol):
    """Yield (source, resolved raw path) while enforcing split and identity uniqueness."""
    identities, paths = set(), set()
    for source in protocol["records"]:
        if source["split"] not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        for name in ("cell_id", "domain"):
            _reference(source[name], name)
        identity = (source["domain"], source["cell_id"])
        raw_path = (path.parent / source["path"]).resolve()
        path_key = os.path.normcase(str(raw_path))
        if identity in identities or path_key in paths:
            raise ValueError("a battery identity or raw file appears more than once in the declared splits")
        identities.add(identity); paths.add(path_key)
        yield source, raw_path


def _load_labels(path, source, config):
    """Label series stay live files: they are small and audited by SHA at load time."""
    labels = source["labels"]
    _reference(labels["definition"], "label definition")
    if (labels.get("eta", config.eta) != config.eta or labels.get("eol_rule", "<=") != "<="
            or labels.get("soh_scale", config.soh_scale) != config.soh_scale):
        raise ValueError("per-record labels must use the experiment threshold and EOL convention")
    declared_source_eta = source.get("provenance", {}).get("soh", {}).get("source_eta")
    if declared_source_eta is not None and labels.get("source_eta") != declared_source_eta:
        raise ValueError("label and early-SOH source thresholds differ")
    label_path = (path.parent / labels["path"]).resolve()
    with np.load(label_path, allow_pickle=False) as arrays:
        return arrays["cycles"].copy(), arrays["soh"].copy(), label_path


def _load_early_soh(path, source, label_path, config):
    if (path.parent / source["early_soh_path"]).resolve() == label_path:
        raise ValueError("early SOH must be prepared separately from benchmark target smoothing")
    with np.load(path.parent / source["early_soh_path"], allow_pickle=False) as arrays:
        if not np.array_equal(arrays["cycles"], np.arange(1, config.observed_cycles + 1)):
            raise ValueError("early SOH cycles must cover the original observation window")
        return arrays["soh"].copy()


def load_manifest(path, config):
    path, content, protocol = _read_protocol(path, config)
    splits, exclusions = {name: [] for name in ("train", "val", "test")}, []
    # Always materialize the full cohort schema.  BatteryDataset applies the
    # requested level later; this keeps category vocabularies and early-fusion
    # channel layouts identical across independently trained arms.
    # Reference experiments (levels 4/5) share the same fully materialized
    # cohort; level 5 simply ignores the state fields at selection time.
    max_level = 4 if max(config.levels) >= 4 else 3
    for source, raw_path in _unique_sources(path, protocol):
        # BatteryLife pickle is an explicitly selected trusted local dataset format.
        with raw_path.open("rb") as stream:
            raw = pickle.load(stream)
        provenance = source.get("provenance", {})
        if provenance.get("soh", {}).get("scale", config.soh_scale) != config.soh_scale:
            raise ValueError("early SOH provenance scale differs from the experiment scale")
        mapping = source["capacity"]
        _reference(mapping.get("basis"), "capacity basis")
        if ("field" in mapping) == ("values" in mapping):
            raise ValueError("declare exactly one scalar capacity field or per-cycle values mapping")
        offset, policy = _record_window(source)
        cycles, early_items, numbers = [], [], set()
        for item in raw["cycle_data"]:
            if not isinstance(item, dict):
                continue
            number = item["cycle_number"]
            if isinstance(number, bool) or not np.isfinite(number) or number != int(number) or number < 1:
                raise ValueError("raw cycle numbers must be positive integers")
            number = int(number) - offset
            if number < 1:
                raise ValueError(f"cycle_number_offset {offset} rebases a recorded cycle below 1")
            if number > config.observed_cycles:
                continue
            if number in numbers:
                raise ValueError("duplicate original cycle number")
            numbers.add(number); early_items.append(item)
            capacity = mapping["values"].get(str(number)) if "values" in mapping else item.get(mapping["field"])
            if capacity is None:
                continue
            if np.ndim(capacity) != 0:
                raise ValueError("per-cycle capacity must be a scalar, not a counter curve")
            cycles.append(Cycle(number=number, voltage=np.asarray(item["voltage_in_V"]),
                                current=np.asarray(item["current_in_A"]), time=item.get("time_in_s"),
                                capacity=float(capacity), capacity_basis=mapping["basis"], current_sign=source.get("current_sign", 1),
                                charge_q=item.get("charge_capacity_in_Ah"), discharge_q=item.get("discharge_capacity_in_Ah"),
                                soc=item.get(source.get("soc_field", "__no_soc__")) if max_level in (3, 4) else None))
        label_cycles, label_soh, label_path = _load_labels(path, source, config)
        early = _load_early_soh(path, source, label_path, config) if "early_soh_path" in source and max_level in (3, 4) else None
        metadata, metadata_audit = early_metadata(raw, early_items, source, config) if max_level >= 2 else ({}, {})
        record = make_record(source["cell_id"], source["domain"], cycles, config,
                             metadata=metadata, early_soh=early, provenance=provenance,
                             label_cycles=label_cycles, label_soh=label_soh, total_life=source["labels"].get("total_life"),
                             eol_source=source["labels"].get("eol_source", "measured"), eol_evidence=source["labels"].get("eol_evidence"),
                             label_definition=source["labels"]["definition"], level=max_level,
                             eol_tolerance=int(protocol.get("eol_convention_tolerance", 0)),
                             require_full_window=policy == "strict",
                             reference_aging_embedding=_reference_aging_embedding(raw_path, config.reference_embedding_dim)
                             if max_level >= 4 else None)
        record.audit.update(metadata_sources=metadata_audit, raw_path=str(raw_path),
                            label_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest())
        if record.audit["multiple_segments"] and protocol["segment_policy"] == "single":
            exclusions.append(dict(cell_id=source["cell_id"], domain=source["domain"], split=source["split"], reason="multiple_phase_segments"))
        else:
            splits[source["split"]].append(record)
    check_splits(splits)
    protocol["manifest_sha256"] = hashlib.sha256(content).hexdigest()
    protocol["exclusions"] = exclusions
    return splits, protocol
