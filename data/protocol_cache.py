"""Read-only cache contract for the BatteryMFormer/BatteryLife input protocols.

The extractor is the only code allowed to talk to a reference data loader.  The
training side reads these files and never imports either reference project.
"""
from __future__ import annotations

import json
from pathlib import Path
import hashlib
import numpy as np
import reference_inputs

FORMAT_VERSION = 2
SHARED_FORMAT_VERSION = 3
SHARED_CACHE_VERSION = "protocol-cache-v3"
# BatteryLife items extended with auxiliary SOH-trajectory labels (same schema,
# split-independent like v3); the label grid must match the run horizon.
AUX_CACHE_VERSION = "protocol-cache-v4-blaux"
_SHARED_CACHE_VERSIONS = (SHARED_CACHE_VERSION, AUX_CACHE_VERSION)
RAW_CURVE_SHAPE = (3, 300)
PHASE_SHAPE = (2, 3, 150)
# Per-cycle cache fields that must move together when a run reads a shortened
# prefix of the extracted window (e.g. the L5-PBT 20-cycle alignment).  Label
# fields (future_soh/total_life/start_soh) are never sliced here.
_PER_CYCLE_FIELDS = {
    "batterylife": ("curves", "curve_mask", "cycle_mask", "capacity", "capacity_mask"),
    "batterymformer": ("curves", "curve_mask", "cycle_mask", "capacity", "capacity_mask",
                       "soc_input", "soc_mask", "soh_input", "soh_mask",
                       "cycle_level_features", "cycle_level_features_mask"),
}


def _scalar(data, name):
    return data[name].item() if data[name].shape == () else data[name].item()


def _check(item, protocol, *, split_independent=False):
    required = {"curves", "curve_mask", "capacity", "capacity_mask", "cycle_mask",
                "future_soh", "future_mask", "trajectory_mask", "total_life", "cell_id", "dataset",
                "source_curve_layout", "normalization_version", "cache_version", "raw_source"}
    if not split_independent:
        required.add("split")
    missing = required - set(item)
    if missing:
        raise ValueError(f"cache item missing fields: {sorted(missing)}")
    if split_independent and {"split", "source_loader", "split_tag", "split_seed", "train_seed", "itr"} & set(item):
        raise ValueError("shared protocol cache must not contain split-specific metadata")
    if tuple(item["curves"].shape[-2:]) != RAW_CURVE_SHAPE:
        raise ValueError(f"cached curves must use raw [S,3,300] layout, got {item['curves'].shape}")
    if item["curve_mask"].shape != item["curves"].shape or item["curve_mask"].dtype != np.bool_:
        raise ValueError("curve_mask must match curves and be bool")
    if item["curves"].dtype.kind not in "fc" or not np.isfinite(item["curves"][item["curve_mask"]]).all():
        raise ValueError("curves must be finite floating point where observed")
    if item["capacity"].shape != item["capacity_mask"].shape or item["capacity_mask"].dtype != np.bool_ or np.any(item["capacity_mask"]):
        raise ValueError("capacity must be zero masked for protocol inputs")
    if item["future_mask"].dtype != np.bool_ or item["trajectory_mask"].dtype != np.bool_ or item["future_mask"].shape != item["future_soh"].shape or item["trajectory_mask"].shape != item["future_soh"].shape:
        raise ValueError("future_soh masks must be boolean and shape-matched")
    if not np.array_equal(item["future_mask"], item["trajectory_mask"]):
        raise ValueError("future_mask and trajectory_mask must agree")
    if not np.isfinite(item["future_soh"][item["future_mask"]]).all():
        raise ValueError("future_soh must be finite where observed")
    if item["cycle_mask"].shape != item["curves"].shape[:-2] or item["cycle_mask"].dtype != np.bool_:
        raise ValueError("cycle_mask must be boolean with shape [S]")
    if str(np.asarray(item["source_curve_layout"]).item()) != "channels_first_3x300":
        raise ValueError("unknown source_curve_layout")
    for name in ("normalization_version", "cache_version", "raw_source"):
        if not str(np.asarray(item[name]).item()).strip():
            raise ValueError(f"{name} must be nonempty")
    if protocol == "batterymformer":
        for name in ("soc_input", "soh_input", "cycle_level_features", "cycle_level_features_mask",
                     "aging_condition_embedding", "aging_condition_mask", "protocol_feature_mask"):
            if name not in item:
                raise ValueError(f"BatteryMFormer cache missing {name}")
        if np.asarray(item["protocol_feature_mask"]).dtype != np.bool_:
            raise ValueError("protocol_feature_mask must be boolean")
        if item["soc_input"].shape != (item["curves"].shape[0], item["curves"].shape[-1]):
            raise ValueError("soc_input must preserve raw [S,300] layout")
        if item.get("soc_mask", np.zeros_like(item["soc_input"], bool)).shape != item["soc_input"].shape or item["soc_mask"].dtype != np.bool_:
            raise ValueError("soc_mask must match soc_input")
        if item["soh_input"].shape != (item["curves"].shape[0], 1):
            raise ValueError("soh_input must have shape [S,1]")
        if item.get("soh_mask", np.zeros_like(item["soh_input"], bool)).shape != item["soh_input"].shape:
            raise ValueError("soh_mask must match soh_input")
        if item["cycle_level_features"].shape != (item["curves"].shape[0], 2):
            raise ValueError("cycle_level_features must have shape [S,2]")
        if item["cycle_level_features_mask"].shape != item["cycle_level_features"].shape or item["cycle_level_features_mask"].dtype != np.bool_:
            raise ValueError("cycle_level_features_mask must match CE/EE fields")
        if item["aging_condition_embedding"].shape != (1, 1024) or item["aging_condition_mask"].shape != (1, 1024) or item["aging_condition_mask"].dtype != np.bool_:
            raise ValueError("aging_condition_embedding must have shape [1,1024]")
        if item["future_soh"].shape[-1] != 5000:
            raise ValueError("BatteryMFormer future_soh must use the absolute 1..5000 grid")
        if item["trajectory_mask"].shape != item["future_soh"].shape or item["trajectory_mask"].dtype != np.bool_:
            raise ValueError("trajectory_mask must match future_soh")
    elif protocol == "batterylife":
        if np.asarray(item["total_life"]).shape != () or not np.isfinite(float(item["total_life"])):
            raise ValueError("BatteryLife cache requires a finite total_life label")
        if bool(np.asarray(item.get("start_soh_mask", np.isfinite(item["start_soh"]), ), bool).any()) \
                and not np.isfinite(np.asarray(item["start_soh"], np.float32)).all():
            raise ValueError("BatteryLife start_soh must be finite where start_soh_mask is set")


def save_protocol_item(path, item, *, protocol, split, source_loader, split_tag,
                       split_seed, train_seed, itr=1, split_independent=False):
    """Write one already-extracted loader sample; no transforms are performed here."""
    item = dict(item)
    if split_independent and protocol != "batterylife":
        raise ValueError("only BatteryLife supports split-independent cache items")
    curves = np.asarray(item["curves"])
    if curves.ndim == 4 and tuple(curves.shape[-3:]) == PHASE_SHAPE:
        item["curves"] = reference_inputs.phase_to_sequence(curves)
        if "curve_mask" in item:
            item["curve_mask"] = reference_inputs.phase_to_sequence(curves, item["curve_mask"])[1]
    soc = item.get("soc_input")
    if soc is not None and np.asarray(soc).ndim >= 3 and tuple(np.asarray(soc).shape[-2:]) == (2, 150):
        item["soc_input"] = np.concatenate((np.asarray(soc)[..., 0, :], np.asarray(soc)[..., 1, :]), axis=-1)
        if "soc_mask" in item:
            sm = np.asarray(item["soc_mask"])
            item["soc_mask"] = np.concatenate((sm[..., 0, :], sm[..., 1, :]), axis=-1)
    if split_independent:
        item.update(format_version=np.asarray(SHARED_FORMAT_VERSION, np.int64),
                    protocol=np.asarray(protocol),
                    source_curve_layout=np.asarray("channels_first_3x300"),
                    normalization_version=np.asarray(item.get("normalization_version", "batterylife-v1")),
                    cache_version=np.asarray(item.get("cache_version", SHARED_CACHE_VERSION)),
                    raw_source=np.asarray(item.get("raw_source", source_loader)))
        for name in ("split", "source_loader", "split_tag", "split_seed", "train_seed", "itr"):
            item.pop(name, None)
    else:
        item.update(format_version=np.asarray(FORMAT_VERSION, np.int64), protocol=np.asarray(protocol),
                    split=np.asarray(split), source_loader=np.asarray(source_loader),
                    split_tag=np.asarray(split_tag), split_seed=np.asarray(split_seed, np.int64),
                    train_seed=np.asarray(train_seed, np.int64), itr=np.asarray(itr, np.int64),
                    source_curve_layout=np.asarray("channels_first_3x300"),
                    normalization_version=np.asarray(item.get(
                        "normalization_version",
                        "batterymformer-soh-0.8-v1" if protocol == "batterymformer" else "batterylife-v1")),
                    cache_version=np.asarray(item.get("cache_version", f"protocol-cache-v{FORMAT_VERSION}")),
                    raw_source=np.asarray(item.get("raw_source", source_loader)))
    item.setdefault("capacity", np.zeros(item["cycle_mask"].shape + (1,), np.float32))
    item.setdefault("capacity_mask", np.zeros_like(item["capacity"], dtype=np.bool_))
    item.setdefault("total_life", np.asarray(np.nan, np.float32))
    item.setdefault("start_soh", np.asarray(np.nan, np.float32))
    item.setdefault("start_soh_mask", np.asarray(np.isfinite(item["start_soh"]), bool))
    item.setdefault("future_soh", np.zeros((5000,), np.float32))
    item.setdefault("future_mask", np.zeros((5000,), np.bool_))
    item.setdefault("trajectory_mask", item["future_mask"])
    item.setdefault("protocol_feature_mask", np.asarray(True, np.bool_))
    item.setdefault("soc_mask", np.zeros_like(item["soc_input"], dtype=np.bool_) if "soc_input" in item else np.asarray(False, np.bool_))
    item.setdefault("soh_mask", np.isfinite(item.get("soh_input", np.asarray(np.nan))))
    if protocol == "batterymformer":
        item.setdefault("cycle_level_features_mask", np.isfinite(item["cycle_level_features"]))
        item.setdefault("aging_condition_mask", np.isfinite(item["aging_condition_embedding"]))
    _check(item, protocol, split_independent=split_independent)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **item)
    tmp.replace(path)


def apply_observed_truncation(item, observed_used):
    """Mask every per-cycle input field beyond ``observed_used`` (in place).

    The storage window — and therefore the array shapes and the absolute
    1..5000 query grid — stays at ``observed_cycles``; only the masks shrink.
    Every downstream consumer gates by mask (compressor and scaler zero masked
    values, the early-fusion packer zeroes masked channels), so the model sees
    exactly the first ``observed_used`` cycles while the labels and evaluation
    interval remain the common 101..5000 future grid.  The start head anchors
    at the last observed slot, not at the storage end.
    """
    s = int(item["cycle_mask"].shape[0])
    if not 1 <= observed_used <= s:
        raise ValueError(f"observed_used {observed_used} outside 1..{s}")
    if not bool(np.asarray(item["cycle_mask"])[:observed_used].all()):
        raise ValueError(
            f"observed_used={observed_used} requires every cell to observe cycles 1..{observed_used}")
    item["cycle_mask"][observed_used:] = False
    item["curve_mask"][observed_used:] = False
    if "capacity_mask" in item:
        item["capacity_mask"][observed_used:] = False
    for name in ("soc_mask", "soh_mask", "cycle_level_features_mask"):
        if name in item:
            item[name][observed_used:] = False


def _read(path, expected_protocol, expected_split=None):
    with np.load(path, allow_pickle=False) as data:
        format_version = int(data["format_version"])
        if format_version not in (FORMAT_VERSION, SHARED_FORMAT_VERSION):
            raise ValueError(f"stale protocol cache: {path}")
        split_independent = format_version == SHARED_FORMAT_VERSION
        protocol = str(data["protocol"])
        if protocol != expected_protocol:
            raise ValueError(f"{path}: protocol {protocol!r} != {expected_protocol!r}")
        if split_independent and protocol != "batterylife":
            raise ValueError(f"{path}: split-independent cache is only valid for BatteryLife")
        item = {key: data[key].copy() for key in data.files
                if key not in {"format_version", "protocol", "source_loader", "split_tag",
                               "split_seed", "train_seed", "itr"}}
        if expected_split is not None and split_independent:
            raise ValueError(f"{path}: split-independent cache has no embedded split")
        if expected_split is not None and str(item["split"]) != expected_split:
            raise ValueError(f"{path}: split mismatch")
        if "future_mask" not in item and "trajectory_mask" in item:
            item["future_mask"] = item["trajectory_mask"]
        _check(item, protocol, split_independent=split_independent)
        audit = {"path": str(path), "normalization_version": str(data["normalization_version"]),
                 "cache_version": str(data["cache_version"]), "raw_source": str(data["raw_source"])}
        if split_independent:
            audit["format_version"] = format_version
        else:
            audit.update(source_loader=str(data["source_loader"]), split_tag=str(data["split_tag"]),
                         split_seed=int(data["split_seed"]), train_seed=int(data["train_seed"]),
                         itr=int(data["itr"]))
        return item, audit


def load_protocol_cache(manifest, config, cache_root=None, *, selected_splits=("train", "val", "test")):
    """Load a manifest-defined protocol cache into ``Record``-compatible dicts.

    The function deliberately imports no reference loader and performs no raw-data
    access.  The manifest owns the fold membership and each item is checked against
    that membership before it is returned.
    """
    from data.dataset import Record
    if (not isinstance(selected_splits, (tuple, list)) or not selected_splits
            or len(set(selected_splits)) != len(selected_splits)
            or any(split not in ("train", "val", "test") for split in selected_splits)):
        raise ValueError("selected_splits must contain distinct train/val/test names")
    manifest = Path(manifest)
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    protocol = spec.get("protocol")
    from data.protocol_extractors.bmf_strict import STRICT_CACHE_VERSION
    strict = spec.get("cache_version") == STRICT_CACHE_VERSION
    if protocol not in ("batterymformer", "batterylife"):
        raise ValueError("manifest protocol must be batterymformer or batterylife")
    expected_seed = spec.get("split_seed")
    expected_tag = spec.get("split_tag")
    expected_cache_version = spec.get("cache_version")
    expected_norm_version = spec.get("normalization_version")
    split_independent = (protocol == "batterylife" and
                         expected_cache_version in _SHARED_CACHE_VERSIONS)
    root = Path(cache_root or (manifest.parent if split_independent else
                               spec.get("cache_root") or manifest.parent))
    observed_used = int(spec["observed_used"]) if spec.get("observed_used") is not None \
        else config.observed_cycles
    if not 1 <= observed_used <= config.observed_cycles:
        raise ValueError(f"manifest observed_used {observed_used} outside "
                         f"1..{config.observed_cycles}")
    result, seen = {}, set()
    for split in selected_splits:
        rows = []
        for entry in spec.get(split, []):
            rel = entry if isinstance(entry, str) else entry["file"]
            path = root / rel
            item, audit = _read(path, protocol, None if split_independent else split)
            if strict:
                for name in ("raw_cycle_data_length", "processed_soh_length"):
                    if name not in item:
                        raise ValueError(f"strict cache {path} lacks {name}")
                audit.update(raw_cycle_data_length=int(item["raw_cycle_data_length"]),
                             processed_soh_length=int(item["processed_soh_length"]))
            if split_independent:
                if audit["format_version"] != SHARED_FORMAT_VERSION:
                    raise ValueError(f"{path}: shared BatteryLife manifest requires format version {SHARED_FORMAT_VERSION}")
                audit.update(source_loader=str(spec.get("source_loader", "unknown")),
                             split_tag=str(expected_tag or ""),
                             split_seed=int(spec.get("split_seed", spec.get("seed", 0))),
                             train_seed=int(spec.get("train_seed", spec.get("seed", 0))),
                             itr=int(spec.get("itr", 1)))
            cached_cycles = int(item["curves"].shape[0])
            if cached_cycles < config.observed_cycles:
                raise ValueError(f"{path}: cached window {cached_cycles} < requested "
                                 f"observed_cycles {config.observed_cycles}")
            if cached_cycles > config.observed_cycles:
                if bool(np.asarray(item["trajectory_mask"]).any()):
                    raise ValueError(f"{path}: shortened observed window ({config.observed_cycles} "
                                     f"of {cached_cycles} cached cycles) is only valid for caches "
                                     "without auxiliary trajectory labels; the label grid would "
                                     "no longer match the query grid")
                for name in _PER_CYCLE_FIELDS[protocol]:
                    if name in item:
                        item[name] = item[name][:config.observed_cycles]
            expected_shape = (config.observed_cycles, 3, 300)
            if tuple(item["curves"].shape) != expected_shape:
                raise ValueError(f"{path}: curves shape {item['curves'].shape} != {expected_shape}")
            if config.points != 150:
                raise ValueError("protocol caches require points=150 after explicit 300->2x150 conversion")
            if expected_seed is not None and audit["split_seed"] != int(expected_seed):
                raise ValueError(f"{path}: split_seed differs from manifest")
            if expected_tag is not None and audit["split_tag"] != str(expected_tag):
                raise ValueError(f"{path}: split_tag differs from manifest")
            if expected_cache_version is not None and audit["cache_version"] != str(expected_cache_version):
                raise ValueError(f"{path}: cache_version differs from manifest")
            if expected_norm_version is not None and audit["normalization_version"] != str(expected_norm_version):
                raise ValueError(f"{path}: normalization_version differs from manifest")
            entry_dataset = entry.get("dataset") if isinstance(entry, dict) else None
            entry_cell = entry.get("cell_id") if isinstance(entry, dict) else None
            if entry_dataset is not None and str(item["dataset"]) != str(entry_dataset):
                raise ValueError(f"{path}: dataset identity differs from manifest")
            if entry_cell is not None and str(item["cell_id"]) != str(entry_cell):
                raise ValueError(f"{path}: cell identity differs from manifest")
            entry_source = entry.get("source_file") if isinstance(entry, dict) else None
            if entry_source is not None and str(item["raw_source"]) != str(entry_source):
                raise ValueError(f"{path}: raw source differs from manifest")
            if split_independent and entry_source is not None:
                audit["raw_source"] = str(entry_source)
            if split_independent:
                for name in ("source_asset", "source_asset_sha256"):
                    if name in spec:
                        audit[name] = spec[name]
            if observed_used < config.observed_cycles:
                apply_observed_truncation(item, observed_used)
                audit["observed_used"] = observed_used
            if protocol == "batterymformer" and (config.horizon != 5000 or item["future_soh"].shape[-1] != 5000):
                raise ValueError("BatteryMFormer protocol requires horizon=5000")
            key = (str(item["dataset"]), str(item["cell_id"]))
            if key in seen:
                raise ValueError(f"battery occurs in multiple splits: {key}")
            seen.add(key)
            curves, curve_mask = reference_inputs.sequence_to_phase(item["curves"], item["curve_mask"])
            inputs = {"curves": curves, "curve_mask": curve_mask,
                      "capacity": item["capacity"], "capacity_mask": item["capacity_mask"],
                      "cycle_mask": item["cycle_mask"], "metadata": {}, "metadata_mask": {}}
            if protocol == "batterymformer":
                inputs["protocol_features"] = {k: item[k] for k in
                    ("soc_input", "soc_mask", "soh_input", "soh_mask", "cycle_level_features", "cycle_level_features_mask",
                     "aging_condition_embedding", "aging_condition_mask",
                     "protocol_feature_mask")}
            elif spec.get("cache_version"):
                # BatteryLife has no BMF protocol fields.  Carry an explicit
                # all-masked block so absence cannot be confused with a zero
                # measurement inside the Adapter.
                inputs["protocol_features"] = {
                    "soc_input": np.zeros((config.observed_cycles, 300), np.float32),
                    "soc_mask": np.zeros((config.observed_cycles, 300), bool),
                    "soh_input": np.zeros((config.observed_cycles, 1), np.float32),
                    "soh_mask": np.zeros((config.observed_cycles, 1), bool),
                    "cycle_level_features": np.zeros((config.observed_cycles, 2), np.float32),
                    "cycle_level_features_mask": np.zeros((config.observed_cycles, 2), bool),
                    "aging_condition_embedding": np.zeros((1, config.reference_embedding_dim), np.float32),
                    "aging_condition_mask": np.zeros((1, config.reference_embedding_dim), bool),
                    "protocol_feature_mask": np.asarray(False, bool),
                }
            if protocol == "batterylife" and item["future_soh"].shape[-1] != config.horizon:
                if bool(np.asarray(item["trajectory_mask"]).any()):
                    raise ValueError(f"{path}: BatteryLife auxiliary curve labels need "
                                     f"horizon={item['future_soh'].shape[-1]}, got config.horizon={config.horizon}")
                future_soh = np.zeros((config.horizon,), np.float32)
                future_mask = np.zeros((config.horizon,), bool)
            else:
                future_soh, future_mask = item["future_soh"], item["future_mask"]
            query_cycles = (np.arange(1, 5001, dtype=np.float32) if protocol == "batterymformer" else
                            np.arange(config.observed_cycles + 1,
                                      config.observed_cycles + config.horizon + 1, dtype=np.float32))
            targets = {"total_life": item["total_life"], "future_soh": future_soh,
                       "future_mask": future_mask, "trajectory_mask": item.get("trajectory_mask", future_mask),
                       "start_soh": item.get("start_soh", np.asarray(np.nan, np.float32)),
                       "query_cycles": query_cycles}
            rows.append(Record(str(item["cell_id"]), str(item["dataset"]), inputs, targets,
                               {"eol_source": "reference_loader", "protocol": protocol, **audit}))
        result[split] = rows
    return result, {"protocol": protocol, "manifest": str(manifest),
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    **({"evaluation_protocol": STRICT_CACHE_VERSION} if strict else {}),
                    "split_protocol": spec.get("split_protocol"), "split_tag": spec.get("split_tag"),
                    "seed": spec.get("seed"), "itr": spec.get("itr", 1),
                    "observed_used": observed_used,
                    "cache_root": str(root)}
