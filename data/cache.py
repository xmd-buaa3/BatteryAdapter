"""The ``data/multicurve`` per-cell input cache, mirroring the ``data/single`` contract.

``experimentals/extract_npz_cache.py`` runs the validated raw loader once (levels forced to
``(1, 2, 3)``) and writes one compact npz per cell to

    <data-root>/multicurve/<domain-lower>/<cell_id>.npz

holding all three information levels (the three tiers in plan.md): the L1 preprocessed curve/capacity
grids, the L2 derived metadata and the L3 SOC grids, plus the resampling audits. Training
then reads these files directly and never re-parses the raw BatteryLife pickles. Label
and early-SOH npz files stay live: they are small, audited by SHA at load time, and not
duplicated into the cache.

Level gating happens at load time: the cache always contains every level, and
``load_manifest_cache`` attaches only what ``config.levels`` permits, exactly like the
raw path builds records at ``max(config.levels)``. Each npz records the manifest SHA256
and the extraction-time config echo, so a stale cache is rejected instead of silently
training on outdated inputs.

npz schema (``format_version`` 5, all arrays written by ``save_cell_cache``):

| field | type | content |
|---|---|---|
| `curves` / `curve_mask` | float32/bool [s,2,C,L] | Level 1: resampled V/I(+Q) grids |
| `capacity` / `capacity_mask` | float32/bool [s,1] | Level 1: per-cycle capacity scalar |
| `cycle_mask` | bool [s] | Level 1: observed cycles |
| `soc` / `soc_mask` | float32/bool [s,2,L] | Level 3: SOC grids (zeros when absent) |
| `soc_declared` | bool | raw pickle carried a usable SOC field |
| `cycle_features` / `cycle_features_mask` | float32/bool [s,2] | Level 3: CE/EE cycle-level features (BatteryMFormer cycle-level state) |
| `metadata_json` | str | Level 2 whitelist metadata (scalars / per-cycle arrays) |
| `metadata_profile` | str | metadata whitelist profile the cache was extracted under |
| `audit_json` | str | capacity_basis / resampling_coordinates / phase_coverage / multiple_segments |
| `metadata_audit_json` | str | per-field metadata provenance from the raw loader |
| `cell_id` / `domain` / `split` / `raw_path` | str | identity cross-checked against the manifest |
| `manifest_sha256` | str | pins the manifest the cache was extracted from |
| `observed_cycles` / `points` / `curve_channels` | int | extraction-time config echo |
| `current_threshold` / `eta` | float64 | extraction-time config echo |
| `input_layout_version` / `soh_scale` | int / str | early-fusion layout and SOH-scale contract |
| `source_eta` | float64 | source threshold used by the affine SOH scale |
| `format_version` | int | bumped on incompatible extraction changes |

``save_cell_features`` writes the manifest-free dataset-root variant used by
``experimentals/extract_npz_cache.py --dataset-root`` (same L1/L2/L3 field names, same
layout as ``data/single``: ``<data-root>/multicurve/<source-lower>/<stem>.npz``).
It replaces the manifest identity fields with dataset identity and provenance:
`cell_id` / `source` / `file_name` / `raw_path`, plus `early_soh` float32[s]
(history100 semantics, identical to ``data/single`` ``input``), `n_valid_cycles`,
`soh_source` (``batterymformer_processed_soh`` or ``raw_pickle``), `soc_interval`,
`cycle_life`, `capacity_source` (the declared per-cycle capacity derivation) and
`cycle_number_offset`: BatteryML built some sources with the first life cycle
missing (numbering starts at 2, e.g. Tongji), so the extractor rebases the first
recorded cycle to life cycle 1 — the same positional convention BatteryMFormer's
own loader applies — and records the derived offset here. Cells whose recorded
life stops before slot 100 (e.g. 99-cycle CALB pickles) are cached with the
missing trailing slots masked False; absence stays explicit for the manifest step.

Protocol levels derive canonical Adapter tensors from the physical cache at load
time. The cache stores protocol features in explicit named arrays; it never stores
a nested native-model input bundle. Level 4 uses the BatteryMFormer fields and
Level 5 carries an explicit all-masked block for fields absent from BatteryLife.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from data.batterylife import _load_early_soh, _load_labels, _read_protocol, _record_window, _unique_sources
from data.dataset import apply_early_soh, check_splits, record_from_inputs
from data.feature_provenance import allowed_metadata_fields, audit_metadata, audit_state

FORMAT_VERSION = 5
_AUDIT_KEYS = ("capacity_basis", "resampling_coordinates", "phase_coverage", "multiple_segments")


def _safe_name(value: str, kind: str) -> str:
    if not value or value in (".", "..") or os.sep in value or (os.altsep and os.altsep in value):
        raise ValueError(f"{kind} must be a safe single path component, got {value!r}")
    return value


def multicurve_cache_path(data_root: str | Path, domain: str, cell_id: str) -> Path:
    """Canonical cache location: ``<data-root>/multicurve/<domain-lower>/<cell_id>.npz``."""
    _safe_name(cell_id, "cell_id")
    return Path(data_root) / "multicurve" / _safe_name(domain, "domain").lower() / f"{cell_id}.npz"


def _jsonify(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def save_cell_cache(path: str | Path, record, *, config, manifest_sha256: str, split: str) -> None:
    """Write one cell's three-level inputs and audits as an atomic npz."""
    inputs = record.inputs
    protocol_features = inputs.get("protocol_features", {})
    _atomic_npz(Path(path),
                curves=inputs["curves"], curve_mask=inputs["curve_mask"],
                capacity=inputs["capacity"], capacity_mask=inputs["capacity_mask"],
                cycle_mask=inputs["cycle_mask"],
                soc=inputs["soc"], soc_mask=inputs["soc_mask"],
                cycle_features=np.asarray(inputs["cycle_features"], np.float32),
                cycle_features_mask=np.asarray(inputs["cycle_features_mask"], bool),
                protocol_cycle_level_features=np.asarray(protocol_features.get("cycle_level_features", np.zeros((config.observed_cycles, 2), np.float32))),
                protocol_aging_condition_embedding=np.asarray(protocol_features.get("aging_condition_embedding", np.zeros((1, config.reference_embedding_dim), np.float32))),
                soc_declared=np.asarray(bool(inputs["soc_mask"].any())),
                metadata_json=np.asarray(json.dumps({k: _jsonify(v) for k, v in inputs["metadata"].items()})),
                metadata_profile=np.asarray(config.metadata_profile),
                audit_json=np.asarray(json.dumps({k: _jsonify(v) for k, v in record.audit.items() if k in _AUDIT_KEYS})),
                metadata_audit_json=np.asarray(json.dumps({k: _jsonify(v) for k, v in record.audit["metadata_sources"].items()})),
                cell_id=np.asarray(record.cell_id), domain=np.asarray(record.domain), split=np.asarray(split),
                raw_path=np.asarray(record.audit["raw_path"]), manifest_sha256=np.asarray(manifest_sha256),
                observed_cycles=np.asarray(config.observed_cycles, np.int64),
                points=np.asarray(config.points, np.int64),
                curve_channels=np.asarray(config.curve_channels, np.int64),
                current_threshold=np.asarray(config.current_threshold, np.float64),
                eta=np.asarray(config.eta, np.float64),
                source_eta=np.asarray(record.audit.get("provenance", {}).get("soh", {}).get(
                    "source_eta", config.eta), np.float64),
                input_layout_version=np.asarray(config.input_layout_version, np.int64),
                soh_scale=np.asarray(config.soh_scale),
                format_version=np.asarray(FORMAT_VERSION, np.int64))


def save_cell_features(path: str | Path, *, cell_id: str, source: str, file_name: str, raw_path: str,
                       inputs, audit, metadata, metadata_sources, early_soh, n_valid_cycles: int,
                       soh_source: str, soc_interval: float, cycle_life: int, capacity_source: str,
                       cycle_number_offset: int, config, source_eta: float | None = None) -> None:
    """Write one cell's three-level inputs for the dataset-root (manifest-free) cache."""
    protocol_features = inputs.get("protocol_features", {})
    _atomic_npz(Path(path),
                curves=inputs["curves"], curve_mask=inputs["curve_mask"],
                capacity=inputs["capacity"], capacity_mask=inputs["capacity_mask"],
                cycle_mask=inputs["cycle_mask"],
                soc=inputs["soc"], soc_mask=inputs["soc_mask"],
                cycle_features=np.asarray(inputs["cycle_features"], np.float32),
                cycle_features_mask=np.asarray(inputs["cycle_features_mask"], bool),
                protocol_cycle_level_features=np.asarray(protocol_features.get("cycle_level_features", np.zeros((config.observed_cycles, 2), np.float32))),
                protocol_aging_condition_embedding=np.asarray(protocol_features.get("aging_condition_embedding", np.zeros((1, config.reference_embedding_dim), np.float32))),
                soc_declared=np.asarray(bool(inputs["soc_mask"].any())),
                metadata_json=np.asarray(json.dumps({k: _jsonify(v) for k, v in metadata.items()})),
                metadata_profile=np.asarray(config.metadata_profile),
                audit_json=np.asarray(json.dumps({k: _jsonify(v) for k, v in audit.items() if k in _AUDIT_KEYS})),
                metadata_audit_json=np.asarray(json.dumps({k: _jsonify(v) for k, v in metadata_sources.items()})),
                early_soh=early_soh, n_valid_cycles=np.asarray(n_valid_cycles, np.int64),
                soh_source=np.asarray(soh_source), soc_interval=np.asarray(soc_interval, np.float64),
                cycle_life=np.asarray(cycle_life, np.int64), capacity_source=np.asarray(capacity_source),
                source_eta=np.asarray(config.eta if source_eta is None else source_eta, np.float64),
                cycle_number_offset=np.asarray(cycle_number_offset, np.int64),
                cell_id=np.asarray(cell_id), source=np.asarray(source), file_name=np.asarray(file_name),
                raw_path=np.asarray(raw_path),
                observed_cycles=np.asarray(config.observed_cycles, np.int64),
                points=np.asarray(config.points, np.int64),
                curve_channels=np.asarray(config.curve_channels, np.int64),
                current_threshold=np.asarray(config.current_threshold, np.float64),
                eta=np.asarray(config.eta, np.float64),
                input_layout_version=np.asarray(config.input_layout_version, np.int64),
                soh_scale=np.asarray(config.soh_scale),
                format_version=np.asarray(FORMAT_VERSION, np.int64))


def load_cell_features(path: str | Path) -> dict:
    """Read one dataset-root cache file back (verification and the future training bridge)."""
    with np.load(path, allow_pickle=False) as data:
        block = {name: data[name] for name in ("curves", "curve_mask", "capacity", "capacity_mask",
                                               "cycle_mask", "soc", "soc_mask", "early_soh",
                                               "cycle_features", "cycle_features_mask")}
        block.update(soc_declared=bool(data["soc_declared"]),
                     n_valid_cycles=int(data["n_valid_cycles"]),
                     cycle_life=int(data["cycle_life"]),
                     soc_interval=float(data["soc_interval"]),
                     cycle_number_offset=int(data["cycle_number_offset"]),
                     metadata=json.loads(str(data["metadata_json"])),
                     metadata_profile=str(data["metadata_profile"]),
                     audit=json.loads(str(data["audit_json"])),
                     metadata_sources=json.loads(str(data["metadata_audit_json"])),
                     soh_source=str(data["soh_source"]), capacity_source=str(data["capacity_source"]),
                     source_eta=float(data["source_eta"]),
                     cell_id=str(data["cell_id"]), source=str(data["source"]),
                     file_name=str(data["file_name"]), raw_path=str(data["raw_path"]),
                     input_layout_version=int(data["input_layout_version"]),
                     soh_scale=str(data["soh_scale"]),
                     format_version=int(data["format_version"]))
        return block


def _load_cell_cache(cache_path: Path, source, raw_path, config, manifest_sha256: str) -> dict:
    """Read and fully validate one cached cell against the manifest and config."""
    if not cache_path.is_file():
        raise ValueError(f"multicurve cache missing for {source['domain']}/{source['cell_id']} "
                         f"({cache_path}); run train_eval_scripts/extract_npz_cache.sh first")
    with np.load(cache_path, allow_pickle=False) as data:
        if int(data["format_version"]) != FORMAT_VERSION:
            raise ValueError(f"{cache_path}: format_version {int(data['format_version'])} != {FORMAT_VERSION}; re-extract")
        if str(data["manifest_sha256"]) != manifest_sha256:
            raise ValueError(f"{cache_path}: cache was extracted from a different manifest version "
                             f"({str(data['manifest_sha256'])[:12]}... != {manifest_sha256[:12]}...); re-extract with --rebuild")
        echo = {"observed_cycles": config.observed_cycles, "points": config.points}
        for name, expected in echo.items():
            if int(data[name]) != expected:
                raise ValueError(f"{cache_path}: cache {name}={int(data[name])} differs from config {expected}; re-extract")
        if int(data["input_layout_version"]) != int(config.input_layout_version):
            raise ValueError(f"{cache_path}: input layout version differs from config; re-extract")
        if str(data["metadata_profile"]) != str(config.metadata_profile):
            raise ValueError(f"{cache_path}: cache metadata_profile {str(data['metadata_profile'])!r} differs "
                             f"from config {config.metadata_profile!r}; re-extract")
        if str(data["soh_scale"]) != str(config.soh_scale):
            raise ValueError(f"{cache_path}: SOH scale differs from config; re-extract")
        declared_source_eta = source.get("provenance", {}).get("soh", {}).get("source_eta")
        if declared_source_eta is not None and not np.isclose(float(data["source_eta"]), float(declared_source_eta)):
            raise ValueError(f"{cache_path}: source SOH threshold differs from the manifest; re-extract")
        cached_channels = int(data["curve_channels"])
        if cached_channels < config.curve_channels:
            raise ValueError(f"{cache_path}: cache curve_channels={cached_channels} lacks the "
                             f"{config.curve_channels} channels the config needs; re-extract")
        for name in ("current_threshold", "eta"):
            if float(data[name]) != float(getattr(config, name)):
                raise ValueError(f"{cache_path}: cache {name}={float(data[name])} differs from config "
                                 f"{getattr(config, name)}; re-extract")
        if os.path.normcase(str(data["cell_id"])) != os.path.normcase(source["cell_id"]) \
                or os.path.normcase(str(data["domain"])) != os.path.normcase(source["domain"]):
            raise ValueError(f"{cache_path}: cached identity does not match the manifest record")
        if os.path.normcase(str(data["raw_path"])) != os.path.normcase(str(raw_path)):
            raise ValueError(f"{cache_path}: cached raw_path refers to a different pickle than the manifest")
        expected_shapes = {"curves": (config.observed_cycles, 2, cached_channels, config.points),
                           "curve_mask": (config.observed_cycles, 2, cached_channels, config.points),
                           "capacity": (config.observed_cycles, 1), "capacity_mask": (config.observed_cycles, 1),
                           "cycle_mask": (config.observed_cycles,),
                           "soc": (config.observed_cycles, 2, config.points),
                           "soc_mask": (config.observed_cycles, 2, config.points),
                           "cycle_features": (config.observed_cycles, 2),
                           "cycle_features_mask": (config.observed_cycles, 2)}
        block = {name: data[name] for name in expected_shapes}
        block["protocol_cycle_level_features"] = data["protocol_cycle_level_features"]
        block["protocol_aging_condition_embedding"] = data["protocol_aging_condition_embedding"]
        for name, shape in expected_shapes.items():
            if block[name].shape != shape:
                raise ValueError(f"{cache_path}: {name} has shape {block[name].shape}, expected {shape}")
        if cached_channels > config.curve_channels:
            # The optional cumulative-capacity channel (genuinely recorded counters only)
            # is carried by the cache and dropped at load time for two-channel configs.
            block["curves"] = block["curves"][:, :, :config.curve_channels, :]
            block["curve_mask"] = block["curve_mask"][:, :, :config.curve_channels, :]
        block["soc_declared"] = bool(data["soc_declared"])
        block["metadata"] = json.loads(str(data["metadata_json"]))
        block["audit"] = json.loads(str(data["audit_json"]))
        block["metadata_sources"] = json.loads(str(data["metadata_audit_json"]))
        return block


def load_manifest_cache(path, config, cache_root):
    """``load_manifest`` equivalent reading per-cell inputs from ``data/multicurve``.

    The manifest still declares splits, capacity mappings, labels and provenance and is
    validated identically; only the raw-pickle parsing and resampling are replaced by the
    cache. Labels and early-SOH npz files are re-read live, exactly like the raw path.
    """
    path, content, protocol = _read_protocol(path, config)
    manifest_sha256 = hashlib.sha256(content).hexdigest()
    cache_root = Path(cache_root)
    splits, exclusions = {name: [] for name in ("train", "val", "test")}, []
    # Cache files carry all three levels.  The dataset transform, rather than
    # cache construction, enforces the requested information level.
    max_level = 4 if max(config.levels) >= 4 else 3
    s = config.observed_cycles
    for source, raw_path in _unique_sources(path, protocol):
        _record_window(source)  # declared window contract must stay valid even though the cache is pre-rebased
        block = _load_cell_cache(multicurve_cache_path(cache_root, source["domain"], source["cell_id"]),
                                 source, raw_path, config, manifest_sha256)
        provenance = source.get("provenance", {})
        if provenance.get("soh", {}).get("scale", config.soh_scale) != config.soh_scale:
            raise ValueError("cached early SOH provenance scale differs from the experiment scale")
        inputs = {name: block[name] for name in ("curves", "curve_mask", "capacity", "capacity_mask", "cycle_mask")}
        if max_level >= 3 and block["soc_declared"]:
            audit_state(provenance, "soc", s)
            inputs["soc"], inputs["soc_mask"] = block["soc"], block["soc_mask"]
        else:
            inputs["soc"] = np.zeros((s, 2, config.points), np.float32)
            inputs["soc_mask"] = np.zeros((s, 2, config.points), bool)
        if max_level >= 3:
            inputs["cycle_features"] = block["cycle_features"]
            inputs["cycle_features_mask"] = block["cycle_features_mask"]
        metadata = block["metadata"] if max_level >= 2 else {}
        metadata = {k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v) for k, v in metadata.items()}
        inputs["metadata"] = (audit_metadata(metadata, allowed=allowed_metadata_fields(config.metadata_profile))
                              if max_level >= 2 else {})
        label_cycles, label_soh, label_path = _load_labels(path, source, config)
        early = _load_early_soh(path, source, label_path, config) if "early_soh_path" in source and max_level >= 3 else None
        apply_early_soh(inputs, early, config, provenance, max_level)
        audit = {k: ({int(cycle_key): value for cycle_key, value in v.items()} if isinstance(v, dict) else v)
                 for k, v in block["audit"].items()}
        record = record_from_inputs(source["cell_id"], source["domain"], inputs, config, input_audit=audit,
                                    provenance=provenance, label_cycles=label_cycles, label_soh=label_soh,
                                    total_life=source["labels"].get("total_life"),
                                    eol_source=source["labels"].get("eol_source", "measured"),
                                    eol_evidence=source["labels"].get("eol_evidence"),
                                    label_definition=source["labels"]["definition"],
                                    eol_tolerance=int(protocol.get("eol_convention_tolerance", 0)),
                                    reference_cycle_features=block["protocol_cycle_level_features"] if max_level >= 4 else None,
                                    reference_aging_embedding=block["protocol_aging_condition_embedding"] if max_level >= 4 else None)
        record.audit.update(metadata_sources=block["metadata_sources"] if max_level >= 2 else {},
                            raw_path=str(raw_path),
                            label_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest())
        if record.audit["multiple_segments"] and protocol["segment_policy"] == "single":
            exclusions.append(dict(cell_id=source["cell_id"], domain=source["domain"], split=source["split"], reason="multiple_phase_segments"))
        else:
            splits[source["split"]].append(record)
    check_splits(splits)
    protocol["manifest_sha256"] = manifest_sha256
    protocol["exclusions"] = exclusions
    return splits, protocol
