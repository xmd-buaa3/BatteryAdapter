"""BatteryMFormer reference-based extraction for the strict prefix protocol.

The historic built-in extractor deliberately remains unchanged.  This path uses
the actual checked-in reference preprocessing function, including its padding,
cleaning and dataset-specific branches, and records its source digest.  It
covers every official BatteryMFormer unit: Li-ion and ZN-coin pure-OOD splits
and the CALB/NA-ion leave-one-aging-condition-out folds.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data.protocol_extractors.common import SkipCell, _embedding

STRICT_CACHE_VERSION = "bmf-strict-prefix-v1"
REFERENCE_LOADER = (Path(__file__).resolve().parents[2] /
                    "reference/BatteryMFormer/data_provider/data_loader_soh_optimized.py")

# The 22 official BatteryMFormer units: split tag -> (reference dataset_name,
# official training seed).  pure-OOD domains train each split with its own
# seed; the LOAO folds all train with seed 2021 (assets/per_fold_hyperparameters.md).
OFFICIAL_FOLDS = {
    **{f"Liion_split_seed{s}": ("Li_ion", s) for s in (2021, 42, 2024)},
    **{f"ZNcoin_split_seed{s}": ("ZN-coin", s) for s in (2021, 42, 2024)},
    **{f"CALB_loao_cond{c}_seed2021": ("CALB", 2021) for c in range(396, 400)},
    **{f"NA-ion_loao_cond{c}_seed2021": ("NA-ion", 2021) for c in range(400, 412)},
}

_ALIASED_EMBEDDINGS = {}
_ALIASED_DIRECTORY = None


def aliased_prompt_embeddings(path, cell_name):
    """Prompt-embedding table with a Stanford _Ref cell aliased to its non-_Ref key.

    The official table keys the three Stanford Nova _Ref test cells by their
    non-_Ref names; the embeddings are aging-condition level and the paired
    values are digest-identical, so the extractor resolves them through the
    same _Ref-stripping fallback.  The reference loader, however, indexes the
    table directly by file name, so the parity replay hands it a derived table
    carrying the alias for the cell it is replaying.  A non-_Ref cell, a table
    that already keys the cell, and a _Ref cell without a non-_Ref twin all
    return the source table unchanged (only additively derived tables are
    materialized, once per source digest and cell name).
    """
    name = Path(cell_name).name
    alias = name.replace("_Ref", "")
    if not (name.startswith("Stanford") and "_Ref" in name and alias != name):
        return path
    global _ALIASED_DIRECTORY
    path = Path(path)
    payload = path.read_bytes()
    key = (str(path.resolve()), hashlib.sha256(payload).hexdigest(), name)
    if key not in _ALIASED_EMBEDDINGS:
        table = pickle.loads(payload)
        if alias in table and name not in table:
            table[name] = table[alias]
        if _ALIASED_DIRECTORY is None:
            import tempfile
            _ALIASED_DIRECTORY = tempfile.mkdtemp(prefix="strict-embedding-alias-")
        target = Path(_ALIASED_DIRECTORY) / f"{key[1][:16]}-{name}"
        with target.open("wb") as handle:
            pickle.dump(table, handle)
        _ALIASED_EMBEDDINGS[key] = target
    return _ALIASED_EMBEDDINGS[key]


def reference_eta(name):
    """Per-dataset EOL threshold of the reference normalization (CALB=0.9)."""
    return .9 if str(name).startswith("CALB") else .8


def reference_loader_sha256():
    return hashlib.sha256(REFERENCE_LOADER.read_bytes()).hexdigest()


def reference_dataset_dir(name):
    name = Path(name).name
    # The reference keeps the hyphenated directory for these three datasets
    # (_prepare_samples: dataset_dir = file_name.split('_')[0]) and only
    # dash-replaces the remaining sources.
    if name.split("_")[0] in ("NA-ion", "ZN-coin", "CALB"):
        return name.split("_")[0]
    dataset = name.split("_")[0].replace("-", "_")
    if dataset == "MICH":
        return "total_MICH"
    if "Tongji" in name:
        return "Tongji"
    if "ISU-ILCC" in name:
        return "ISU_ILCC"
    if "Stanford" in name:
        return "Stanford_2"
    return dataset


def strict_soh(root, path):
    name = Path(path).name
    source = Path(root) / reference_dataset_dir(name) / name
    if not source.is_file() and name.startswith("Stanford") and "_Ref" in name:
        # On-disk Stanford Nova files drop the _Ref infix (the raw dispatch in
        # reference_raw_data_path already mirrors this); the processed-SOH tree
        # follows the same naming.
        source = Path(root) / reference_dataset_dir(name) / name.replace("_Ref", "")
    if not source.is_file():
        # The reference loader prints and continues on a missing processed SOH
        # file (_prepare_samples), so the cell is excluded from the official
        # protocol rather than fatal; record it as a skip.
        raise SkipCell("reference excludes cell with no processed SOH file")
    with source.open("rb") as f:
        y = np.asarray(pickle.load(f)["SOH"], np.float32).reshape(-1)
    if not y.size or not np.isfinite(y).all():
        raise ValueError(f"invalid reference SOH: {source}")
    return y, str(source)


def reference_raw_data_path(path):
    """Resolve the source actually read by the reference prefix dispatch."""
    root, name = path.parent.parent, path.name
    prefix = name.split("_")[0]
    if prefix.startswith("SMICH"):
        return root / "MICH_EXP" / name[1:]
    if prefix.startswith("Stanford"):
        return root / "Stanford_2" / name.replace("_Ref", "")
    if prefix.startswith("MICH"):
        return root / "total_MICH" / name
    if prefix.startswith("UL-PUR"):
        return root / "UL_PUR" / name
    if prefix.startswith("ISU-ILCC"):
        return root / "ISU_ILCC" / name
    return root / reference_dataset_dir(name) / name


@lru_cache(maxsize=1)
def _reference_preprocessor():
    # Import only when extracting strict data; restore the reference module's
    # global sys.path / warning filter changes after import.
    before = sys.path[:]
    try:
        with warnings.catch_warnings():
            spec = importlib.util.spec_from_file_location("_bmf_reference_loader", REFERENCE_LOADER)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    finally:
        sys.path[:] = before
    obj = module.Dataset_SOH_Forecasting.__new__(module.Dataset_SOH_Forecasting)
    obj.args = SimpleNamespace(model="BatteryMFormer")
    obj.seq_len = 1
    obj.early_cycle_threshold = 100
    obj.charge_discharge_len = 300
    obj.input_mode = "current_voltage"
    obj.use_capacity_resample = True
    obj.alignment_check = False
    obj.need_keys = ["current_in_A", "voltage_in_V", "charge_capacity_in_Ah",
                     "discharge_capacity_in_Ah", "time_in_s"]
    # The reference pins these ZN-coin cells to charge-first ordering inside
    # __init__ (data_loader_soh_optimized.py); the literal is pinned by the
    # reference loader digest, so copy it verbatim.
    obj.ZN_coin_charge_first_file_names = [
        'ZN-coin_402-1_20231209225636_01_1.pkl', 'ZN-coin_402-2_20231209225727_01_2.pkl',
        'ZN-coin_402-3_20231209225844_01_3.pkl', 'ZN-coin_403-1_20231209225922_01_4.pkl',
        'ZN-coin_428-1_20231212185048_01_2.pkl', 'ZN-coin_428-2_20231212185058_01_4.pkl',
        'ZN-coin_429-1_20231212185129_01_5.pkl', 'ZN-coin_429-2_20231212185157_01_8.pkl',
        'ZN-coin_430-1_20231212185250_02_6.pkl', 'ZN-coin_430-2_20231212185305_02_7.pkl',
        'ZN-coin_430-3_20231212185323_03_2.pkl',
    ]
    return obj


def extract_batterymformer_strict(path, *, dataset, processed_soh_root, prompt_embeddings_path):
    """Build one strict cache item; ``dataset`` is the cell's constituent source."""
    path = Path(path)
    if path.parent.name != reference_dataset_dir(path.name):
        raise ValueError(f"raw source does not match reference dataset directory: {path}")
    ref = _reference_preprocessor()
    ref.root_path = str(path.parent.parent)
    raw_data_path = reference_raw_data_path(path)
    if not path.is_file() or not raw_data_path.is_file():
        raise FileNotFoundError(f"reference raw paths missing: {path} / {raw_data_path}")
    raw = ref.read_cell_data_according_to_prefix(path.name)
    soh, soh_source = strict_soh(processed_soh_root, path)
    if len(soh) <= 100:
        raise SkipCell("reference excludes SOH trajectory length <= 100")
    if len(soh) > 5000:
        raise ValueError("reference trajectory exceeds the configured 5000-element grid")
    records = raw["cycle_data"]
    if not records:
        raise ValueError("reference input has no raw cycles")
    nominal = float(raw["nominal_capacity_in_Ah"])
    interval = np.asarray(raw["SOC_interval"], dtype=float).reshape(-1)
    if not np.isfinite(nominal) or nominal <= 0 or interval.size < 2:
        raise ValueError("invalid reference nominal capacity or SOC interval")
    curves, ce, ee = ref._prepare_charge_discharge_curves(
        path.name, records, nominal, float(interval[0]), float(interval[1]))
    curves = np.asarray(curves, dtype=np.float32)
    ce, ee = np.asarray(ce, np.float32), np.asarray(ee, np.float32)
    if curves.shape != (100, 4, 300) or ce.shape != (100,) or ee.shape != (100,):
        raise ValueError("reference preprocessing returned unexpected shapes")
    if not np.isfinite(curves).all() or not np.isfinite(ce).all() or not np.isfinite(ee).all():
        raise ValueError("nonfinite reference preprocessing output")
    embedding_fallback = False
    try:
        embedding, embedding_source = _embedding(prompt_embeddings_path, path.name)
    except SkipCell:
        if not (path.name.startswith("Stanford") and "_Ref" in path.name):
            raise
        # The official embedding table keys the Stanford test cells by their
        # non-_Ref names; the embeddings are aging-condition level and the
        # paired values are identical (verified by digest).
        embedding, embedding_source = _embedding(
            prompt_embeddings_path, path.name.replace("_Ref", ""))
        embedding_fallback = True
    end = min(len(records), 100)
    cycle_mask = np.arange(100) < end
    curve_mask = np.broadcast_to(cycle_mask[:, None, None], (100, 3, 300)).copy()
    soc_mask = np.broadcast_to(cycle_mask[:, None], (100, 300)).copy()
    feature_mask = np.broadcast_to(cycle_mask[:, None], (100, 2)).copy()
    eta = reference_eta(path.name)
    y = (np.asarray(soh, np.float32) - eta) / (1 - eta)
    future = np.zeros(5000, np.float32)
    future[:len(y)] = y
    mask = np.zeros(5000, bool)
    mask[end:len(y)] = True
    return dict(curves=curves[:, :3], curve_mask=curve_mask, cycle_mask=cycle_mask,
                capacity=np.zeros((100, 1), np.float32), capacity_mask=np.zeros((100, 1), bool),
                soc_input=curves[:, 3], soc_mask=soc_mask, soh_input=y[:100, None],
                soh_mask=cycle_mask[:, None].copy(), cycle_level_features=np.stack((ce, ee), axis=1),
                cycle_level_features_mask=feature_mask, aging_condition_embedding=embedding,
                aging_condition_mask=np.ones((1, 1024), bool), protocol_feature_mask=np.asarray(True),
                future_soh=future, future_mask=mask, trajectory_mask=mask.copy(),
                total_life=np.asarray(float(len(y)), np.float32), start_soh=np.asarray(y[end-1], np.float32),
                start_soh_mask=np.asarray(True), raw_cycle_data_length=np.asarray(len(records), np.int64),
                processed_soh_length=np.asarray(len(y), np.int64), reference_loader_sha256=np.asarray(reference_loader_sha256()),
                cache_version=np.asarray(STRICT_CACHE_VERSION), normalization_version=np.asarray(f"batterymformer-soh-{eta:g}-v1"),
                raw_source=np.asarray(str(path)),
                provenance_json=np.asarray(json.dumps({"soh_source": soh_source, "embedding_source": embedding_source,
                                                       "raw_data_source": str(raw_data_path),
                                                       "embedding_fallback": embedding_fallback}, sort_keys=True)))
