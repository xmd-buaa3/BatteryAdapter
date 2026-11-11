"""Reference-data parity audit used before a strict Li-ion search is registered."""
import copy
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from data.protocol_cache import _read
from data.protocol_extractors.bmf_strict import OFFICIAL_FOLDS, STRICT_CACHE_VERSION, extract_batterymformer_strict, reference_loader_sha256, strict_soh, _reference_preprocessor, aliased_prompt_embeddings, reference_raw_data_path, SkipCell


class _IdentityScaler:
    protocol = "batterymformer"
    def transform(self, record, level):
        return copy.deepcopy(record.inputs)


def _verify_samples(item, raw, soh, embedding, dataset_name):
    """Compare the real reference Dataset's generated samples, not a second mask formula."""
    from data.dataset import BMFPrefixDataset, Record
    from reference_inputs import sequence_to_phase, phase_to_sequence
    ref = _reference_preprocessor()
    # The loader indexes the embedding table by file name; hand it the derived
    # table carrying this cell's Stanford _Ref alias so its replay matches what
    # the extractor's documented fallback produced for those cells.
    ref.args.prompt_embeddings_path = str(aliased_prompt_embeddings(embedding, raw.name))
    ref.dataset_name = dataset_name
    ref.root_path = str(raw.parent.parent)
    ref.processed_SOH_path = str(soh.parent.parent)
    ref.files = [raw.name]
    ref.max_trajectory_len = 5000
    ref.samples = []
    ref.label_scaler = SimpleNamespace(mean_=np.asarray([0.]), var_=np.asarray([1.]))
    with contextlib.redirect_stdout(io.StringIO()):
        ref._prepare_samples()
    curves, masks = sequence_to_phase(item["curves"], item["curve_mask"])
    inputs = {"curves": curves, "curve_mask": masks, "cycle_mask": item["cycle_mask"],
              "capacity": item["capacity"], "capacity_mask": item["capacity_mask"],
              "protocol_features": {k: item[k] for k in ("soc_input", "soc_mask", "soh_input", "soh_mask",
                    "cycle_level_features", "cycle_level_features_mask", "aging_condition_embedding",
                    "aging_condition_mask", "protocol_feature_mask")}}
    targets = {"future_soh": item["future_soh"], "future_mask": item["future_mask"],
               "trajectory_mask": item["trajectory_mask"], "query_cycles": np.arange(1, 5001, dtype=np.float32)}
    record = Record(str(item["cell_id"]), str(item["dataset"]), inputs, targets,
                    {"eol_source": "reference_loader", "raw_cycle_data_length": int(item["raw_cycle_data_length"]),
                     "processed_soh_length": int(item["processed_soh_length"])})
    dataset = BMFPrefixDataset([record], _IdentityScaler())
    if len(ref.samples) != len(dataset):
        raise ValueError(f"reference prefix membership differs: {raw}")
    for index, expected in enumerate(ref.samples):
        sample = dataset[index]
        prefix = index + 1
        if sample["prefix"] != prefix or not np.array_equal(sample["inputs"]["cycle_mask"].numpy(), expected["curve_attn_mask"].astype(bool)):
            raise ValueError(f"reference prefix mask differs: {raw}/{prefix}")
        ref_curves = expected["cycle_curve_data"].copy()
        ref_curves[prefix:] = 0  # collate_fn_soh applies this mask
        actual = phase_to_sequence(sample["inputs"]["curves"].numpy())
        features = sample["inputs"]["protocol_features"]
        comparisons = ((actual, ref_curves[:, :3]), (features["soc_input"].numpy(), ref_curves[:, 3]),
                       (features["soh_input"].numpy(), expected["soh_input"]),
                       (features["cycle_level_features"].numpy(), np.concatenate((expected["CEs"], expected["EEs"]), axis=1)),
                       (features["aging_condition_embedding"].numpy(), expected["aging_condition_embedding"]),
                       (sample["targets"]["future_soh"].numpy(), expected["soh_trajectory"]),
                       (sample["targets"]["trajectory_mask"].numpy(), expected["trajectory_mask"]))
        if any(not np.allclose(a, b, rtol=1e-6, atol=1e-7) for a, b in comparisons):
            raise ValueError(f"reference prefix values differ: {raw}/{prefix}")
    ref.samples = []


def verify_manifest(path, *, sha256):
    """Compare every cached cell to the pinned reference preprocessing output.

    Test labels are inspected solely for predeclared data/mask parity, never for
    candidate selection. The caller records every returned source hash.
    """
    manifest = Path(path).resolve()
    spec = json.loads(manifest.read_text(encoding="utf8"))
    tag = spec.get("split_tag")
    if (spec.get("protocol") != "batterymformer" or spec.get("cache_version") != STRICT_CACHE_VERSION
            or tag not in OFFICIAL_FOLDS
            or spec.get("reference_loader_sha256") != reference_loader_sha256()):
        raise ValueError(f"not a current strict BatteryMFormer manifest: {manifest}")
    repo = Path(__file__).resolve().parents[1]
    asset = repo / "data/protocol_splits" / f"batterymformer_{tag}.json"
    source = json.loads(asset.read_text(encoding="utf8"))
    reference_split = repo / source["source_file"]
    if sha256(asset) != spec.get("source_asset_sha256") or sha256(reference_split) != source["source_sha256"]:
        raise ValueError("strict manifest or reference split changed")
    reference = json.loads(reference_split.read_text(encoding="utf8"))
    root = Path(spec.get("cache_root") or manifest.parent).resolve()
    hashes = {str(manifest): sha256(manifest), str(asset): sha256(asset),
              str(reference_split): sha256(reference_split)}
    fields = ("curves", "curve_mask", "cycle_mask", "soc_input", "soc_mask", "soh_input", "soh_mask",
              "cycle_level_features", "cycle_level_features_mask", "aging_condition_embedding",
              "aging_condition_mask", "future_soh", "future_mask", "trajectory_mask", "total_life",
              "start_soh", "raw_cycle_data_length", "processed_soh_length")
    counts = {}
    seen = set()
    for split in ("train", "val", "test"):
        expected = sorted(Path(name).name for name in reference[split])
        names = [Path(row["source_file"]).name for row in spec.get(split, [])]
        excluded = [Path(row["source_file"]).name for row in spec.get("skipped", []) if row["split"] == split]
        if sorted(names + excluded) != expected:
            raise ValueError(f"strict cohort differs from reference split {split}")
        for row in spec.get("skipped", []):
            if row["split"] == split:
                reason = row["reason"]
                raw = Path(row["source_file"]).resolve()
                if reason == "reference excludes SOH trajectory length <= 100":
                    y, soh_source = strict_soh(Path(spec["processed_soh_root"]), raw)
                    if len(y) > 100:
                        raise ValueError(f"strict excluded cell violates reference EOL filter: {raw}")
                    hashed = (raw, reference_raw_data_path(raw).resolve(), Path(soh_source).resolve())
                elif reason == "reference excludes cell with no processed SOH file":
                    try:
                        strict_soh(Path(spec["processed_soh_root"]), raw)
                    except SkipCell:
                        pass
                    else:
                        raise ValueError(f"strict excluded cell has a processed SOH file: {raw}")
                    hashed = (raw, reference_raw_data_path(raw).resolve())
                else:
                    raise ValueError("strict cohort contains an unverified skipped cell")
                for path_ in hashed:
                    hashes[str(path_)] = sha256(path_)
        samples = points = 0
        for row in spec.get(split, []):
            cell = Path(row["source_file"]).name
            if cell in seen:
                raise ValueError(f"cell occurs more than once: {cell}")
            seen.add(cell)
            cache = (root / row["file"]).resolve()
            item, audit = _read(cache, "batterymformer", split)
            provenance = json.loads(str(item["provenance_json"]))
            raw = Path(str(item["raw_source"])).resolve()
            raw_data = Path(provenance["raw_data_source"]).resolve()
            soh = Path(provenance["soh_source"]).resolve()
            embedding = Path(provenance["embedding_source"]).resolve()
            if (raw.name != cell or str(item["cell_id"]) != str(row["cell_id"])
                    or str(item["dataset"]) != str(row["dataset"])
                    or audit["split_tag"] != tag or audit["split_seed"] != int(spec["split_seed"])
                    or audit["cache_version"] != STRICT_CACHE_VERSION
                    or str(item["reference_loader_sha256"]) != reference_loader_sha256()):
                raise ValueError(f"reference identity differs: {cache}")
            actual = extract_batterymformer_strict(raw, dataset=str(item["dataset"]),
                         processed_soh_root=soh.parent.parent, prompt_embeddings_path=embedding)
            if any(not np.array_equal(np.asarray(item[k]), np.asarray(actual[k])) for k in fields):
                raise ValueError(f"reference preprocessing mismatch: {cache}")
            _verify_samples(item, raw, soh, embedding, OFFICIAL_FOLDS[tag][0])
            raw_len, eol = int(item["raw_cycle_data_length"]), int(item["processed_soh_length"])
            if not 100 < eol <= 5000 or raw_len < 1:
                raise ValueError(f"invalid reference sample bounds: {cache}")
            for prefix in range(1, min(raw_len, 100) + 1):
                # Reference mask construction: trajectory_mask[prefix:eol] = 1.
                mask = np.zeros(5000, dtype=bool)
                mask[prefix:eol] = True
                if not np.isfinite(item["future_soh"][mask]).all():
                    raise ValueError(f"nonfinite reference targets: {cache}/{prefix}")
                samples += 1
                points += int(mask.sum())
            for path_ in (cache, raw, raw_data, soh, embedding):
                hashes[str(path_)] = sha256(path_)
        if not names or not samples or not points:
            raise ValueError(f"empty strict {split} cohort")
        counts[split] = {"cells": len(names), "samples": samples, "points": points,
                         "excluded_reference_short": len(excluded)}
    return {"manifest": str(manifest), "split_tag": tag, "counts": counts, "hashes": hashes,
            "reference_loader_sha256": reference_loader_sha256(), "protocol": STRICT_CACHE_VERSION}
