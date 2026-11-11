"""Training-only transforms and fixed-grid labels kept outside the encoder."""
from dataclasses import dataclass, field
from numbers import Real
import numpy as np
import torch
from torch.utils.data import Dataset
from data.feature_provenance import allowed_metadata_fields, audit_metadata, audit_state, check_profile, metadata_whitelists, select_inputs
from data.preprocessing import preprocess, _reference_cycle_features_and_mask


@dataclass
class Record:
    cell_id: str
    domain: str
    inputs: dict
    targets: dict
    audit: dict = field(default_factory=dict)


def apply_early_soh(inputs, early_soh, config, provenance, level):
    """Fill the Level-3 SOH state channel; shared by the raw and multicurve-cache paths."""
    s = config.observed_cycles
    inputs["soh"] = np.zeros((s, 1), np.float32)
    inputs["soh_mask"] = np.zeros((s, 1), bool)
    if early_soh is not None and level in (3, 4):
        audit_state(provenance, "soh", s)
        if np.shape(early_soh) != (s,):
            raise ValueError("early_soh must contain exactly the observed window")
        y = np.asarray(early_soh, np.float32)
        eol_eta = 0.0 if config.protocol == "batterymformer" else config.eta
        if np.any(np.isfinite(y) & (y <= eol_eta)):
            raise ValueError("battery reached EOL in the observed window")
        inputs["soh"][:, 0] = np.nan_to_num(y)
        inputs["soh_mask"][:, 0] = np.isfinite(y) & inputs["cycle_mask"]
    return inputs


def make_record(cell_id, domain, cycles, config, *, metadata=None, early_soh=None,
                provenance=None, label_cycles=(), label_soh=(), total_life=None,
                eol_source="measured", label_definition="", eol_evidence=None,
                eol_tolerance=0, require_full_window=True, reference_aging_embedding=None, level=None):
    """SOH inputs and benchmark labels are deliberately different arguments."""
    if level is None:
        level = max(config.levels)
    if level not in (1, 2, 3, 4, 5):
        raise ValueError("level must be 1, 2, 3, 4 or 5")
    if level in (4, 5) and config.curve_channels < 3:
        raise ValueError("reference levels require the recorded V/I/capacity curve channel")
    if not label_definition:
        raise ValueError("record the SOH/DoD/smoothing/extrapolation label definition")
    provenance = provenance or {}
    s = config.observed_cycles
    cycles = list(cycles)
    has_soc = level in (3, 4) and any(c.soc is not None for c in cycles if c.number <= config.observed_cycles)
    if has_soc:
        audit_state(provenance, "soc", config.observed_cycles)
    inputs, audit = preprocess(cycles, config, include_soc=has_soc,
                               require_full_window=require_full_window)
    inputs["metadata"] = (audit_metadata(metadata or {}, allowed=allowed_metadata_fields(config.metadata_profile))
                          if level >= 2 else {})
    apply_early_soh(inputs, early_soh, config, provenance, level)
    # Level 3 adds the BatteryMFormer cycle-level state (CE/EE); Level 4 routes
    # the same features through the namespaced protocol block instead.
    features = None
    if level >= 3:
        features, feature_mask = _reference_cycle_features_and_mask(cycles, config.observed_cycles)
        inputs["cycle_features"] = features
        inputs["cycle_features_mask"] = feature_mask
    return record_from_inputs(cell_id, domain, inputs, config, input_audit=audit, provenance=provenance,
                              label_cycles=label_cycles, label_soh=label_soh, total_life=total_life,
                              eol_source=eol_source, label_definition=label_definition, eol_evidence=eol_evidence,
                              eol_tolerance=eol_tolerance,
                              reference_cycle_features=features if level == 4 else None,
                              reference_aging_embedding=reference_aging_embedding)


def record_from_inputs(cell_id, domain, inputs, config, *, input_audit=None, provenance=None,
                       label_cycles=(), label_soh=(), total_life=None,
                       eol_source="measured", label_definition="", eol_evidence=None,
                       eol_tolerance=0, reference_cycle_features=None,
                       reference_aging_embedding=None):
    """Targets, validations and audits for already-preprocessed inputs.

    The raw path builds ``inputs`` through :func:`make_record`; the multicurve cache
    (``batteryadapter/data/cache.py``) reassembles the identical input dict from
    ``data/multicurve`` npz files and shares everything below the inputs.
    """
    if not label_definition:
        raise ValueError("record the SOH/DoD/smoothing/extrapolation label definition")
    provenance = provenance or {}
    s = config.observed_cycles
    # Reference protocols share the Adapter's physical tensors.  Their extra
    # fields are kept in a namespaced feature block; no native model bundle is
    # constructed here.
    if "protocol_features" not in inputs and reference_cycle_features is not None:
        inputs["protocol_features"] = {
            "cycle_level_features": np.asarray(reference_cycle_features, np.float32),
            "aging_condition_embedding": np.asarray(
                reference_aging_embedding if reference_aging_embedding is not None
                else np.zeros((1, config.reference_embedding_dim), np.float32), np.float32),
            "soc_input": np.asarray(inputs.get("soc", np.zeros((s, 2, config.points), np.float32)), np.float32),
            "soc_mask": np.asarray(inputs.get("soc_mask", np.zeros((s, 2, config.points), bool)), bool),
            "soh_input": np.asarray(inputs.get("soh", np.zeros((s, 1), np.float32)), np.float32),
            "soh_mask": np.asarray(inputs.get("soh_mask", np.zeros((s, 1), bool)), bool),
            "cycle_level_features_mask": np.isfinite(np.asarray(reference_cycle_features, np.float32)),
            "aging_condition_mask": np.full((1, config.reference_embedding_dim),
                                              reference_aging_embedding is not None, bool),
            "protocol_feature_mask": np.asarray([True, True, True, True,
                                                  reference_aging_embedding is not None], bool),
        }
    if config.protocol == "batterymformer":
        # BatteryMFormer labels are source-normalized around its 0.8 EOL
        # threshold.  Keep the conversion explicit in the cache-compatible
        # record path; the physical input curves remain untouched.
        label_soh = np.asarray(label_soh, float)
        label_soh = np.where(np.isfinite(label_soh), (label_soh - 0.8) / 0.2, label_soh)
    audit = dict(input_audit or {})
    s, h = config.observed_cycles, config.horizon
    n, y = np.asarray(label_cycles, float), np.asarray(label_soh, float)
    if n.ndim != 1 or y.shape != n.shape or not np.isfinite(n).all() or np.any(np.diff(n) <= 0) or np.any(n != np.floor(n)) or np.any(n < 1):
        raise ValueError("labels need increasing, unique integer cycle indices")
    minimum_soh = 0.0 if config.protocol == "batterymformer" else 0.0
    if np.isinf(y).any() or np.any(y[np.isfinite(y)] < minimum_soh):
        raise ValueError("SOH labels must be nonnegative finite values or NaN for missing")
    eol_eta = 0.0 if config.protocol == "batterymformer" else config.eta
    hit = n[np.isfinite(y) & (y <= eol_eta)]
    if hit.size and hit[0] <= s:
        raise ValueError("main task excludes early failures")
    if eol_source not in ("measured", "extrapolated", "censored"):
        raise ValueError("eol_source must be measured, extrapolated or censored")
    if eol_source == "censored" and (total_life is not None or hit.size):
        raise ValueError("censored observations cannot supply a lifetime label or observed crossing")
    supplied = total_life is not None
    if supplied and (isinstance(total_life, bool) or not isinstance(total_life, Real) or not np.isfinite(total_life) or total_life != int(total_life)):
        raise ValueError("supplied EOL must be a finite integer cycle under the declared <= convention")
    if supplied and hit.size and abs(total_life - hit[0]) > eol_tolerance:
        raise ValueError("supplied EOL contradicts the first crossing in the label series "
                         f"beyond the declared convention tolerance ({eol_tolerance})")
    if eol_source == "extrapolated" or (supplied and not hit.size):
        evidence = eol_evidence or {}
        if not isinstance(evidence.get("reference"), str) or not evidence["reference"].strip() or evidence.get("eta") != config.eta or evidence.get("rule") != "<=":
            raise ValueError("external/extrapolated EOL needs a reference, matching eta and <= rule")
        if supplied and not hit.size and np.any((n == total_life) & np.isfinite(y) & (y > config.eta)):
            raise ValueError("SOH at the declared EOL is still above the threshold")
    if total_life is None:
        total_life = hit[0] if hit.size else float("nan")
        if not hit.size:
            eol_source = "censored"
    if np.isfinite(total_life) and total_life <= s:
        raise ValueError("total life must exceed the observation window")
    grid = np.arange(s + 1, s + h + 1, dtype=np.float32)
    future, mask = np.zeros(h, np.float32), np.zeros(h, bool)
    take = (n > s) & (n <= s + h) & np.isfinite(y)
    positions = n[take].astype(int) - s - 1
    future[positions], mask[positions] = y[take], True
    start = y[n == s]
    targets = dict(total_life=np.float32(total_life), future_soh=future, future_mask=mask,
                   trajectory_mask=mask.copy(), query_cycles=grid,
                   start_soh=np.float32(start[0] if start.size else np.nan))
    audit.update(provenance=provenance, label_definition=label_definition, eol_source=eol_source,
                 eol_evidence=eol_evidence or {"reference": "label_series" if hit.size else "no_observed_crossing",
                                               "eta": config.eta, "rule": "<="})
    if config.protocol != "legacy":
        audit.setdefault("protocol", config.protocol)
    return Record(cell_id, domain, inputs, targets, audit)


class Scaler:
    """Global physical-channel scaling, numeric metadata scaling and training vocabulary."""

    def __init__(self, metadata_profile="legacy"):
        check_profile(metadata_profile)
        self.metadata_profile = metadata_profile

    def fit(self, records, *, split, observed_prefix=None):
        if split != "train" or not records:
            raise ValueError("fit requires nonempty training records")
        if observed_prefix is not None and (isinstance(observed_prefix, bool) or
                not isinstance(observed_prefix, int) or not 1 <= observed_prefix <= 100):
            raise ValueError("observed_prefix must be an integer in 1..100")
        numeric_names, categorical_names = metadata_whitelists(self.metadata_profile)
        self.stats, self.vocab = {}, {}
        self.protocols = {r.audit.get("protocol") for r in records if r.audit.get("protocol")}
        self.protocol = next(iter(self.protocols)) if len(self.protocols) == 1 else None
        for name in ("curves", "capacity"):
            mask_name = "curve_mask" if name == "curves" else "capacity_mask"
            x = np.stack([r.inputs[name] for r in records])
            m = np.stack([r.inputs[mask_name] for r in records]) & np.isfinite(x)
            if observed_prefix is not None:
                m[:, observed_prefix:] = False
            axes = (0, 1, 4) if name == "curves" else (0, 1)
            count = m.sum(axis=axes)
            mean = np.where(m, x, 0).sum(axis=axes) / np.maximum(count, 1)
            spread = mean[..., None] if name == "curves" else mean
            var = np.where(m, (x - spread) ** 2, 0).sum(axis=axes) / np.maximum(count, 1)
            std = var ** .5
            self.stats[name] = (mean.tolist(), np.where(std > 1e-8, std, 1.).tolist())
        if self.protocol == "batterylife":
            values = np.asarray([r.targets.get("total_life", np.nan) for r in records], float)
            values = values[np.isfinite(values)]
            mean = float(values.mean()) if values.size else 0.0
            std = float(values.std()) if values.size else 1.0
            self.stats["total_life"] = (mean, std if std > 1e-8 else 1.0)
        for name in numeric_names:
            values = []
            for r in records:
                if name in r.inputs.get("metadata", {}):
                    value = np.asarray(r.inputs["metadata"][name], float)
                    if observed_prefix is not None and value.shape == (r.inputs["capacity"].shape[0],):
                        value = value[:observed_prefix]
                    values.append(value.reshape(-1))
            v = np.concatenate(values) if values else np.array([])
            v = v[np.isfinite(v)]
            std = float(v.std()) if v.size else 0.
            self.stats[name] = (float(v.mean()) if v.size else 0., std if std > 1e-8 else 1.)
        for name in categorical_names:
            categories = sorted({str(r.inputs.get("metadata", {}).get(name)) for r in records
                                 if r.inputs.get("metadata", {}).get(name) is not None})
            self.vocab[name] = {value: i + 1 for i, value in enumerate(categories)}
        return self

    def state_dict(self):
        return dict(stats=self.stats, vocab=self.vocab, protocol=getattr(self, "protocol", None),
                    metadata_profile=self.metadata_profile)

    @classmethod
    def from_state_dict(cls, state):
        result = cls(state.get("metadata_profile", "legacy"))
        result.stats, result.vocab = state["stats"], state["vocab"]
        result.protocol = state.get("protocol")
        result.protocols = {result.protocol} if result.protocol else set()
        return result

    def transform(self, record, level):
        if not hasattr(self, "stats"):
            raise RuntimeError("fit the scaler on training data first")
        source = select_inputs(record.inputs, level)
        out = {k: np.array(v, copy=True) for k, v in source.items() if k not in ("metadata", "metadata_mask")}
        for key in ("curves", "capacity"):
            mean, std = (np.asarray(v) for v in self.stats[key])
            if key == "curves":
                mean, std = mean[..., None], std[..., None]
            mask = source["curve_mask" if key == "curves" else "capacity_mask"]
            if self.protocol in ("batterymformer", "batterylife"):
                # Protocol extractors already applied their source-specific
                # normalization; applying the legacy global scaler would erase
                # that contract and mix the two benchmarks.
                out[key] = np.where(mask, source[key], 0).astype(np.float32)
            else:
                out[key] = np.where(mask, (source[key] - mean) / std, 0).astype(np.float32)
        if level >= 2:
            s = source["capacity"].shape[0]
            allowed_fields = allowed_metadata_fields(self.metadata_profile)
            _, categorical_names = metadata_whitelists(self.metadata_profile)
            metadata = audit_metadata(source.get("metadata", {}), allowed=allowed_fields)
            out["metadata"], out["metadata_mask"] = {}, {}
            for name in allowed_fields:
                value = metadata.get(name)
                if name in categorical_names:
                    out["metadata"][name] = np.int64(self.vocab[name].get(str(value), 0))
                    out["metadata_mask"][name] = np.asarray(value is not None, bool)
                else:
                    if value is not None and np.shape(value) not in ((), (s,)):
                        raise ValueError(f"{name} must be a scalar or one value per early cycle")
                    a = np.broadcast_to(np.asarray(np.nan if value is None else value, np.float32), (s,))
                    mean, std = self.stats[name]
                    out["metadata"][name] = np.where(np.isfinite(a), (a - mean) / std, 0).astype(np.float32)
                    out["metadata_mask"][name] = np.isfinite(a)
        if level in (4, 5):
            features = source.get("protocol_features")
            if level == 4 and features is None:
                raise ValueError("Level 4 requires extracted BatteryMFormer protocol_features")
            if features is not None:
                out["protocol_features"] = tensors_numpy(features)
        return out


def tensors_numpy(tree):
    """Copy a protocol feature tree at training precision (float32) without
    applying legacy metadata scaling.  Bool masks keep their dtype."""
    if isinstance(tree, dict):
        return {key: tensors_numpy(value) for key, value in tree.items()}
    array = np.array(tree, copy=True)
    if array.dtype.kind == "f" and array.dtype != np.float32:
        array = array.astype(np.float32)
    return array


def tensors(tree):
    return {k: tensors(v) if isinstance(v, dict) else torch.as_tensor(v) for k, v in tree.items()}


class BatteryDataset(Dataset):
    def __init__(self, records, scaler, level=3):
        self.records, self.scaler, self.level = records, scaler, level

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        targets = dict(r.targets)
        if self.scaler.protocol == "batterylife" and "total_life" in targets:
            mean, std = self.scaler.stats.get("total_life", (0.0, 1.0))
            value = np.asarray(targets["total_life"], np.float32)
            targets["total_life_scaled"] = np.where(np.isfinite(value), (value - mean) / std, value).astype(np.float32)
        return dict(inputs=tensors(self.scaler.transform(r, self.level)), targets=tensors(targets),
                    cell_id=r.cell_id, domain=r.domain, eol_source=r.audit["eol_source"])


class BMFPrefixDataset(BatteryDataset):
    """Lazy reference windows, preserving one underlying record per cell.

    ``prefixes="sweep"`` enumerates every 1..min(100, raw length) window, the
    reference's early-prediction evaluation protocol.  ``prefixes="observed"``
    yields one window per cell at its longest observed prefix: that is the
    training/validation sample, because training the sweep multiplies each
    epoch by up to 100x without adding input information (the masks already
    gate every cycle beyond the observed window). ``prefixes="spaced50"``
    yields 50 deterministic, evenly spaced windows across each cell's available
    prefix range. Numeric prefix modes use a fixed stride anchored at the
    available end; prefix1 selects only that end.
    """
    PREFIX_MODES = ("sweep", "observed", "spaced50", "prefix1", "prefix10",
                    "prefix20", "prefix50", "prefix100")
    SPACED_PREFIX_COUNT = 50

    @classmethod
    def spaced_prefix_indices(cls, end):
        if end < cls.SPACED_PREFIX_COUNT:
            raise ValueError(
                f"spaced50 requires at least {cls.SPACED_PREFIX_COUNT} available prefixes; got {end}")
        indices = np.rint(np.linspace(1, end, cls.SPACED_PREFIX_COUNT)).astype(int).tolist()
        if len(set(indices)) != cls.SPACED_PREFIX_COUNT:
            raise ValueError(f"could not select {cls.SPACED_PREFIX_COUNT} unique prefixes from 1..{end}")
        return indices

    def __init__(self, records, scaler, level=4, *, prefixes="sweep", observed_cap=None):
        if level != 4 or scaler.protocol != "batterymformer":
            raise ValueError("strict prefix evaluation requires the BatteryMFormer Level 4 protocol")
        if prefixes not in self.PREFIX_MODES:
            raise ValueError(f"prefixes must be one of {self.PREFIX_MODES}")
        super().__init__(records, scaler, level)
        self.prefixes = prefixes
        # Windows never extend past the training observation window: a model
        # retrained on a shorter prefix must not receive longer inputs.
        self.observed_cap = 100 if observed_cap is None else int(observed_cap)
        if not 1 <= self.observed_cap <= 100:
            raise ValueError("observed_cap must be an integer in 1..100")
        self.index = []
        self.prefix_indices_by_cell = {}
        for i, r in enumerate(records):
            raw_len = r.audit.get("raw_cycle_data_length")
            soh_len = r.audit.get("processed_soh_length")
            if raw_len is None or soh_len is None or not 100 < soh_len <= 5000 or raw_len < 1:
                raise ValueError(f"missing or invalid strict reference lengths for {r.cell_id}")
            end = min(self.observed_cap, raw_len)
            if prefixes == "observed":
                selected = [end]
            elif prefixes == "spaced50":
                selected = self.spaced_prefix_indices(end)
            elif prefixes == "prefix1":
                selected = [end]
            elif prefixes.startswith("prefix"):
                stride = {"prefix10": 10, "prefix20": 5, "prefix50": 2, "prefix100": 1}[prefixes]
                selected = list(range(end, 0, -stride))[::-1]
            else:
                selected = list(range(1, end + 1))
            self.prefix_indices_by_cell[r.cell_id] = selected
            self.index.extend((i, k) for k in selected)

    def __len__(self):
        return len(self.index)

    def select_prefix(self, prefix):
        """Restrict evaluation to one absolute observed-cycle count."""
        if isinstance(prefix, bool) or not isinstance(prefix, int) or not 1 <= prefix <= 100:
            raise ValueError("evaluation prefix must be an integer in 1..100")
        self.index = [(i, k) for i, k in self.index if k == prefix]
        if not self.index:
            raise ValueError(f"no cells have evaluation prefix {prefix}")

    def __getitem__(self, index):
        i, k = self.index[index]
        r = self.records[i]
        inputs = self.scaler.transform(r, self.level)
        for name in ("cycle_mask", "curve_mask", "capacity_mask"):
            inputs[name][k:] = False
        inputs["curves"][k:] = 0
        inputs["capacity"][k:] = 0
        features = inputs["protocol_features"]
        for name in ("soc_mask", "soh_mask", "cycle_level_features_mask"):
            features[name][k:] = False
        for name in ("soc_input", "soh_input", "cycle_level_features"):
            features[name][k:] = 0
        # The endpoint is separate from curve-validity masks: the reference
        # may retain an observed, zero-filled cycle when parsing fails.
        features["observed_end"] = np.asarray(k, dtype=np.int64)
        targets = dict(r.targets)
        mask = np.zeros_like(targets["future_soh"], dtype=bool)
        mask[k:r.audit["processed_soh_length"]] = True
        targets["future_mask"] = mask
        targets["trajectory_mask"] = mask.copy()
        targets["start_soh"] = np.asarray(targets["future_soh"][k-1], np.float32)
        return dict(inputs=tensors(inputs), targets=tensors(targets), cell_id=r.cell_id,
                    domain=r.domain, eol_source=r.audit["eol_source"], prefix=k)


def collate(records):
    def stack(values):
        if isinstance(values[0], dict):
            if any(set(v) != set(values[0]) for v in values):
                raise ValueError("use one declared input level and fixed query grid per batch")
            return {k: stack([v[k] for v in values]) for k in values[0]}
        return torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    if not records:
        raise ValueError("empty batch")
    return stack(records)


def check_splits(splits):
    seen = {}
    for split, records in splits.items():
        for r in records:
            key = (r.domain, r.cell_id)
            if key in seen:
                raise ValueError(f"duplicate battery {key} in {seen[key]} and {split}")
            seen[key] = split
