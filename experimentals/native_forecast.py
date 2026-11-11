"""Native TSFM zero-shot and LoRA experiments on the strict BMF prefix cache.

The prefix is the only observed input.  Future SOH is used solely as a training
label or by the evaluator; unknown future covariates are always masked.  This
runner intentionally does not modify the BatteryAdapter input path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MODELS = ("chronos2", "moirai2", "moment", "sundial", "toto1")
TARGETS = {"chronos2": "all-linear", "sundial": ["q_proj", "k_proj", "v_proj", "o_proj"],
           "moirai2": ["q_proj", "k_proj", "v_proj", "self_attn.out_proj"], "moment": ["q", "k", "v", "o"],
           "toto1": ["wQKV", "wO"]}
FEATURE_VERSION = "strict-prefix-curve4-soc4-ce-ee-v1"


def family(name):
    return name


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _pooled(values, mask, bins=4):
    """Per-cycle, per-phase masked means; no other cycle is inspected."""
    values, mask = np.asarray(values), np.asarray(mask, dtype=bool)
    columns = []
    for value_bin, mask_bin in zip(np.array_split(values, bins, axis=-1), np.array_split(mask, bins, axis=-1)):
        valid = mask_bin & np.isfinite(value_bin)
        count = valid.sum(axis=-1)
        total = np.where(valid, value_bin, 0).sum(axis=-1)
        columns.append(np.divide(total, count, out=np.full(total.shape, np.nan, dtype=np.float64), where=count > 0))
    return np.stack(columns, axis=-1).reshape(values.shape[0], -1).astype(np.float32)


def sample_inputs(sample, model, static=None):
    """Return SOH [k] and optional observed-only covariates [features,k]."""
    k = int(sample["prefix"])
    inputs = sample["inputs"]
    pf = inputs["protocol_features"]
    history = _numpy(pf["soh_input"])[:k, 0].astype(np.float32, copy=True)
    observed = _numpy(pf["soh_mask"])[:k, 0].astype(bool)
    history[~observed] = np.nan
    if len(history) != k or not np.isfinite(history).all():
        raise ValueError("strict native prediction requires a finite observed SOH prefix")
    if family(model) in ("sundial", "moment"):
        return history, None
    curves = _numpy(inputs["curves"])[:k]
    curve_mask = _numpy(inputs["curve_mask"])[:k]
    soc = _numpy(pf["soc_input"])[:k].reshape(k, 2, -1)
    soc_mask = _numpy(pf["soc_mask"])[:k].reshape(k, 2, -1)
    cycle = _numpy(pf["cycle_level_features"])[:k].astype(np.float32, copy=True)
    cycle_mask = _numpy(pf["cycle_level_features_mask"])[:k].astype(bool)
    cycle[~cycle_mask] = np.nan
    features = np.concatenate((_pooled(curves, curve_mask), _pooled(soc, soc_mask), cycle), axis=1)
    if static is not None:
        features = np.concatenate((features, np.broadcast_to(np.asarray(static, np.float32), (k, len(static)))), axis=1)
    return history, features.T.copy()


def future_labels(sample, length):
    k = int(sample["prefix"])
    values = _numpy(sample["targets"]["future_soh"])[k:k + length].astype(np.float32, copy=True)
    mask = _numpy(sample["targets"]["trajectory_mask"])[k:k + length].astype(bool, copy=True)
    if len(values) < length:
        values = np.pad(values, (0, length - len(values)))
        mask = np.pad(mask, (0, length - len(mask)))
    mask &= np.isfinite(values)
    if not mask.any():
        raise ValueError("prefix has no valid future SOH labels")
    values[~mask] = 0
    return values, mask


def fit_static(records, dimensions=8):
    """Fixed train-only PCA; the actual native-model sensitivity is probed separately."""
    vectors = []
    for record in records:
        fields = record.inputs["protocol_features"]
        value = np.asarray(fields["aging_condition_embedding"], dtype=np.float64).reshape(-1)
        mask = np.asarray(fields["aging_condition_mask"], dtype=bool).reshape(-1)
        if not mask.all() or not np.isfinite(value).all():
            return None
        vectors.append(value)
    if len(vectors) < 2:
        return None
    matrix = np.stack(vectors)
    mean = matrix.mean(axis=0)
    _, singular, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    keep = min(dimensions, int((singular > 1e-10).sum()))
    if not keep:
        return None
    components = vt[:keep].copy()
    # Make the otherwise arbitrary singular-vector signs reproducible.
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    scales = ((matrix - mean) @ components.T).std(axis=0)
    return {"mean": mean.astype(np.float32), "components": components.astype(np.float32),
            "scales": scales.astype(np.float32)}


def static_values(sample, projection):
    value = _numpy(sample["inputs"]["protocol_features"]["aging_condition_embedding"]).reshape(-1)
    return (value - projection["mean"]) @ projection["components"].T


def _quantile_loss(prediction, future, mask, levels):
    target = torch.as_tensor(future, dtype=prediction.dtype, device=prediction.device)
    valid = torch.as_tensor(mask, dtype=torch.bool, device=prediction.device)
    q = torch.as_tensor(levels, dtype=prediction.dtype, device=prediction.device)
    error = target[:, None] - prediction
    loss = torch.maximum(q * error, (q - 1) * error)
    return loss[valid].mean()


class _NativeHolder(nn.Module):
    """Keep custom model APIs intact when PEFT injects attention adapters."""
    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, *args, **kwargs):
        return self.native(*args, **kwargs)


class NativePredictor:
    def __init__(self, model, model_path, device, mode="lora", prediction=None):
        if model == "moment" and mode == "zero-shot":
            raise ValueError("MOMENT 原生零样本预测接口尚未验证；不使用重建后线性外推替代")
        from backbones.base import specification

        spec = specification(model)
        self.name, self.kind, self.device = model, family(model), torch.device(device)
        self.mode, self.prediction = mode, prediction or {}
        self.revision, self.model_id = spec["revision"], spec["model_id"]
        candidate = Path(model_path)
        if candidate.is_dir():
            self.path = str(candidate.resolve())
            weight = candidate / Path(spec["weight_file"]).name
            digest = hashlib.sha256()
            with weight.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != spec["weight_sha256"]:
                raise ValueError(f"{weight} does not match the pinned model checksum")
        elif str(model_path) == self.model_id:
            self.path = self.model_id
        else:
            raise ValueError(f"model path must be the pinned {self.model_id} or a verified local directory")
        if self.kind == "chronos2":
            from chronos import Chronos2Pipeline
            self.engine = Chronos2Pipeline.from_pretrained(self.path, revision=self.revision, device_map=device)
            self.native = self.engine.model
        elif self.kind == "toto1":
            from toto.model.toto import Toto
            from toto.inference.forecaster import TotoForecaster
            self.native = Toto.from_pretrained(self.path, revision=self.revision).model.to(self.device)
            self.engine = TotoForecaster(self.native)
        elif self.kind == "moirai2":
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
            self.native = Moirai2Module.from_pretrained(self.path, revision=self.revision).to(self.device)
            self.engine = Moirai2Forecast(module=self.native, prediction_length=1, context_length=100,
                target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to(self.device)
        elif self.kind == "moment":
            from momentfm import MOMENTPipeline
            self.native = MOMENTPipeline.from_pretrained(self.path, revision=self.revision,
                model_kwargs=({"task_name": "reconstruction"} if mode == "zero-shot" else
                              {"task_name": "forecasting", "forecast_horizon": 64}))
            self.native.init()
            self.native = self.native.to(self.device)
            self.engine = None
        else:
            from transformers import AutoModelForCausalLM
            self.native = AutoModelForCausalLM.from_pretrained(self.path, revision=self.revision,
                code_revision=self.revision, trust_remote_code=True).to(self.device)
            self.engine = None
        self.native.eval()
        self.adapter = None

    def add_lora(self, rank=16, alpha=32, dropout=.05):
        from peft import LoraConfig, get_peft_model, inject_adapter_in_model
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                            bias="none", target_modules=TARGETS[self.kind])
        if self.kind == "chronos2":
            # This is the existing Chronos native LoRA configuration.
            self.native = get_peft_model(self.native, config)
            self.engine.model = self.native
            self.adapter = self.native
        else:
            self.adapter = inject_adapter_in_model(config, _NativeHolder(self.native))
        self.adapter.eval()
        if self.kind == "moment":
            self.native.head.requires_grad_(True)
        parameters = [(name, p) for name, p in self.adapter.named_parameters() if p.requires_grad]
        if not any("lora_" in name for name, _ in parameters) or any(
            "lora_" not in name and not (self.kind == "moment" and name.startswith("native.head."))
            for name, _ in parameters
        ):
            raise RuntimeError("LoRA injection did not freeze the base or created no adapter parameters")
        for module in self.adapter.modules():
            dropout_layer = getattr(module, "lora_dropout", None)
            if isinstance(dropout_layer, nn.ModuleDict):
                dropout_layer.train()

    def save_adapter(self, directory, metadata, projection):
        from peft import get_peft_model_state_dict
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        state = get_peft_model_state_dict(self.adapter)
        if not state or any("lora_" not in key for key in state):
            raise RuntimeError("adapter checkpoint is empty or contains base weights")
        torch.save(state, directory / "adapter.pt")
        if self.kind == "moment":
            torch.save(self.native.head.state_dict(), directory / "forecast_head.pt")
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        if projection is not None:
            np.savez(directory / "static_projection.npz", **projection)

    def load_adapter(self, directory):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (meta["model"] != self.name or meta["base_model"] != self.path
                or meta["revision"] != self.revision or meta["feature_version"] != FEATURE_VERSION):
            raise ValueError("adapter model, base checkpoint or feature contract differs")
        self.add_lora(**meta["lora"])
        state = torch.load(directory / "adapter.pt", map_location="cpu", weights_only=True)
        expected = set(get_peft_model_state_dict(self.adapter))
        if set(state) != expected:
            raise ValueError("adapter checkpoint has missing or unexpected adapter weights")
        result = set_peft_model_state_dict(self.adapter, state)
        if not state or any("lora_" not in key for key in state) or any("lora_" in key for key in result.missing_keys):
            raise ValueError("adapter checkpoint is incomplete or contains base weights")
        if not expected:
            raise ValueError("no LoRA weights in initialized model")
        if self.kind == "moment":
            self.native.head.load_state_dict(torch.load(directory / "forecast_head.pt", map_location="cpu", weights_only=True))
        self.adapter.eval()
        projection = None
        if meta["static_conditioning"]["enabled"]:
            with np.load(directory / "static_projection.npz", allow_pickle=False) as data:
                projection = {key: data[key] for key in data.files}
        return meta, projection


    def _moirai(self, history, cov, length):
        history = np.asarray(history, np.float32)[-100:]
        target = torch.as_tensor(history[None, :, None], device=self.device)
        kwargs = {}
        if cov is not None:
            past = torch.as_tensor(np.asarray(cov, np.float32)[:, -len(history):].T[None], device=self.device)
            observed = torch.isfinite(past)
            kwargs = {"past_feat_dynamic_real": torch.nan_to_num(past, nan=0.0),
                      "past_observed_feat_dynamic_real": observed}
        with self.engine.hparams_context(context_length=len(history), prediction_length=length,
                                         past_feat_dynamic_real_dim=0 if cov is None else len(cov)):
            return self.engine(past_target=target, past_observed_target=torch.isfinite(target),
                               past_is_pad=torch.zeros((1, len(history)), dtype=torch.bool, device=self.device),
                               **kwargs)[0].transpose(0, 1)

    def _moment(self, history):
        history = np.asarray(history, np.float32)[-100:]
        values = torch.as_tensor(history, device=self.device)[None, None, :]
        mask = torch.ones((1, len(history)), dtype=torch.long, device=self.device)
        values = F.pad(values, (self.native.seq_len - len(history), 0))
        mask = F.pad(mask, (self.native.seq_len - len(history), 0))
        return self.native.forecast(x_enc=values, input_mask=mask).forecast[0, 0]

    def _toto1_inputs(self, history, cov):
        rows = np.asarray(history, np.float32)[None, :]
        if cov is not None:
            rows = np.concatenate((rows, cov), axis=0)
        observed = np.isfinite(rows)
        patch = int(self.native.patch_embed.patch_size)
        pad = (-rows.shape[-1]) % patch
        values = F.pad(torch.as_tensor(np.nan_to_num(rows), device=self.device)[None], (pad, 0))
        mask = F.pad(torch.as_tensor(observed, device=self.device)[None], (pad, 0), value=False)
        return values, mask, patch

    def loss(self, sample, projection=None):
        static = static_values(sample, projection) if projection is not None else None
        history, cov = sample_inputs(sample, self.name, static)
        if self.kind == "toto1":
            from toto.model.losses import CombinedLoss
            values, observed, patch = self._toto1_inputs(history, cov)
            future, mask = future_labels(sample, patch)
            out = self.native(values, observed, torch.zeros_like(values, dtype=torch.long))
            targets = torch.zeros_like(values)
            targets[0, 0, -patch:] = torch.as_tensor(future, device=self.device)
            scaled = (targets - out.loc) / (out.scale + torch.finfo(values.dtype).eps)
            losses = CombinedLoss()(out.distribution, scaled)[0, 0, -patch:]
            return losses[torch.as_tensor(mask, device=self.device)].mean()
        if self.kind == "chronos2":
            horizon = 64
            future, mask = future_labels(sample, horizon)
            rows = np.concatenate((history[None, :], cov), axis=0)
            context = torch.as_tensor(rows, device=self.device)
            labels = torch.zeros((len(rows), horizon), dtype=context.dtype, device=self.device)
            masks = torch.zeros_like(labels, dtype=torch.bool)
            labels[0] = torch.as_tensor(future, device=self.device)
            masks[0] = torch.as_tensor(mask, device=self.device)
            patches = math.ceil(horizon / int(self.native.chronos_config.output_patch_size))
            return self.native(context=context, group_ids=torch.zeros(len(rows), dtype=torch.long, device=self.device),
                               num_output_patches=patches, future_target=labels, future_target_mask=masks).loss
        if self.kind == "moirai2":
            horizon = self.native.patch_size * self.native.num_predict_token
            future, mask = future_labels(sample, horizon)
            return _quantile_loss(self._moirai(history, cov, horizon), future, mask, self.native.quantile_levels)
        if self.kind == "moment":
            future, mask = future_labels(sample, 64)
            pred = self._moment(history)
            target = torch.as_tensor(future, dtype=pred.dtype, device=self.device)
            valid = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            return (pred[valid] - target[valid]).square().mean()
        horizon = int(self.native.config.output_token_lens[-1])
        future, mask = future_labels(sample, horizon)
        x = torch.as_tensor(history, dtype=torch.float32, device=self.device)[None, :]
        mean, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True, unbiased=False) + 1e-5
        x = (x - mean) / std
        target = torch.as_tensor(future, dtype=x.dtype, device=self.device)[None, :]
        valid = torch.as_tensor(mask, dtype=torch.bool, device=self.device)[None, :]
        target = torch.where(valid, (target - mean) / std, 0)
        hidden = self.native.model(input_ids=x, use_cache=False, return_dict=True).last_hidden_state[:, -1, :]
        repeat = int(self.native.config.diffusion_batch_mul)
        return self.native.flow_loss(target.repeat(repeat, 1), hidden.repeat(repeat, 1),
                                     torch.ones(repeat, device=self.device), valid.repeat(repeat, 1))

    def block(self, history, cov, length):
        with torch.inference_mode():
            if self.kind == "chronos2":
                item = {"target": np.asarray(history, np.float32)[None, :]}
                if cov is not None:
                    item["past_covariates"] = {f"feat_{i}": row for i, row in enumerate(cov)}
                quantiles, _ = self.engine.predict_quantiles([item], prediction_length=length, batch_size=1)
                return _numpy(quantiles[0])[0, :, 4]
            if self.kind == "toto1":
                from toto.data.util.dataset import MaskedTimeseries
                values, mask, _ = self._toto1_inputs(history, cov)
                n, length_in = values.shape[1:]
                times = torch.arange(length_in, device=self.device)[None, None, :].expand(1, n, -1)
                inputs = MaskedTimeseries(series=values, padding_mask=mask,
                    id_mask=torch.zeros_like(values, dtype=torch.long), timestamp_seconds=times,
                    time_interval_seconds=torch.ones((1, n), device=self.device, dtype=torch.long))
                result = self.engine.forecast(inputs, prediction_length=length,
                    num_samples=self.prediction["num_samples"], samples_per_batch=self.prediction["samples_per_batch"],
                    use_kv_cache=True)
                return _numpy(result.quantile(.5))[0, 0, :length]
            if self.kind == "moirai2":
                return _numpy(self._moirai(history, cov, length)[:, self.native.quantile_levels.index(.5)])
            if self.kind == "moment":
                return _numpy(self._moment(history)[:length])
            x = torch.as_tensor(history, dtype=torch.float32, device=self.device)[None, :]
            count = self.prediction["num_samples"]
            samples = self.native.generate(x, max_new_tokens=length, num_samples=count)
            values = _numpy(samples)
            if values.ndim == 3 and values.shape[1] == count:
                values = values[0]
            elif values.ndim == 3 and values.shape[2] == count:
                values = values[0].T
            else:
                raise ValueError(f"unexpected Sundial sample shape {values.shape}")
            return np.median(values[:, :length], axis=0)

    def predict(self, history, cov, length):
        history = np.asarray(history, np.float32).copy()
        cov = None if cov is None else np.asarray(cov, np.float32).copy()
        pieces = []
        while length:
            limit = getattr(self, "prediction", {}).get("block_length", 320)
            take = min(length, limit, self.native.patch_size * self.native.num_predict_token if self.kind == "moirai2" else
                       64 if self.kind == "moment" and self.mode != "zero-shot" else limit)
            window = history[-100:]
            c = None if cov is None else cov[:, -len(window):]
            block = np.asarray(self.block(window, c, take), dtype=np.float32).reshape(-1)
            if len(block) != take or not np.isfinite(block).all():
                raise FloatingPointError("native forecast is nonfinite or has the wrong horizon")
            pieces.append(block)
            history = np.concatenate((history, block))
            if cov is not None:
                cov = np.concatenate((cov, np.full((len(cov), take), np.nan, np.float32)), axis=1)
            length -= take
        return np.concatenate(pieces)


def probe_static(predictor, dataset, projection, tolerance=1e-6):
    """Only enable compressed static inputs if the base responds at short and long prefixes."""
    if projection is None or predictor.kind in ("sundial", "moment"):
        return {"enabled": False, "reason": "no variable train-only static projection or SOH-only model", "deltas": []}
    record = dataset.records[0]
    end = min(100, int(record.audit["raw_cycle_data_length"]))
    indices = []
    for k in sorted(set((1, min(10, end), end))):
        index = next((i for i, pair in enumerate(dataset.index) if pair == (0, k)), None)
        if index is None:
            return {"enabled": False, "reason": "probe requires a full prefix sweep", "deltas": []}
        indices.append(index)
    deltas = []
    for index in indices:
        sample = dataset[index]
        value = static_values(sample, projection)
        history, cov = sample_inputs(sample, predictor.name, value)
        changed = value + np.maximum(projection["scales"], 1e-4)
        _, altered = sample_inputs(sample, predictor.name, changed)
        first = predictor.block(history, cov, 1)
        second = predictor.block(history, altered, 1)
        delta = float(np.max(np.abs(np.asarray(first) - np.asarray(second))))
        if not np.isfinite(delta):
            raise FloatingPointError("nonfinite static-conditioning probe")
        deltas.append({"prefix": int(sample["prefix"]), "max_abs_delta": delta})
    enabled = all(item["max_abs_delta"] > tolerance for item in deltas)
    return {"enabled": enabled, "reason": "passed perturbation probe" if enabled else "constant-channel conditioning was ineffective",
            "tolerance": tolerance, "deltas": deltas}


def train(predictor, dataset, projection, epochs, learning_rate, seed):
    generator = np.random.default_rng(seed)
    named = [(name, p) for name, p in predictor.adapter.named_parameters() if p.requires_grad]
    groups = {"LoRA": [p for name, p in named if "lora_" in name]}
    if predictor.kind == "moment":
        groups["forecast head"] = [p for name, p in named if name.startswith("native.head.")]
    if any(not group for group in groups.values()):
        raise ValueError("missing trainable LoRA or forecast head parameters")
    params = [p for _, p in named]
    initial = {name: [p.detach().clone() for p in group] for name, group in groups.items()}
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    losses = []
    nonzero_gradient = {name: False for name in groups}
    for _ in range(epochs):
        for index in generator.permutation(len(dataset)):
            optimizer.zero_grad(set_to_none=True)
            loss = predictor.loss(dataset[int(index)], projection)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite native LoRA training loss")
            loss.backward()
            grads = [p.grad for p in params if p.grad is not None]
            if not grads or not all(torch.isfinite(grad).all() for grad in grads):
                raise FloatingPointError("missing or nonfinite native LoRA gradients")
            for name, group in groups.items():
                nonzero_gradient[name] |= any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in group)
            torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    if not losses:
        raise ValueError("no native LoRA training samples")
    for name, group in groups.items():
        if not nonzero_gradient[name] or not any(
            not torch.equal(before, after.detach()) for before, after in zip(initial[name], group)
        ):
            raise RuntimeError(f"native {name} received no nonzero gradient or did not update")
    predictor.adapter.eval()
    return losses


def evaluate(predictor, dataset, projection):
    from data.protocol_extractors.common import source_eta
    abs_sum = rel_sum = 0.0
    n_mae = n_mape = 0
    seen = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        key = (str(sample["domain"]), str(sample["cell_id"]), int(sample["prefix"]))
        if key in seen:
            raise ValueError(f"duplicate strict prefix: {key}")
        seen.add(key)
        static = static_values(sample, projection) if projection is not None else None
        history, cov = sample_inputs(sample, predictor.name, static)
        k = int(sample["prefix"])
        prediction = predictor.predict(history, cov, 5000 - k)
        target = _numpy(sample["targets"]["future_soh"])[k:].astype(np.float64)
        mask = _numpy(sample["targets"]["trajectory_mask"])[k:].astype(bool)
        eta = source_eta(sample["domain"])
        truth = target[mask] * (1 - eta) + eta
        pred = prediction[mask].astype(np.float64) * (1 - eta) + eta
        if not np.isfinite(truth).all() or not np.isfinite(pred).all():
            raise FloatingPointError("nonfinite strict prediction or target")
        error = np.abs(pred - truth)
        relative = truth > .01
        abs_sum += float(error.sum())
        rel_sum += float((error[relative] / truth[relative]).sum())
        n_mae += int(error.size)
        n_mape += int(relative.sum())
    if len(seen) != len(dataset) or not n_mae or not n_mape:
        raise ValueError("incomplete strict prefix evaluation")
    return {"mae": abs_sum / n_mae, "mape": 100 * rel_sum / n_mape,
            "mae_abs_sum": abs_sum, "mae_count": n_mae,
            "mape_relative_sum": rel_sum, "mape_count": n_mape,
            "sample_count": len(seen), "aggregation": "all-valid-points", "protocol": "bmf-strict-prefix-v1"}


def main(argv=None, *, mode=None):
    from config import Config
    from data.dataset import BMFPrefixDataset, Scaler
    from data.protocol_cache import load_protocol_cache
    from data.protocol_extractors.bmf_strict import OFFICIAL_FOLDS, STRICT_CACHE_VERSION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("zero-shot", "lora"), default=mode, required=mode is None)
    parser.add_argument("--allow-source-protocol", action="store_true",
                        help="明确采用来源实现的原生输入/训练配方，不表示论文相同输入协议已验证")
    parser.add_argument("--adapter", type=Path, help="load an existing adapter instead of training")
    parser.add_argument("--config", type=Path, help="model config containing the native experiment settings")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int, help="training seed; defaults to the official fold seed")
    parser.add_argument("--train-prefixes", choices=BMFPrefixDataset.PREFIX_MODES, default="observed")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--eval-prefix", type=int, choices=range(1, 101),
                        help="score one observed prefix, beginning at the next cycle")
    args = parser.parse_args(argv)
    if mode is not None and args.mode != mode:
        parser.error("不能通过 --mode 改变该入口的预测方式")
    if not args.allow_source_protocol:
        raise ValueError("原生预测的论文相同输入协议未验证；仅在接受来源配方时显式传 --allow-source-protocol")
    if args.model == "moment" and args.mode == "zero-shot":
        raise ValueError("MOMENT 原生零样本预测接口尚未验证；不使用重建后线性外推替代")
    config_path = args.config or Path(__file__).resolve().parents[1] / "configs/models" / f"{args.model}.json"
    native = json.loads(config_path.read_text(encoding="utf-8"))["native"]
    if args.epochs is None:
        args.epochs = native.get("epochs")
    if args.learning_rate is None:
        args.learning_rate = native.get("learning_rate")
    prediction = native.get("prediction", {})
    if not isinstance(prediction.get("block_length"), int) or prediction["block_length"] < 1:
        parser.error("native.prediction.block_length must be a positive integer")
    if args.model == "toto1" and not {"num_samples", "samples_per_batch"} <= set(prediction):
        parser.error("Toto1 requires native.prediction num_samples and samples_per_batch")
    if args.model == "sundial" and "num_samples" not in prediction:
        parser.error("Sundial requires native.prediction.num_samples")
    if any(not isinstance(prediction.get(key), int) or prediction[key] < 1
           for key in (("num_samples", "samples_per_batch") if args.model == "toto1" else
                       ("num_samples",) if args.model == "sundial" else ())):
        parser.error("native prediction sample counts must be positive integers")
    if ((args.mode == "lora" and not args.adapter and
         (args.epochs is None or args.epochs < 1 or args.learning_rate is None or args.learning_rate <= 0))
            or (args.mode == "zero-shot" and args.adapter)):
        parser.error("invalid epochs, learning rate or adapter for zero-shot")
    if args.output.exists():
        raise FileExistsError("choose a new output directory")
    cfg = Config(points=150, horizon=5000, levels=(4,), input_protocol="batterymformer")
    splits, protocol = load_protocol_cache(args.manifest, cfg, args.cache_root)
    if protocol.get("evaluation_protocol") != STRICT_CACHE_VERSION:
        raise ValueError("strict native experiments require the strict reference cache")
    if protocol.get("split_tag") not in OFFICIAL_FOLDS:
        raise ValueError("strict native experiments require an official BatteryMFormer fold")
    if args.seed is None:
        args.seed = OFFICIAL_FOLDS[protocol["split_tag"]][1]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if any(not splits[part] for part in ("train", "val", "test")):
        raise ValueError("strict native experiments require train, validation and test cells")
    scaler = Scaler(metadata_profile=cfg.metadata_profile).fit(splits["train"], split="train")
    train_data = BMFPrefixDataset(splits["train"], scaler, 4, prefixes=args.train_prefixes)
    probe_data = BMFPrefixDataset(splits["train"], scaler, 4, prefixes="sweep")
    # One window per cell at its longest available prefix (capped at
    # --eval-prefix): folds whose raw cycles never reach 100 (e.g. CALB cond397
    # tops out at 99) would otherwise lose their evaluation entirely.
    eval_data = BMFPrefixDataset(splits[args.split], scaler, 4, prefixes="observed",
                                 observed_cap=args.eval_prefix if args.eval_prefix is not None else 100)
    predictor = NativePredictor(args.model, args.model_path, args.device, args.mode, prediction)
    if args.adapter:
        metadata, projection = predictor.load_adapter(args.adapter)
        if metadata["manifest_sha256"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest() or metadata["seed"] != args.seed:
            raise ValueError("adapter manifest or seed differs from the current strict experiment")
        static_report = metadata["static_conditioning"]
    else:
        projection = fit_static(splits["train"]) if predictor.kind not in ("sundial", "moment") else None
        static_report = probe_static(predictor, probe_data, projection)
        if not static_report["enabled"]:
            projection = None
    lora = metadata["lora"] if args.adapter else native.get("lora")
    losses = []
    if args.mode == "lora" and not args.adapter:
        if not isinstance(lora, dict) or set(lora) != {"rank", "alpha", "dropout"}:
            raise ValueError("native.lora must provide rank, alpha and dropout")
        predictor.add_lora(**lora)
        losses = train(predictor, train_data, projection, args.epochs, args.learning_rate, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"model": args.model, "model_id": predictor.model_id, "base_model": predictor.path,
                "revision": predictor.revision, "mode": args.mode,
                "feature_version": FEATURE_VERSION, "static_conditioning": static_report,
                "lora": lora if args.mode == "lora" else None, "seed": args.seed,
                "train_prefixes": (metadata["train_prefixes"] if args.adapter else args.train_prefixes)
                                  if args.mode == "lora" else None,
                "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "split_tag": protocol["split_tag"],
                "soh_only": predictor.kind in ("sundial", "moment"), "future_covariates": "masked",
                "past_covariates": [] if predictor.kind in ("sundial", "moment") else ["masked curve phase-bin means", "masked SOC phase-bin means", "CE", "EE"],
                "static_projection": "unlabeled training split PCA and sensitivity probe" if predictor.kind not in ("sundial", "moment") else "not used",
                "epochs": (metadata["epochs"] if args.adapter else args.epochs) if args.mode == "lora" else None,
                "learning_rate": (metadata["learning_rate"] if args.adapter else args.learning_rate)
                                 if args.mode == "lora" else None,
                "train_steps": metadata.get("train_steps") if args.adapter else len(losses),
                "last_train_loss": metadata.get("last_train_loss") if args.adapter else (losses[-1] if losses else None),
                "trained_forecast_head": predictor.kind == "moment" and args.mode == "lora",
                "prediction_parameters": prediction,
                "forecast_method": "native-forecast", "paper_protocol_verified": False,
                "source_protocol_explicitly_allowed": args.allow_source_protocol}
    if projection is not None:
        digest = hashlib.sha256()
        for key in ("mean", "components", "scales"):
            digest.update(projection[key].tobytes())
        metadata["static_projection_sha256"] = digest.hexdigest()
    if args.mode == "lora" and not args.adapter:
        predictor.save_adapter(args.output / "adapter", metadata, projection)
    # Reset after training so stochastic native generation starts at the same seed
    # in the zero-shot and adapted experiment.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    metrics = evaluate(predictor, eval_data, projection)
    result = {**metadata, "eval_split": args.split, "metrics": metrics}
    if args.eval_prefix is not None:
        result.update(eval_prefix=args.eval_prefix, eligible_cells=len(eval_data), total_cells=len(eval_data.records))
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
