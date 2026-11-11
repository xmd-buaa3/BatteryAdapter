"""A: masked reconstruction, B: frozen input encoding, C: differentiable joint fitting."""
from dataclasses import fields, is_dataclass
import hashlib
import math
import random
import time
from numbers import Integral
import torch
from training.losses import losses


def map_tensors(tree, fn):
    if isinstance(tree, torch.Tensor):
        return fn(tree)
    if isinstance(tree, dict):
        return {k: map_tensors(v, fn) for k, v in tree.items()}
    if isinstance(tree, (tuple, list)):
        return type(tree)(map_tensors(v, fn) for v in tree)
    if is_dataclass(tree):
        return type(tree)(**{f.name: map_tensors(getattr(tree, f.name), fn) for f in fields(tree)})
    return tree


class NonfiniteGradientError(RuntimeError):
    """Backward produced inf/NaN gradients; raised before the optimizer step,
    so parameters still hold the last finite update.  The message keeps the
    original ``clip_grad_norm_`` text, then names the offending parameters."""


class Trainer:
    def __init__(self, model, device="cpu", *, fixed_encoder=False):
        self.model, self.device = model.to(device), torch.device(device)
        self.config = model.config
        self.random = random.Random(self.config.seed)
        self.cache, self.history = {}, []
        self.stage, self.optimizer = None, None
        self.fixed_encoder = fixed_encoder
        self.lr_scale = 1.  # shrinks only on in-process stage-C restarts

    def set_stage(self, stage):
        if stage not in "ABC" or len(stage) != 1:
            raise ValueError("stage must be A, B or C")
        self.stage = stage
        if not self.fixed_encoder:
            # Without a fixed encoder the stage transition retrains encoder-path
            # weights, so cached representations are stale.
            self.cache.clear()  # entry to B and C invalidates every previous representation
        self.model.requires_grad_(stage != "A")
        # The compressor learns only from the stage-A reconstruction objective
        # unless the "joint" ablation reopens it to task gradients in stage C.
        self.model.compressor.requires_grad_(
            stage == "A" or (stage == "C" and self.config.stage_c_compressor == "joint"))
        self.model.input_adapter.requires_grad_(stage != "A")
        if self.model.backbone is not None and not self.model.backbone.trainable:
            self.model.backbone.requires_grad_(False)
        if self.config.backbone_lora and self.model.backbone is not None:
            self.model.backbone.requires_grad_(False)
            if stage != "A":
                for name, parameter in self.model.backbone.named_parameters():
                    if "lora_" in name:
                        parameter.requires_grad_(True)
                if not any("lora_" in name and parameter.requires_grad
                           for name, parameter in self.model.backbone.named_parameters()):
                    raise RuntimeError("LoRA stage found no trainable adapter parameters")
        if self.fixed_encoder:
            if stage == "A":
                raise ValueError("head controls must use the shared, already trained compressor")
            for name in ("compressor", "input_adapter", "bridge"):
                getattr(self.model, name).requires_grad_(False)
        groups = []
        if self.model.backbone is not None and self.model.backbone.trainable:
            native_ids = {id(p) for p in self.model.backbone.native.parameters()}
            native = [p for p in self.model.parameters()
                      if p.requires_grad and id(p) in native_ids]
            rest = [p for p in self.model.parameters()
                    if p.requires_grad and id(p) not in native_ids]
            # backbone group keeps its own lr; the rest group falls back to the default
            if native:
                groups.append({"params": native, "lr": self.config.backbone_lr * self.lr_scale})
            if rest:
                groups.append({"params": rest})
        else:
            rest = [p for p in self.model.parameters() if p.requires_grad]
            if rest:
                groups.append({"params": rest})
        if not groups:
            raise ValueError("set_stage left no trainable parameters")
        self.optimizer = torch.optim.AdamW(groups, lr=self.config.learning_rate * self.lr_scale,
                                           weight_decay=self.config.weight_decay)

    def _masked_inputs(self, inputs, level):
        x = {**inputs}
        p = self.config.field_dropout
        if level >= 2 and "metadata_mask" in x:
            x["metadata_mask"] = {k: v & (torch.rand((v.shape[0],) + (1,)*(v.ndim-1), device=v.device) >= p)
                                  for k, v in x["metadata_mask"].items()}
        if level == 3:
            for key in ("soc_mask", "soh_mask", "cycle_features_mask"):
                if key in x:
                    mask = x[key]
                    x[key] = mask & (torch.rand((mask.shape[0],) + (1,)*(mask.ndim-1), device=mask.device) >= p)
        return x

    def _base(self, x, level):
        if not self.config.cache_stage_b or not (self.stage == "B" or self.fixed_encoder):
            return self.model.encode_base(x, level)
        # With cache_stage_b, stage B and any fixed-encoder stage freeze the
        # encoder path (the guard below rejects a trainable input adapter), so
        # the representation of a given sample is computed once and reused
        # across epochs and stages.
        if any(parameter.requires_grad for parameter in self.model.input_adapter.parameters()):
            raise RuntimeError("cache_stage_b cannot cache a trainable early input adapter")
        digest = hashlib.sha256()
        digest.update(str(level).encode())
        def update(value, name):
            if isinstance(value, dict):
                for child in sorted(value):
                    update(value[child], f"{name}.{child}")
            elif isinstance(value, torch.Tensor):
                value = value.detach().cpu().contiguous()
                digest.update(str((name, value.shape, value.dtype)).encode())
                digest.update(value.numpy().tobytes())
        update(x, "inputs")
        key = digest.hexdigest()
        if key not in self.cache:
            self.cache[key] = map_tensors(self.model.encode_base(x, level), lambda t: t.detach().cpu())
        return map_tensors(self.cache[key], lambda t: t.to(self.device))

    def step(self, batch):
        if self.stage is None:
            raise RuntimeError("select a training stage first")
        batch = map_tensors(batch, lambda t: t.to(self.device))
        self.model.train()
        if not any(parameter.requires_grad for parameter in self.model.compressor.parameters()):
            self.model.compressor.eval()
        if self.fixed_encoder:
            for name in ("compressor", "input_adapter", "bridge"):
                getattr(self.model, name).eval()
            if self.model.backbone is not None and not self.model.backbone.trainable:
                self.model.backbone.eval()  # frozen native: deterministic encodings
        self.optimizer.zero_grad(set_to_none=True)
        if self.stage == "A":
            x = self.model.validate(batch["inputs"], self.config.levels[0])
            mask = x["curve_mask"]
            hidden = (torch.rand_like(x["curves"]) < .25) & mask
            observed = mask & ~hidden
            compressed = self.model.compressor(x["curves"], observed)[0]
            reconstructed = self.model.compressor.decode(compressed)
            error = (reconstructed - x["curves"]).square()
            loss = torch.where(hidden, error, 0).sum() / hidden.sum().clamp_min(1)
        else:
            level = self.random.choice(self.config.levels)
            x = self.model.validate(self._masked_inputs(batch["inputs"], level), level)
            query = self.model.query_grid(self.device)
            expected = query[None].expand_as(batch["targets"]["query_cycles"])
            if not torch.equal(expected, batch["targets"]["query_cycles"]):
                raise ValueError("targets must share the configured query grid; no EOL-derived horizon")
            prediction = self.model.predict_encoded(x, level, query, self._base(x, level))
            loss = losses(prediction, batch["targets"], self.config)["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        named = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        try:
            # error_if_nonfinite raises before any gradient is rescaled, so the
            # scan below still sees the raw backward values.
            torch.nn.utils.clip_grad_norm_([p for _, p in named], 5., error_if_nonfinite=True)
        except RuntimeError as error:
            bad = [name for name, p in named if p.grad is not None and not torch.isfinite(p.grad).all()]
            raise NonfiniteGradientError(
                f"{error} [stage {self.stage}: {len(bad)} parameter(s) with non-finite gradients; "
                f"first: {bad[0] if bad else 'unknown'}]") from error
        self.optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        values = []
        for batch in loader:
            batch = map_tensors(batch, lambda t: t.to(self.device))
            expected = self.model.query_grid(self.device)[None].expand_as(batch["targets"]["query_cycles"])
            if not torch.equal(expected, batch["targets"]["query_cycles"]):
                raise ValueError("validation labels must use the configured query grid")
            per_level = []
            for level in self.config.levels:
                x = self.model.validate(batch["inputs"], level)
                query = self.model.query_grid(self.device)
                expected = query[None].expand_as(batch["targets"]["query_cycles"])
                if not torch.equal(expected, batch["targets"]["query_cycles"]):
                    raise ValueError("validation labels must use the configured query grid")
                if self.config.cache_stage_b and self.fixed_encoder:
                    prediction = self.model.predict_encoded(x, level, query, self._base(x, level))
                else:
                    prediction = self.model(batch["inputs"], level)
                per_level.append(float(losses(prediction, batch["targets"], self.config)["total"]))
            values.append((sum(per_level) / len(per_level), len(batch["cell_id"])))
        return sum(v*n for v, n in values)/sum(n for _, n in values) if values else None

    def fit(self, train_loader, val_loader=None, *, checkpoint_policy="last", val_scorer=None,
            patience=30, progress=None, val_every=1, epoch_observer=None):
        if checkpoint_policy not in ("last", "best-c-val-mae", "best-c-bmf-mape-v1"):
            raise ValueError("unknown checkpoint policy")
        if isinstance(patience, bool) or not isinstance(patience, Integral) or patience < 1:
            raise ValueError("patience must be a positive integer")
        if isinstance(val_every, bool) or not isinstance(val_every, Integral) or val_every < 1:
            raise ValueError("val_every must be a positive integer")
        select_best = checkpoint_policy != "last"
        strict_mape = checkpoint_policy == "best-c-bmf-mape-v1"
        if select_best and (self.config.levels != (4,) or self.fixed_encoder
                            or self.config.backbone_train != "frozen"
                            or self.config.backbone_init != "pretrained"
                            or self.config.backbone_lora or self.model.backbone.trainable):
            raise ValueError("best-C selection requires frozen pretrained Level-4 adapter training")
        if select_best and (not self.config.epochs[2] or val_loader is None or val_scorer is None):
            raise ValueError("best-C selection requires stage C, validation data and a scorer")
        self.checkpoint_selection = dict(checkpoint_policy=checkpoint_policy)
        best_state = None
        stale = 0
        self.lr_scale = 1.
        restarts = []
        for stage, count in zip("ABC", self.config.epochs):
            if count == 0:
                continue
            if stage == "C":
                # End-of-B restore point: a stage-C divergence restarts C from
                # here instead of discarding the whole A/B/C run.
                stage_start = self._snapshot()
                selection_start = dict(self.checkpoint_selection)
                history_start = len(self.history)
            while True:
                self.set_stage(stage)
                try:
                    best_state, stale = self._run_stage_epochs(
                        stage, count, train_loader, val_loader, progress, val_every, epoch_observer,
                        select_best, strict_mape, val_scorer, patience, best_state, stale)
                except (NonfiniteGradientError, FloatingPointError) as error:
                    if stage != "C" or len(restarts) >= self.config.stage_c_max_restarts:
                        raise
                    lr = self.config.learning_rate * self.lr_scale
                    self.lr_scale *= self.config.stage_c_restart_lr_factor
                    restarts.append(dict(attempt=len(restarts), lr=lr,
                                         failed_epoch=len(self.history) - history_start + 1,
                                         next_lr=self.config.learning_rate * self.lr_scale,
                                         error=f"{type(error).__name__}: {error}"))
                    if progress is not None:
                        progress(dict(event="stage_c_restart", **restarts[-1]))
                    # The failed attempt's epochs, best snapshot and patience
                    # counter belong to a discarded trajectory.
                    del self.history[history_start:]
                    self._restore(stage_start)
                    self.checkpoint_selection = {**selection_start, "stage_c_restarts": list(restarts)}
                    best_state, stale = None, 0
                    continue
                break
        if select_best:
            restored = self.model.load_state_dict(best_state, strict=False)
            if restored.unexpected_keys or any(not key.startswith("backbone.native.")
                                               for key in restored.missing_keys):
                raise RuntimeError("best-C snapshot is missing adapter state or has unexpected keys")
            if not strict_mape:
                self.checkpoint_selection.update(
                    best_val_mae_x1e2=self.checkpoint_selection["best_val_mae"] * 100,
                    final_epoch_val_x1e2=self.checkpoint_selection["final_epoch_val"] * 100)
        return self.history

    def _snapshot(self):
        """Weights, buffers and RNG streams needed to replay a stage from here.

        Frozen native weights never change and are skipped; a trainable
        backbone (full / LoRA) is captured with everything else.
        """
        skip_native = self.model.backbone is not None and not self.model.backbone.trainable
        state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()
                 if not (skip_native and key.startswith("backbone.native."))}
        rng = dict(torch=torch.get_rng_state(), python=self.random.getstate(),
                   cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        return state, rng

    def _restore(self, snapshot):
        state, rng = snapshot
        restored = self.model.load_state_dict(state, strict=False)
        if restored.unexpected_keys or any(not key.startswith("backbone.native.")
                                           for key in restored.missing_keys):
            raise RuntimeError("stage snapshot is missing adapter state or has unexpected keys")
        torch.set_rng_state(rng["torch"])
        self.random.setstate(rng["python"])
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

    def _run_stage_epochs(self, stage, count, train_loader, val_loader, progress, val_every,
                          epoch_observer, select_best, strict_mape, val_scorer, patience,
                          best_state, stale):
        """Train one stage; returns the updated (best_state, stale) pair."""
        for epoch in range(count):
            total, n = 0., 0
            for batch in train_loader:
                size = len(batch["cell_id"])
                total += self.step(batch)*size; n += size
                if progress is not None and n == size:
                    progress(dict(event="first_train_step", stage=stage, epoch=epoch+1))
            if not n:
                raise ValueError("empty training split")
            row = dict(stage=stage, epoch=epoch+1, train_loss=total/n)
            due = (val_loader is not None and stage != "A"
                   and ((epoch + 1) % val_every == 0 or epoch + 1 == count))
            if due:
                started = time.perf_counter()
                row["val_loss"] = self.validate(val_loader)
                row["val_loss_seconds"] = time.perf_counter() - started
            if select_best and stage == "C" and due:
                started = time.perf_counter()
                with torch.no_grad():
                    score = float(val_scorer(self.model, val_loader))
                metric = "val_mape" if strict_mape else "val_mae"
                row[metric + "_seconds"] = time.perf_counter() - started
                if not math.isfinite(score):
                    raise ValueError("nonfinite validation curve metric")
                row[metric] = score
                self.checkpoint_selection["final_epoch_val"] = score
                best_key = "best_val_mape" if strict_mape else "best_val_mae"
                improved = best_state is None or (score <= self.checkpoint_selection[best_key] if strict_mape
                                                   else score < self.checkpoint_selection[best_key])
                if improved:
                    # Frozen native weights stay in place; clone all adapter weights and buffers.
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in self.model.state_dict().items()
                                  if not key.startswith("backbone.native.")}
                    self.checkpoint_selection.update(best_stage="C", best_epoch=epoch+1,
                                                     **{best_key: score})
                    if strict_mape:
                        self.checkpoint_selection["val_metric_unit"] = "percent"
                    else:
                        self.checkpoint_selection["val_mae_unit"] = "physical_soh"
                    stale = 0
                elif strict_mape and epoch + 1 > 1:
                    stale += 1
            self.history.append(row)
            if progress is not None:
                progress(dict(event="epoch_completed", **row))
            natural_stop = strict_mape and stage == "C" and due and stale >= patience
            if epoch_observer is not None and strict_mape and stage == "C" and due:
                epoch_observer(epoch + 1, self.checkpoint_selection["best_val_mape"],
                               natural_stop or epoch + 1 == count)
            if natural_stop:
                self.checkpoint_selection["early_stopped"] = True
                break
        return best_state, stale

    def save(self, path, *, scaler, protocol, run_metadata=None, include_native=True):
        from checkpoint import save
        save(path, self.model, scaler=scaler, protocol=protocol, history=self.history,
             stage=self.stage, run_metadata=run_metadata, include_native=include_native)
