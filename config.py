from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
import os

from data.feature_provenance import METADATA_PROFILES

# A/B/C epochs pinned by the strict BatteryMFormer family.  Stage C runs the
# user's cost-reduced budget -- up to 100 epochs with patience-10 early
# stopping on validation MAPE -- deliberately not the reference's official
# 300/30; the selection metric and protocol are unchanged, but the budget cut
# materially affects each fold's champion and the final numbers (recorded in
# the strict registration limitations).  The trainer enforces the patience,
# which adapts the actual cost per candidate.  Both the strict runner and the
# training entry point read these constants so they cannot drift.
# The default is the cost-reduced 40-epoch budget; a single refit campaign can
# opt into the historical 100-epoch budget with
# BATTERYADAPTER_STRICT_EPOCHS=10,20,100 without touching the default.
STRICT_EPOCHS = tuple(int(x) for x in os.environ.get(
    "BATTERYADAPTER_STRICT_EPOCHS", "10,20,40").split(","))
STRICT_PATIENCE = 10


@dataclass(frozen=True)
class Config:
    observed_cycles: int = 100
    points: int = 128
    current_threshold: float = 1e-5  # amperes; independent of rated capacity
    curve_channels: int = 3  # V, I, and a genuinely recorded within-cycle Q curve
    reference_embedding_dim: int = 1024
    input_protocol: str = "auto"
    task_head: str = "auto"
    life_loss: str = "scaled_huber"
    curve_loss: str = "huber"  # huber | mse (simplest-loss baseline)
    # Joint-task guard: when set, the curve loss may not backprop into the
    # lifetime/start heads (their values enter the decoder detached), so the
    # life prediction answers only to the life label.
    protect_life_head: bool = False
    # Attribution controls (batch-1 plan): each ablation flips exactly one of
    # these switches; the rest of the configuration stays the frozen champion.
    backbone_init: str = "pretrained"  # pretrained | random (same architecture, fresh weights)
    backbone_train: str = "frozen"  # frozen | full (unfrozen backbone, separate lr group)
    backbone_lr: float = 1e-4  # lr for the backbone param group when backbone_train="full"
    bridge_read: str = "attention"  # attention | pool (pool keeps the local query path)
    compressor: str = "phase_cnn"  # phase_cnn | cp_flatten (whole-cycle MLP, CP-style)
    tiny_trainable: bool = False  # tiny encoder trains: information-matched no-TSF control
    backbone_lora: bool = False  # PEFT LoRA on attention projections (batch-2 cell)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    compressed_dim: int = 16
    hidden_dim: int = 256
    local_dim: int = 32
    heads: int = 8
    expert_dim: int = 64
    life_experts: int = 4
    shape_experts: int = 4
    life_queries: int = 8
    shape_queries: int = 32
    shape_bins: int = 32
    eta: float = 0.8
    shape_floor: float = 1e-3
    start_margin: float = 1e-3
    remaining_min: float = 1.0
    remaining_scale: float = 900.0
    horizon: int = 1000
    backbone: str = "tiny"  # explicit synthetic-test model, never a TSFM result
    use_tsfm: bool = True
    use_moe: bool = True
    use_interaction: bool = True
    use_soc: bool = True
    use_soh: bool = True
    use_cycle_features: bool = True  # Level-3 CE/EE state channels (BatteryMFormer cycle-level features)
    observed_start: bool = True
    scheme_b: bool = True
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    sigma_soh: float = 0.02
    loss_weights: tuple = (1.0, 1.0, 0.1)
    epochs: tuple = (5, 10, 20)
    # Stage-C compressor policy.  "joint" (default) keeps the historical
    # B-frozen / C-trainable schedule, matching every completed Na-ion, CALB,
    # Li-ion and Zn result.  "frozen" keeps the stage-A compressor fixed
    # through B and C: its 16 channels enter every native TSFM normalizer
    # (toto2/chronos2/moirai2/timestral) whose detached loc/scale turn the
    # backward pass into a 1/scale amplifier for near-constant windows, which
    # is where every stage-C divergence originated.  Enable it only for the
    # frozen ablation line, together with timestral_frozen_scale.
    stage_c_compressor: str = "joint"
    # In-process stage-C recovery: on a nonfinite loss or gradient, restore the
    # end-of-B weights and restart C with lr multiplied by the factor, at most
    # ``stage_c_max_restarts`` times before the error propagates.
    stage_c_max_restarts: int = 2
    stage_c_restart_lr_factor: float = 0.25
    # Timestral scaler gradient policy.  False (default) passes values through
    # the native scale() unchanged, matching every completed timestral result.
    # True applies frozen_statistics_scale: statistics computed without grad
    # (the native transform is verified on every call), removing the sqrt
    # backward on zero-variance rows.  Enable it only with the frozen line.
    timestral_frozen_scale: bool = False
    levels: tuple = (1, 2, 3)
    information_ablation: int | None = None  # strict Level-4 input ablation: 1 or 2
    input_drop: tuple[str, ...] = ()
    field_dropout: float = 0.1
    cache_stage_b: bool = False
    # The primary experiment injects every permitted information tier before the
    # TSFM.  The historical TierFusion implementation is intentionally not
    # constructed by BatteryAdapter anymore.
    fusion_mode: str = "early"
    state_bins: int = 4
    static_basis_dim: int = 4
    input_layout_version: int = 1
    # legacy keeps the original six-field metadata whitelist; "bmf" extends Level-2
    # registration to the full BatteryMFormer aging-condition tuple (see
    # data/feature_provenance.py) and pins input_layout_version=2.
    metadata_profile: str = "legacy"
    soh_scale: str = "source_eta"
    # BatteryLife fits these values on the training split.  Legacy and
    # BatteryMFormer runs leave the identity transform in place.
    eol_mean: float = 0.0
    eol_std: float = 1.0
    seed: int = 17

    def __post_init__(self):
        for name in ("observed_cycles", "points", "hidden_dim", "local_dim", "heads",
                     "expert_dim", "life_experts", "shape_experts", "life_queries", "shape_queries", "shape_bins", "horizon", "compressed_dim", "curve_channels",
                     "state_bins", "static_basis_dim", "input_layout_version", "reference_embedding_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("eta", "shape_floor", "start_margin", "remaining_min", "remaining_scale",
                     "sigma_soh", "current_threshold", "dropout", "field_dropout", "learning_rate", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("eol_mean", "eol_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.eol_std <= 0:
            raise ValueError("eol_std must be positive")
        if not isinstance(self.backbone, str) or not self.backbone.strip():
            raise ValueError("backbone must be a nonempty name")
        if self.backbone_init not in ("pretrained", "random"):
            raise ValueError("backbone_init must be pretrained or random")
        if self.backbone_train not in ("frozen", "full"):
            raise ValueError("backbone_train must be frozen or full")
        if self.bridge_read not in ("attention", "pool"):
            raise ValueError("bridge_read must be attention or pool")
        if self.compressor not in ("phase_cnn", "cp_flatten", "capacity_only"):
            raise ValueError("compressor must be phase_cnn, cp_flatten or capacity_only")
        for name in ("use_tsfm", "use_moe", "use_interaction", "use_soc", "use_soh", "use_cycle_features", "observed_start", "scheme_b", "cache_stage_b", "protect_life_head", "tiny_trainable", "backbone_lora"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.backbone_lora:
            if not self.use_tsfm:
                raise ValueError("backbone_lora requires use_tsfm")
            # the config's backbone field is "tiny" in protocol runs (run.py swaps in
            # the real backbone later); target selection is enforced in model.py
            if self.backbone not in ("tiny", "chronos2", "toto1"):
                raise ValueError("backbone_lora targets are only defined for chronos2 and toto1")
        for name in ("lora_r", "lora_alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must lie in [0, 1)")
        if self.points < 4 or self.hidden_dim % self.heads:
            raise ValueError("points >= 4 and hidden_dim divisible by heads required")
        if self.compressed_dim not in (8, 16, 32) or self.curve_channels not in (2, 3):
            raise ValueError("supported curve dimensions: 8/16/32; channels: 2/3")
        if self.fusion_mode != "early":
            raise ValueError("the active BatteryAdapter path only supports fusion_mode='early'")
        if self.state_bins < 2 or self.static_basis_dim < 2 or self.input_layout_version < 1:
            raise ValueError("state_bins/static_basis_dim/input_layout_version are invalid")
        if self.metadata_profile not in METADATA_PROFILES:
            raise ValueError("metadata_profile must be legacy or bmf")
        if (self.metadata_profile == "bmf") != (self.input_layout_version == 2):
            raise ValueError("metadata_profile='bmf' requires input_layout_version=2; legacy requires 1")
        if self.observed_cycles <= self.static_basis_dim:
            raise ValueError("observed_cycles must exceed static_basis_dim for non-degenerate carriers")
        if self.soh_scale not in ("source_eta", "raw"):
            raise ValueError("soh_scale must be source_eta or raw")
        if not 0 < self.eta < 1 or not 0 < self.shape_floor < 1:
            raise ValueError("eta and shape_floor must lie in (0, 1)")
        if min(self.start_margin, self.remaining_min, self.remaining_scale, self.sigma_soh) <= 0:
            raise ValueError("numerical scales must be positive")
        if self.current_threshold < 0 or self.weight_decay < 0 or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive; threshold and weight_decay nonnegative")
        if self.backbone_lr <= 0:
            raise ValueError("backbone_lr must be positive")
        if self.stage_c_compressor not in ("frozen", "joint"):
            raise ValueError("stage_c_compressor must be frozen or joint")
        if not isinstance(self.timestral_frozen_scale, bool):
            raise ValueError("timestral_frozen_scale must be a boolean")
        if self.information_ablation is not None and (
            isinstance(self.information_ablation, bool) or not isinstance(self.information_ablation, Integral)
            or self.information_ablation not in (1, 2)
            or tuple(self.levels) != (4,)
        ):
            raise ValueError("information_ablation requires Level 4 and value 1 or 2")
        allowed_drop = ("curve", "cyc", "aging")
        if (not isinstance(self.input_drop, (tuple, list))
                or any(value not in allowed_drop for value in self.input_drop)
                or len(set(self.input_drop)) != len(self.input_drop)):
            raise ValueError("input_drop must be a unique subset of curve/cyc/aging")
        if self.input_drop and (tuple(self.levels) != (4,) or self.information_ablation is not None):
            raise ValueError("input_drop requires Level 4 without information_ablation")
        object.__setattr__(self, "input_drop", tuple(value for value in allowed_drop if value in self.input_drop))
        if (isinstance(self.stage_c_max_restarts, bool) or not isinstance(self.stage_c_max_restarts, Integral)
                or self.stage_c_max_restarts < 0):
            raise ValueError("stage_c_max_restarts must be a nonnegative integer")
        factor = self.stage_c_restart_lr_factor
        if isinstance(factor, bool) or not isinstance(factor, Real) or not 0 < factor < 1:
            raise ValueError("stage_c_restart_lr_factor must lie in (0, 1)")
        if not isinstance(self.epochs, (tuple, list)) or len(self.epochs) != 3 or any(isinstance(e, bool) or not isinstance(e, Integral) or e < 0 for e in self.epochs):
            raise ValueError("epochs must contain three nonnegative counts")
        if not isinstance(self.levels, (tuple, list)) or not self.levels or any(isinstance(l, bool) or not isinstance(l, Integral) or l not in (1, 2, 3, 4, 5) for l in self.levels) or len(set(self.levels)) != len(self.levels):
            raise ValueError("levels must be drawn from 1, 2, 3, 4 or 5")
        if not isinstance(self.loss_weights, (tuple, list)) or len(self.loss_weights) != 3 or any(isinstance(w, bool) or not isinstance(w, Real) or not math.isfinite(w) or w < 0 for w in self.loss_weights) or not any(self.loss_weights):
            raise ValueError("loss_weights must contain three finite nonnegative weights, at least one positive")
        if any(l >= 4 for l in self.levels) and len(self.levels) != 1:
            raise ValueError("Level 4/5 are separate input protocols; select one per run")
        expected_protocol = {4: "batterymformer", 5: "batterylife"}.get(max(self.levels), "legacy")
        if self.levels == (5,) or self.levels == [5]:
            # "curve" on the BatteryLife protocol trains the trajectory head on
            # the auxiliary SOH labels (the curve-first experiment); its EOL is
            # the decode crossing by construction.
            expected_heads = ("scalar", "joint", "curve")
        else:
            expected_heads = ("curve",)
        if self.input_protocol not in ("auto", expected_protocol):
            raise ValueError(f"selected levels require input_protocol={expected_protocol}")
        if self.task_head != "auto" and self.task_head not in expected_heads:
            raise ValueError(f"selected levels require task_head in {expected_heads}")
        if self.life_loss not in ("scaled_huber", "log_huber"):
            raise ValueError("life_loss must be scaled_huber or log_huber")
        if self.curve_loss not in ("huber", "mse"):
            raise ValueError("curve_loss must be huber or mse")
        if expected_protocol != "legacy" and (not self.use_tsfm or self.curve_channels != 3):
            raise ValueError("Level 4/5 require TSFM + BatteryAdapter and three physical channels")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("epochs", "levels", "loss_weights"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not 0 <= self.field_dropout < 1 or not 0 <= self.dropout < 1:
            raise ValueError("dropout probabilities must lie in [0, 1)")

    def to_dict(self):
        return asdict(self)

    @property
    def protocol(self):
        return {4: "batterymformer", 5: "batterylife"}.get(max(self.levels), "legacy") \
            if self.input_protocol == "auto" else self.input_protocol

    @property
    def prediction_task(self):
        return ("scalar" if tuple(self.levels) == (5,) else "curve") \
            if self.task_head == "auto" else self.task_head
