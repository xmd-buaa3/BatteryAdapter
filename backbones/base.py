"""The common token contract; projections belong to the trainable feature bridge."""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import json
import os
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Tokens:
    tokens: torch.Tensor
    token_valid: torch.Tensor
    token_cycle: torch.Tensor
    token_variable: torch.Tensor


def patchify(x, mask, patch, minimum=0, side="left"):
    """Input [B,S,D] -> [B,D,P,K], preserving the original cycle coordinates."""
    b, s, d = x.shape
    length = max(s, minimum)
    length += (-length) % patch
    pad = (length - s, 0) if side == "left" else (0, length - s)
    values = F.pad(torch.where(mask, x, 0).transpose(1, 2), pad).unfold(-1, patch, patch)
    valid = F.pad(mask.transpose(1, 2), pad, value=False).unfold(-1, patch, patch)
    cycles = F.pad(torch.arange(1, s + 1, device=x.device), pad, value=0).unfold(-1, patch, patch).amax(-1)
    return values, valid, cycles


def pack(hidden, valid, cycles):
    b, d, p, width = hidden.shape
    if valid.shape != (b, d, p):
        raise ValueError("native token shape disagrees with patch metadata")
    if not torch.isfinite(hidden[valid]).all():
        raise FloatingPointError("nonfinite native history features")
    hidden = torch.where(valid[..., None], hidden, 0)
    cycle = cycles[None, None].expand(b, d, p)
    variable = torch.arange(d, device=hidden.device)[None, :, None].expand(b, d, p)
    return Tokens(hidden.reshape(b, d*p, width), valid.reshape(b, d*p),
                  cycle.reshape(b, d*p), variable.reshape(b, d*p))


class FrozenBackbone(nn.Module):
    def __init__(self, native, width, trainable=False):
        super().__init__()
        self.native, self.width = native, int(width)
        self.trainable = bool(trainable)
        self.native.requires_grad_(self.trainable).eval()

    def train(self, mode=True):
        super().train(mode)
        if not self.trainable:
            self.native.eval()
        return self

    def forward(self, x, mask):
        return self.encode(x, mask)


class TinyBackbone(FrozenBackbone):
    """Synthetic-test encoder with temporal mixing. Not a pretrained baseline."""
    def __init__(self, width=32, trainable=False):
        native = nn.Sequential(nn.Conv1d(1, width, 3, padding=1), nn.GELU(),
                               nn.Conv1d(width, width, 3, padding=1))
        super().__init__(native, width, trainable=trainable)

    def encode(self, x, mask):
        b, s, d = x.shape
        h = self.native(torch.where(mask, x, 0).transpose(1, 2).reshape(b*d, 1, s))
        return pack(h.transpose(1, 2).reshape(b, d, s, self.width), mask.transpose(1, 2),
                    torch.arange(1, s+1, device=x.device))


MODEL_IDS = {"moirai2": "Salesforce/moirai-2.0-R-small", "toto1": "Datadog/Toto-Open-Base-1.0",
             "moment": "AutonLab/MOMENT-1-large", "sundial": "thuml/sundial-base-128m",
             "chronos2": "amazon/chronos-2"}


def backbone_family(name):
    return name


def specification(name):
    root = Path(__file__).resolve().parents[1]
    entries = json.loads((root / "deployment/manifests/model_revisions.json").read_text())
    entry = next(e for e in entries if e["model_id"] == MODEL_IDS[name])
    requirements = root / "deployment/requirements" / f"{backbone_family(name)}.in"
    return {**entry, "requirements": requirements.read_text().strip()}


def local_model_override(model_id):
    """Offline deployment override: ``$BATTERYADAPTER_MODEL_ROOT/<repo-basename>``.

    The fleet keeps immutable weight directories under ``$MODEL_ROOT`` named by the
    repo basename (e.g. ``amazon/chronos-2`` -> ``chronos-2``); the deployment manifest
    pins their SHA-256, so loading a verified local directory is equivalent to loading
    the pinned revision.  ``revision`` is ignored for local directories by
    ``from_pretrained``.
    """
    root = os.environ.get("BATTERYADAPTER_MODEL_ROOT")
    if not root:
        return None
    candidate = Path(root) / model_id.split("/")[-1]
    return str(candidate) if candidate.is_dir() else None


def load_backbone(name, *, init="pretrained", trainable=False, frozen_scale=False):
    if name == "tiny":
        return TinyBackbone(trainable=trainable)
    if name not in MODEL_IDS:
        raise ValueError(f"unknown backbone {name}; choose {list(MODEL_IDS)}")
    spec = specification(name)
    cls = getattr(import_module(f"backbones.{backbone_family(name)}"), "Backbone")
    identifier = local_model_override(spec["model_id"]) or spec["model_id"]
    try:
        if init == "pretrained":
            result = cls.load(identifier, spec["revision"], trainable=trainable)
        elif init == "random":
            result = cls(cls.build_random(identifier), trainable=trainable)
        else:
            raise ValueError(f"unknown backbone init {init!r}")
    except ImportError as exc:
        raise RuntimeError(f"install deployment/requirements/{backbone_family(name)}.in in its own environment") from exc
    # frozen_scale is the timestral family's scaler-gradient policy; other
    # backbones ignore it.
    result.frozen_scale = bool(frozen_scale)
    result.spec = spec
    return result
