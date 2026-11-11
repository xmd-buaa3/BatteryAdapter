"""Versioned inference checkpoints. Training optimizer/RNG resumption is not implied."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import torch
from config import Config
from data.dataset import Scaler

SCHEMA_VERSION = 4


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(str((key, tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def runtime_record():
    root = Path(__file__).resolve().parent
    files = {p.name if p.parent == root else p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*")) if p.suffix in (".py", ".json")}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                          if d.metadata["Name"] and "_vendor" not in Path(d.locate_file("")).parts},
                git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
                source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())


def save(path, model, *, scaler, protocol, history, stage=None, run_metadata=None, include_native=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    baseline = False
    spec = getattr(model.backbone, "spec", {"name": "tiny" if model.backbone else "none"})
    # Match the JSON report schema and keep metadata safe for weights_only loading.
    def json_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"checkpoint metadata is not JSON serializable: {type(value).__name__}")

    state = model.state_dict()
    native_omitted = False
    if not include_native:
        if (model.config.backbone_train == "full" or model.config.backbone_init == "random"):
            raise ValueError("native backbone weights are trained or random-initialized; "
                             "they must be stored in the checkpoint")
        state = {key: value for key, value in state.items()
                 if not key.startswith("backbone.native.")
                 or (model.config.backbone_lora and ".lora_" in key)}
        native_omitted = any(key.startswith("backbone.native.") for key in model.state_dict())
    metadata = dict(schema_version=SCHEMA_VERSION, kind="adapter",
                    config=model.config.to_dict(), scaler=scaler.state_dict(),
                    protocol=protocol, history=history, stage=stage, backbone_spec=spec,
                    fusion_mode=getattr(model.config, "fusion_mode", None),
                    moe_expert_counts=dict(lifetime=getattr(model.config, "life_experts", None),
                                           shape=getattr(model.config, "shape_experts", None)),
                    input_layout=(getattr(getattr(model, "input_adapter", None), "layout", None)),
                    native_omitted=native_omitted,
                    run=run_metadata or {}, runtime=runtime_record())
    metadata = json.loads(json.dumps(metadata, allow_nan=False, default=json_scalar))
    torch.save({**metadata, "model": state}, temporary)
    temporary.replace(path)


def validate_resident_backbone(config, backbone, *, device=None):
    """Only a pinned frozen backbone may be shared between independent jobs."""
    from backbones.base import specification
    expected = {"name": "tiny"} if config.backbone == "tiny" else specification(config.backbone)
    if (not config.use_tsfm or config.backbone_train != "frozen"
            or config.backbone_init != "pretrained" or config.backbone_lora
            or config.tiny_trainable or getattr(backbone, "trainable", True)
            or any(p.requires_grad for p in backbone.parameters())
            or getattr(backbone, "spec", {"name": "tiny"}) != expected):
        raise ValueError("resident backbone must match the frozen pretrained configuration")
    if device is not None:
        requested = torch.device(device)
        if requested.type == "cuda" and requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
        if any(value.device != requested for value in (*backbone.parameters(), *backbone.buffers())):
            raise ValueError("resident backbone must already be on the requested device")


def load(path, device="cpu", *, reference_root=None, expected_protocol=None, backbone=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        if schema_version == 3:
            raise ValueError("unsupported checkpoint schema 3 after MoE parameterization; rebuild with schema 4")
        raise ValueError("unsupported checkpoint schema; rebuild legacy checkpoints explicitly")
    config = Config(**payload["config"])
    if backbone is not None:
        validate_resident_backbone(config, backbone, device=device)
        if (payload.get("native_omitted") is not True
                or any(key.startswith("backbone.native.") for key in payload["model"])):
            raise ValueError("resident loading requires native-omitted weights without a native payload")
    protocol = payload["protocol"]
    for key in ("eta", "horizon", "observed_cycles", "soh_scale", "input_layout_version", "metadata_profile"):
        if key in protocol and protocol[key] != getattr(config, key):
            raise ValueError(f"checkpoint protocol/config mismatch: {key}")
    if expected_protocol is not None:
        for key, value in expected_protocol.items():
            if protocol.get(key) != value:
                raise ValueError(f"checkpoint data protocol mismatch: {key}")
    scaler = Scaler.from_state_dict(payload["scaler"])
    sizes = {k: len(v)+1 for k, v in scaler.vocab.items()}
    if payload["kind"] == "adapter":
        from model import BatteryAdapter
        model = BatteryAdapter(config, vocab_sizes=sizes, **({"backbone": backbone} if backbone is not None else {}))
        spec = getattr(model.backbone, "spec", {"name": "tiny" if model.backbone else "none"})
        if payload.get("fusion_mode") != config.fusion_mode:
            raise ValueError("checkpoint fusion mode differs from the active early-fusion implementation")
        expected_counts = {"lifetime": config.life_experts, "shape": config.shape_experts}
        if payload.get("moe_expert_counts") != expected_counts:
            raise ValueError("checkpoint MoE expert counts differ from the active configuration")
        if int(config.input_layout_version) != int(payload["config"].get("input_layout_version", -1)):
            raise ValueError("checkpoint input layout version differs from the active implementation")
        expected_layout = {key: list(value) for key, value in model.input_adapter.layout.items()}
        if payload.get("input_layout") != expected_layout:
            raise ValueError("checkpoint input layout differs from the active implementation")
    elif payload["kind"] == "baseline":
        raise ValueError("native reference checkpoints are unsupported; use external result files only")
    else:
        raise ValueError("unknown checkpoint model kind")
    if spec != payload["backbone_spec"]:
        raise ValueError("checkpoint model/dependency/source specification differs from the installed implementation")
    if payload.get("native_omitted"):
        if (config.backbone_train == "full" or config.backbone_init == "random"):
            # Older saves dropped the native subtree unconditionally; for a
            # trained or random-initialized backbone the deployment weights the
            # constructor just built are NOT the weights that were trained, so
            # restoring this file would silently replace the fine-tuned model.
            raise ValueError("native-omitted checkpoint carries trained or random-init "
                             "native backbone weights and cannot be restored")
        # Frozen native weights were omitted at save time; the constructor above rebuilt
        # them from the pinned deployment weights, so every missing key must be native.
        result = model.load_state_dict(payload["model"], strict=False)
        if result.unexpected_keys or not all(key.startswith("backbone.native.") for key in result.missing_keys):
            raise ValueError("native-omitted checkpoint is missing trainable weights or carries unexpected keys")
        if config.backbone_lora and any(".lora_" in key for key in result.missing_keys):
            raise ValueError("native-omitted checkpoint is missing LoRA adapter weights")
    else:
        model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), scaler, payload
