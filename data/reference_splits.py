"""Read fixed split definitions from the pinned reference repositories.

This module only reads JSON/Python source data; it never imports or executes a
reference model.  The returned file lists are used to build extraction manifests.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path


def _entries(values, dataset):
    return [{"dataset": dataset, "cell_id": Path(value).stem, "source_file": value}
            for value in values]


def read_batterymformer_split(path: str | Path) -> dict:
    """Return the official train/val/test file list and split metadata."""
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    dataset = str(spec.get("dataset", "unknown"))
    result = {"protocol": "batterymformer", "split_protocol": spec.get("protocol"),
              "split_tag": path.stem, "split_seed": int(spec.get("seed", spec.get("val_seed", 0))),
              "train": _entries(spec.get("train", []), dataset),
              "val": _entries(spec.get("val", []), dataset),
              "test": _entries(spec.get("test", []), dataset)}
    identities = set()
    for split in ("train", "val", "test"):
        for row in result[split]:
            key = (row["dataset"], row["cell_id"])
            if key in identities:
                raise ValueError(f"BatteryMFormer split overlaps train/val/test: {key}")
            identities.add(key)
    return result


def _class_lists(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    env = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(t, ast.Name) for t in targets):
                continue
            def evaluate(value):
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return [value.value]
                if isinstance(value, (ast.List, ast.Tuple)):
                    result = []
                    for element in value.elts:
                        part = evaluate(element)
                        if part is None: return None
                        result.extend(part)
                    return result
                if isinstance(value, ast.Name):
                    return env.get(value.id)
                if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                    left, right = evaluate(value.left), evaluate(value.right)
                    return None if left is None or right is None else left + right
                return None
            values = evaluate(node.value)
            for target in targets:
                if isinstance(target, ast.Name) and values is not None:
                    env[target.id] = values
    return env


def read_batterylife_preset(path: str | Path, preset: str) -> dict:
    """Read one fixed BatteryLife script preset without importing its loader."""
    path = Path(path)
    lists = _class_lists(path)
    presets = {
        "cpmlp": ("CALB", "CALB", 2021),
        "cpmlp42": ("CALB_42", "CALB", 42),
        "cpmlp2024": ("CALB_2024", "CALB", 2024),
        "cptransformer": ("MIX_large", "MIX_large", 2024),
        "naion42": ("NAion_42", "NAion", 42),
        "naion2021": ("NAion_2021", "NAion", 2021),
        "naion2024": ("NAion_2024", "NAion", 2024),
        "znion42": ("ZN_42", "ZN-coin", 42),
        "znion2021": ("ZNcoin", "ZN-coin", 2021),
        "znion2024": ("ZN_2024", "ZN-coin", 2024),
    }
    try:
        prefix, dataset, seed = presets[preset.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown BatteryLife preset: {preset}") from exc
    names = {split: f"{prefix}_{split}_files" for split in ("train", "val", "test")}
    missing = [name for name in names.values() if name not in lists]
    if missing:
        raise ValueError(f"BatteryLife split recorder missing {missing}")
    result = {"protocol": "batterylife", "split_protocol": f"BatteryLife:{preset}",
              "split_tag": preset.lower(), "split_seed": seed,
              **{split: _entries(lists[name], dataset) for split, name in names.items()}}
    identities = set()
    for split in ("train", "val", "test"):
        for row in result[split]:
            key = (row["dataset"], row["cell_id"])
            if key in identities:
                raise ValueError(f"BatteryLife split overlaps train/val/test: {key}")
            identities.add(key)
    return result


def all_batterymformer_splits(root: str | Path) -> list[dict]:
    """Enumerate the 22 pinned BatteryMFormer fold definitions."""
    root = Path(root)
    files = sorted((root / "pure_ood").glob("*.json")) + sorted((root / "loao").glob("*.json"))
    result = [read_batterymformer_split(path) for path in files]
    if len(result) != 22:
        raise ValueError(f"expected 22 BatteryMFormer split files, found {len(result)}")
    return result
