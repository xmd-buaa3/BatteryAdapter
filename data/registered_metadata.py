"""BatteryMFormer registered aging-condition metadata, parsed from the official prompts.

The 2026-10-02 information-registration completion aligns the Level-2 whitelist with
what BatteryMFormer actually registers per cell: the aging-condition tuple
(positive/negative electrode, electrolyte, package structure, nominal capacity,
manufacturer, formation protocol, charge protocol, discharge protocol, operating
temperature, SOC window).  The authoritative per-cell source in this repository is the
official prompt table ``reference/BatteryMFormer/Prompts/*_protocol_prompt.py`` with the
cell→prompt mapping in ``Prompts/Mapping_helper.py`` — the same mapping BatteryMFormer's
own feature pipeline uses to build the aging-condition embeddings.

This module turns each cell's registered prompt text into the structured whitelist
fields declared in :mod:`batteryadapter.data.feature_provenance` (``BMF_NUMERIC`` /
``BMF_CATEGORICAL``).  It never reads raw pickles and never invents values: a factor the
prompt does not state (or states as "unknown") stays absent, so the downstream metadata
mask records genuine registered absence.  The multi-stage charge/discharge protocol
sentences are registered twice over — as numeric summaries (rates, cutoff voltages, CV
flag, stage count) and as exact-protocol categorical IDs (normalized-sentence hash, so
cells whose registered protocol text agrees share one category).
"""
from __future__ import annotations

import hashlib
import re
import sys
import warnings
from pathlib import Path

PROMPT_REFERENCE = "reference/BatteryMFormer/Prompts (BatteryMFormer official per-cell aging-condition registration)"

_NUM = r"(-?\d+(?:\.\d+)?)"
_SPEC_PATTERNS = {
    "battery_format": re.compile(r"in a format of (.+?) battery\."),
    "battery_chemistry": re.compile(r"Its positive electrode is (.+?)\. "),
    "battery_anode": re.compile(r"Its negative electrode is (.+?)\. "),
    "battery_electrolyte": re.compile(r"The electrolyte formula (?:is|consists of) (.+?)\. "),
    "battery_manufacturer": re.compile(r"The battery manufacturer is (.+?)\. "),
    "battery_formation": re.compile(r"The working history of this battery is (.+?)\. "),
    "ambient_temperature": re.compile(r"ambient temperature of this battery is " + _NUM + r" degrees Celsius"),
    "soc_range": re.compile(r"state-of-charge of this battery ranges from " + _NUM + r"% to " + _NUM + r"%"),
    "nominal_capacity": re.compile(r"The nominal capacity is " + _NUM + r" Ah"),
}
_CHARGE_RATE = re.compile(r"charged at a constant current of " + _NUM + r" C")
_CHARGE_VOLTAGE = re.compile(r"charged at a constant current of " + _NUM + r" C[^.]*?" + _NUM + r"\s*V")
_CHARGE_CV_VOLTAGE = re.compile(r"constant[- ]voltage(?: charged)? (?:of|to) " + _NUM + r"\s*V")
_DISCHARGE_RATE = re.compile(r"discharged at a constant current of " + _NUM + r" C")
_DISCHARGE_VOLTAGE = re.compile(r"discharged at a constant current of " + _NUM + r" C[^.]*?" + _NUM + r"\s*V")

_UNKNOWN = {"unknown", "not_disclosed"}
_CATEGORICAL_FIELDS = ("battery_format", "battery_chemistry", "battery_anode", "battery_electrolyte",
                       "battery_manufacturer", "battery_formation")
_NUMERIC_FIELDS = ("ambient_temperature", "charge_rate_max", "charge_cutoff_voltage", "has_cv_stage",
                   "discharge_rate", "discharge_cutoff_voltage", "n_charge_stages",
                   "soc_range_lo", "soc_range_hi")
_PROTOCOL_FIELDS = ("charge_protocol_id", "discharge_protocol_id")

_MappingHelper = None


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _protocol_id(text: str) -> str | None:
    if not text.strip():
        return None
    return "p" + hashlib.sha1(_normalize(text).encode()).hexdigest()[:10]


def _mapping_class():
    """Load the official cell→prompt mapping (vendored reference tree, read-only)."""
    global _MappingHelper
    if _MappingHelper is None:
        root = Path(__file__).resolve().parents[1] / "reference" / "BatteryMFormer"
        if not root.is_dir():
            raise LookupError(f"BatteryMFormer reference tree missing: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from Prompts.Mapping_helper import Mapping_helper
        _MappingHelper = Mapping_helper
    return _MappingHelper


def registered_prompt(cell_stem: str) -> tuple[str, str]:
    """Return ``(prompt_text, module_name)`` for one cell stem (e.g. ``CALB_0_B182``)."""
    helper = _mapping_class()
    try:
        # The vendored helper raises bare Exception for unmapped prefixes.
        text = helper(prompt_type="PROTOCOL", cell_name=cell_stem).do_mapping()
    except Exception as error:  # noqa: BLE001 - normalized into LookupError for callers
        raise LookupError(f"no BatteryMFormer registered prompt maps to cell {cell_stem!r}: {error}") from error
    if not isinstance(text, str) or len(text) < 50:
        raise LookupError(f"BatteryMFormer registered prompt for {cell_stem!r} is missing or empty")
    return text, PROMPT_REFERENCE


def parse_registered(cell_stem: str) -> dict:
    """Parse one cell's official prompt into ``{"metadata": ..., "reference": ...}``.

    ``metadata`` carries exactly the ``BMF_NUMERIC``/``BMF_CATEGORICAL`` whitelist fields
    (absent entries omitted, so the metadata mask stays explicit); ``reference`` carries
    the prompt provenance plus the registered nominal capacity for cross-checking
    against the raw-pickle value (the whitelist keeps the measured one).
    """
    text, module_name = registered_prompt(cell_stem)
    metadata: dict = {}
    for field, pattern in _SPEC_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            continue
        if field in ("ambient_temperature", "nominal_capacity"):
            value = float(match.group(1))
        elif field == "soc_range":
            lo, hi = (float(group) for group in match.groups())
            metadata["soc_range_lo"], metadata["soc_range_hi"] = lo / 100.0, hi / 100.0
            continue
        else:
            value = _normalize(match.group(1))
            if value in _UNKNOWN:
                continue
        if field in _CATEGORICAL_FIELDS:
            metadata[field] = value
        elif field == "ambient_temperature":
            metadata[field] = value
        # nominal_capacity is cross-check only; the whitelist keeps the raw-derived value.
    operating = text.split("Operating condition:")[-1]
    sentences = re.split(r"(?<=\.)\s+", operating)
    charge = " ".join(s for s in sentences if "charg" in s)
    discharge = " ".join(s for s in sentences if "discharg" in s)
    rates = [float(v) for v in _CHARGE_RATE.findall(charge)]
    if rates:
        metadata["charge_rate_max"] = max(rates)
        metadata["n_charge_stages"] = float(len(rates))
    voltages = [float(v) for _, v in _CHARGE_VOLTAGE.findall(charge)]
    voltages += [float(v) for v in _CHARGE_CV_VOLTAGE.findall(charge)]
    if voltages:
        metadata["charge_cutoff_voltage"] = max(voltages)
    if charge.strip():
        metadata["has_cv_stage"] = 1.0 if re.search(r"constant[- ]voltage", charge) else 0.0
        metadata["charge_protocol_id"] = _protocol_id(charge)
    rates = [float(v) for v in _DISCHARGE_RATE.findall(discharge)]
    if rates:
        metadata["discharge_rate"] = max(rates)
    voltages = [float(v) for _, v in _DISCHARGE_VOLTAGE.findall(discharge)]
    if voltages:
        metadata["discharge_cutoff_voltage"] = min(voltages)
    if discharge.strip():
        metadata["discharge_protocol_id"] = _protocol_id(discharge)
    unknown = (set(metadata) - set(_CATEGORICAL_FIELDS) - set(_NUMERIC_FIELDS) - set(_PROTOCOL_FIELDS))
    if unknown:
        raise ValueError(f"registered metadata parser produced non-whitelist fields for {cell_stem!r}: {sorted(unknown)}")
    nominal = _SPEC_PATTERNS["nominal_capacity"].search(text)
    return {"metadata": metadata,
            "reference": {"profile": "bmf", "reference": module_name, "cell_stem": cell_stem,
                          "prompt_nominal_capacity_Ah": float(nominal.group(1)) if nominal else None}}


def registered_fields(cell_stem: str) -> dict:
    """Whitelist metadata only (the manifest ``metadata`` block for the bmf profile)."""
    return parse_registered(cell_stem)["metadata"]
