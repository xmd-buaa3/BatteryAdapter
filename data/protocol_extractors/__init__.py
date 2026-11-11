"""Built-in BatteryMFormer/BatteryLife input-protocol extractors.

These modules read raw cell pickles and never import reference models or
training code.  They intentionally expose a small, stable per-cell schema.
"""
from data.protocol_extractors.common import discover_cells, extract_batterymformer, extract_batterylife, SkipCell

__all__ = ["discover_cells", "extract_batterymformer", "extract_batterylife", "SkipCell"]
