"""Pure tensor layout conversions for extracted input protocols.

This module contains no BatteryMFormer/BatteryLife model or data-loader code.
The server-side extractor writes source-normalized arrays; training only uses
these explicit layout conversions.
"""
from __future__ import annotations
import numpy as np


def phase_to_sequence(curves, mask=None):
    """Convert ``[..., 2, C, P]`` charge/discharge arrays to ``[..., C, 2P]``."""
    curves = np.asarray(curves)
    if curves.ndim < 4 or curves.shape[-3] != 2:
        raise ValueError("phase curves must have shape [...,2,channels,points]")
    sequence = np.concatenate((curves[..., 0, :, :], curves[..., 1, :, :]), axis=-1)
    if mask is None:
        return sequence
    mask = np.asarray(mask, bool)
    if mask.shape != curves.shape:
        raise ValueError("phase mask must match phase curves")
    return sequence, np.concatenate((mask[..., 0, :, :], mask[..., 1, :, :]), axis=-1)


def sequence_to_phase(sequence, mask=None):
    """Convert source ``[..., C, 2P]`` arrays to Adapter ``[...,2,C,P]``."""
    sequence = np.asarray(sequence)
    if sequence.ndim < 3 or sequence.shape[-1] % 2:
        raise ValueError("sequence arrays must end in an even 2P axis")
    points = sequence.shape[-1] // 2
    phase = np.stack((sequence[..., :points], sequence[..., points:]), axis=-3)
    if mask is None:
        return phase
    mask = np.asarray(mask, bool)
    if mask.shape != sequence.shape:
        raise ValueError("sequence mask must match sequence arrays")
    return phase, np.stack((mask[..., :points], mask[..., points:]), axis=-3)


def batterymformer_to_adapter(cycle_curve_data, curve_mask=None):
    """Convert BMF ``[...,3,300]`` to Adapter ``[...,2,3,150]``."""
    cycle_curve_data = np.asarray(cycle_curve_data)
    if cycle_curve_data.ndim < 3 or cycle_curve_data.shape[-2:] != (3, 300):
        raise ValueError("BatteryMFormer cycle_curve_data must use [S,3,300] channels-first layout")
    return sequence_to_phase(cycle_curve_data, curve_mask)


def batterylife_to_adapter(curve_data, curve_mask=None):
    """Convert BatteryLife channels-first ``[...,3,300]`` explicitly."""
    curve_data = np.asarray(curve_data)
    if curve_data.ndim < 3 or curve_data.shape[-2:] != (3, 300):
        raise ValueError("BatteryLife cycle_curve_data must use [S,3,300] channels-first layout")
    return sequence_to_phase(curve_data, curve_mask)
