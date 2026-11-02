"""Blood-oxygen-saturation (SpO2) estimation.

A pulse oximeter compares the pulsatile (AC) and non-pulsatile (DC) absorption
at two wavelengths (typically red ~660 nm and infrared ~940 nm).  An ordinary
RGB camera has red and blue channels which can be used as a rough stand-in:

    R = (AC_red / DC_red) / (AC_blue / DC_blue)
    SpO2 = a - b * R

The linear coefficients ``a`` and ``b`` are device/lighting specific and MUST be
calibrated against a reference oximeter.  The value reported here is therefore
an *estimate*, not a medical measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import filters


@dataclass
class SpO2Result:
    spo2: float
    ratio: float
    confidence: float


def _ac_dc(ch: np.ndarray, fs: float, band_hz=(0.7, 4.0)) -> tuple[float, float]:
    ch = np.asarray(ch, dtype=np.float64)
    dc = float(np.mean(ch))
    if dc <= 1e-9 or ch.size < 12:
        return 0.0, dc
    ac = filters.bandpass(filters.detrend(ch), fs, band_hz[0], band_hz[1], 4)
    ac_amp = float(np.std(ac))
    return ac_amp, dc


def estimate(rgb_mean: np.ndarray, fs: float, a: float = 100.0,
             b: float = 5.0, band_hz=(0.7, 4.0)) -> SpO2Result:
    """Estimate SpO2 from (n_frames, 3) mean RGB values (columns R, G, B)."""
    rgb = np.asarray(rgb_mean, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] < 3 or rgb.shape[0] < 12:
        return SpO2Result(0.0, 0.0, 0.0)

    red = rgb[:, 0]
    blue = rgb[:, 2]

    ac_r, dc_r = _ac_dc(red, fs, band_hz)
    ac_b, dc_b = _ac_dc(blue, fs, band_hz)

    if dc_r <= 1e-9 or dc_b <= 1e-9 or ac_b <= 1e-12:
        return SpO2Result(0.0, 0.0, 0.0)

    r_red = ac_r / dc_r
    r_blue = ac_b / dc_b
    ratio = r_red / r_blue if r_blue > 1e-12 else 0.0

    spo2 = float(np.clip(a - b * ratio, 0.0, 100.0))

    # confidence grows with pulsatile amplitude (a proxy for signal quality)
    conf = float(np.clip(ac_r / (dc_r * 1e-3) * 0.05, 0.0, 1.0))
    return SpO2Result(spo2, float(ratio), conf)
