"""Heart-rate-variability (HRV) analysis.

HRV is computed from the inter-beat-interval (IBI) tachogram:

* time domain  - SDNN, RMSSD, pNN50
* frequency    - VLF / LF / HF band powers (Welch PSD on the interpolated,
  de-trended tachogram), plus the LF/HF ratio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from ..utils import metrics as U


@dataclass
class HRVResult:
    mean_hr: float = 0.0
    sdnn_ms: float = 0.0
    rmssd_ms: float = 0.0
    pnn50: float = 0.0
    lf_hf: float = 0.0
    lf_power: float = 0.0
    hf_power: float = 0.0
    n_beats: int = 0


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(m):
        return 0.0
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapz(psd[m], freqs[m]))


def estimate(ibi_ms: np.ndarray, bands: dict | None = None) -> HRVResult:
    """Compute HRV metrics from inter-beat intervals (ms)."""
    bands = bands or {}
    ibi = np.asarray(ibi_ms, dtype=np.float64)
    if ibi.size < 3:
        return HRVResult(n_beats=int(ibi.size))

    mean_ibi = float(np.mean(ibi))
    mean_hr = 60000.0 / mean_ibi if mean_ibi > 0 else 0.0

    # time domain
    sdnn = float(np.std(ibi))
    diff = np.abs(np.diff(ibi))
    rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0
    pnn50 = float(np.mean(diff > 50.0) * 100.0) if diff.size else 0.0

    # frequency domain on a 4 Hz re-sampled, de-trended tachogram
    vlf = tuple(bands.get("vlf", [0.0033, 0.04]))
    lf = tuple(bands.get("lf", [0.04, 0.15]))
    hf = tuple(bands.get("hf", [0.15, 0.40]))
    lf_power = hf_power = lf_hf = 0.0

    if ibi.size >= 8:
        t = np.cumsum(np.concatenate([[0.0], ibi[:-1]])) / 1000.0  # seconds
        fs = 4.0
        t_end = t[-1]
        n_resample = max(int(t_end * fs), 8)
        t_grid = np.linspace(0.0, t_end, n_resample)
        rr_grid = np.interp(t_grid, t, ibi)
        rr_grid = U.interp_nan(rr_grid)
        rr_grid = sps.detrend(rr_grid, type="linear")

        freqs, psd = sps.welch(rr_grid, fs=fs, nperseg=min(len(rr_grid), 256))
        lf_power = _band_power(freqs, psd, lf)
        hf_power = _band_power(freqs, psd, hf)
        lf_hf = lf_power / hf_power if hf_power > 1e-12 else 0.0

    return HRVResult(
        mean_hr=mean_hr, sdnn_ms=sdnn, rmssd_ms=rmssd, pnn50=pnn50,
        lf_hf=lf_hf, lf_power=lf_power, hf_power=hf_power,
        n_beats=int(ibi.size),
    )
