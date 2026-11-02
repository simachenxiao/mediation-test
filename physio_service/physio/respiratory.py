"""Respiratory-rate estimation from the rPPG colour trace.

Respiration modulates the rPPG signal in three ways (amplitude, frequency and
baseline wander).  We extract the respiratory band from the *raw* green channel
(which retains the low-frequency baseline wander that colour-combination methods
such as POS deliberately remove) and combine it with the amplitude envelope of
the pulse band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import filters


@dataclass
class RespiratoryResult:
    breaths_per_min: float
    confidence: float


def _interp_peak(psd: np.ndarray, idx: int) -> tuple[float, float]:
    if idx <= 0 or idx >= len(psd) - 1:
        return float(idx), float(psd[idx])
    a, b, c = psd[idx - 1], psd[idx], psd[idx + 1]
    denom = a - 2 * b + c
    if abs(denom) < 1e-12:
        return float(idx), float(b)
    delta = max(-1.0, min(1.0, 0.5 * (a - c) / denom))
    return float(idx + delta), float(b - 0.25 * (a - c) * delta)


def estimate(raw_green: np.ndarray, fs: float, band_hz=(0.1, 0.5),
             order: int = 4) -> RespiratoryResult:
    """Estimate respiratory rate from a raw (green) colour trace.

    ``raw_green`` should be the *unnormalised* mean green intensity of a face ROI.
    """
    raw = np.asarray(raw_green, dtype=np.float64)
    n = raw.size
    if n < 12 or fs <= 0:
        return RespiratoryResult(0.0, 0.0)

    base = filters.detrend(raw)

    # (1) baseline wander: respiratory frequency of the raw trace
    resp_wander = filters.bandpass(base, fs, band_hz[0], band_hz[1], order)

    # (2) amplitude modulation: envelope of the pulse band
    pulse = filters.bandpass(base, fs, 0.7, 4.0, order)
    env = np.abs(pulse)
    resp_env = filters.bandpass(filters.detrend(env), fs, band_hz[0], band_hz[1], order)

    r1 = resp_wander / (resp_wander.std() + 1e-9)
    r2 = resp_env / (resp_env.std() + 1e-9)
    combo = r1 + 0.7 * r2

    nfft = max(1 << int(np.ceil(np.log2(n * 4))), 1024)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(combo * win, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    lo = max(0, int(np.searchsorted(freqs, band_hz[0])))
    hi = min(len(freqs), int(np.searchsorted(freqs, band_hz[1])) + 1)
    if hi - lo < 2:
        return RespiratoryResult(0.0, 0.0)
    band = spec[lo:hi]
    k = int(np.argmax(band)) + lo
    f_bin, amp = _interp_peak(spec, k)
    nb = max(0, k - 2)
    ne = min(len(spec), k + 3)
    peak_power = spec[nb:ne].sum()
    conf = float(np.clip(peak_power / max(band.sum(), 1e-12), 0.0, 1.0))

    f_hz = f_bin * fs / nfft
    return RespiratoryResult(float(f_hz * 60.0), conf)
