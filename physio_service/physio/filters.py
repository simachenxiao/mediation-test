"""Digital filters used by the rPPG signal-processing chain.

All filters are implemented with scipy Butterworth + zero-phase (filtfilt)
filtering so that no group delay is introduced into the waveforms.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def _valid_band(fs: float, low: float, high: float) -> tuple[float, float]:
    nyq = fs / 2.0
    low = max(0.01, float(low))
    high = min(nyq * 0.99, float(high))
    if high <= low:
        raise ValueError(f"invalid band [{low}, {high}] Hz for fs={fs}")
    return low, high


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase band-pass filter."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3 * order:
        return x.copy()
    low, high = _valid_band(fs, low, high)
    sos = sps.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3 * order:
        return x.copy()
    cutoff = min(cutoff, fs / 2 * 0.99)
    sos = sps.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def detrend(x: np.ndarray) -> np.ndarray:
    """Remove linear trend and DC component."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 4:
        return x - x.mean() if x.size else x
    return sps.detrend(x, type="linear")


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    if n == 1 or x.size < n:
        return x.astype(np.float64)
    kernel = np.ones(n, dtype=np.float64) / n
    return np.convolve(x, kernel, mode="same")


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Return (x - mean) / mean so that amplitudes are relative (AC/DC)."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean()
    if abs(mean) < 1e-12:
        return x - mean
    return (x - mean) / mean
