"""Small numeric utilities shared by the physiological estimators."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def median_filter(x: Sequence[float], k: int = 5) -> np.ndarray:
    """1-D running median filter (edge-padded)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    k = max(1, int(k) // 2 * 2 + 1)  # odd
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.zeros_like(x)
    for i in range(x.size):
        out[i] = np.median(xp[i:i + k])
    return out


def zscore_outlier_mask(x: Sequence[float], thresh: float = 3.0) -> np.ndarray:
    """Return a boolean mask that rejects outliers beyond ``thresh`` * MAD/std."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return np.ones(x.size, dtype=bool)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return np.abs(x - med) <= thresh * 1.4826 * mad


def smooth(x: Sequence[float], window: int = 5) -> np.ndarray:
    """Moving-average smoothing."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")


def interp_nan(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs (used on tachograms before PSD)."""
    x = np.asarray(x, dtype=np.float64).copy()
    nans = np.isnan(x)
    if not np.any(nans):
        return x
    idx = np.arange(x.size)
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def moving_slope(t: Sequence[float], y: Sequence[float], window_sec: float) -> float:
    """Least-squares slope of y vs t over the trailing ``window_sec`` seconds.

    Note: least squares is not robust to outliers; prefer
    :func:`theil_sen_slope` for noisy physiological signals.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.size < 3:
        return 0.0
    lo = t[-1] - window_sec
    m = t >= lo
    if m.sum() < 3:
        m = np.ones(t.size, dtype=bool)
    A = np.vstack([t[m], np.ones(m.sum())]).T
    slope, _ = np.linalg.lstsq(A, y[m], rcond=None)[0]
    return float(slope)


def theil_sen_slope(t: Sequence[float], y: Sequence[float]) -> float:
    """Robust slope = median of all pairwise slopes (Theil-Sen estimator).

    Much more resistant to outliers / spurious beats than least squares, which
    is what makes the "heart rate change rate" unstable when computed from noisy
    instantaneous-HR samples.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = t.size
    if n < 2 or y.size != n:
        return 0.0
    slopes: list[np.ndarray] = []
    for i in range(n - 1):
        dt = t[i + 1:] - t[i]
        dy = y[i + 1:] - y[i]
        m = np.abs(dt) > 1e-9
        if np.any(m):
            slopes.append(dy[m] / dt[m])
    if not slopes:
        return 0.0
    return float(np.median(np.concatenate(slopes)))


def median_filter(x: Sequence[float], k: int = 3) -> np.ndarray:
    """Median filter (edge-padded) to remove single-sample spikes."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    k = max(1, int(k) // 2 * 2 + 1)
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = np.median(xp[i:i + k])
    return out
