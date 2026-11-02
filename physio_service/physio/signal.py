"""rPPG colour-signal extraction.

Given the mean R/G/B intensity of a facial skin ROI over time, reduce the three
channels to a single pulse signal.  Three established methods are provided:

* ``GREEN``  - the raw green channel (simple, robust, widely used)
* ``CHROM``  - chrominance based (de Haan & Jeanne, 2013)
* ``POS``    - plane-orthogonal-to-skin (Wang et al., 2017), default
"""
from __future__ import annotations

import numpy as np

from . import filters


def _temporal_normalize(ch: np.ndarray) -> np.ndarray:
    """C / mean(C) - 1 for each channel (per-channel relative amplitude)."""
    ch = np.asarray(ch, dtype=np.float64)
    m = ch.mean()
    if abs(m) < 1e-9:
        return ch - m
    return ch / m - 1.0


def green(rgb_mean: np.ndarray) -> np.ndarray:
    """Return the normalized green channel.

    ``rgb_mean`` shape is (n_frames, 3), columns are R, G, B.
    """
    rgb_mean = np.asarray(rgb_mean, dtype=np.float64)
    g = rgb_mean[:, 1]
    return filters.normalize_signal(g)


def chrom(rgb_mean: np.ndarray) -> np.ndarray:
    """CHROM method (de Haan & Jeanne 2013)."""
    rgb_mean = np.asarray(rgb_mean, dtype=np.float64)
    if rgb_mean.shape[0] < 8:
        return green(rgb_mean)

    rn = _temporal_normalize(rgb_mean[:, 0])
    gn = _temporal_normalize(rgb_mean[:, 1])
    bn = _temporal_normalize(rgb_mean[:, 2])

    xs = 3.0 * rn - 2.0 * gn
    ys = 1.5 * rn + gn - 1.5 * bn
    sx = xs.std()
    sy = ys.std()
    if sy < 1e-12:
        return filters.normalize_signal(xs)
    s = xs - (sx / sy) * ys
    return filters.normalize_signal(s)


def pos(rgb_mean: np.ndarray) -> np.ndarray:
    """POS method (Wang et al. 2017)."""
    rgb_mean = np.asarray(rgb_mean, dtype=np.float64)
    n = rgb_mean.shape[0]
    if n < 8:
        return green(rgb_mean)

    # per-channel temporal normalization
    c = np.vstack(
        [_temporal_normalize(rgb_mean[:, i]) for i in range(3)]
    )  # (3, n)

    # projection onto the plane orthogonal to [1, 1, 1]
    s1 = c[1] - c[2]          # G - B
    s2 = -2.0 * c[0] + c[1] + c[2]  # -2R + G + B
    std1, std2 = s1.std(), s2.std()
    if std2 < 1e-12:
        s = s1
    else:
        alpha = std1 / std2
        s = s1 + alpha * s2

    return filters.normalize_signal(s)


_METHODS = {"green": green, "chrom": chrom, "pos": pos}


def extract(rgb_mean: np.ndarray, method: str = "pos") -> np.ndarray:
    """Extract a single rPPG waveform from (n_frames, 3) mean RGB values."""
    fn = _METHODS.get(str(method).lower(), pos)
    return fn(np.asarray(rgb_mean, dtype=np.float64))
