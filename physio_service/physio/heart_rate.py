"""Heart-rate estimation from an rPPG waveform."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import filters


@dataclass
class HeartRateResult:
    bpm: float
    confidence: float            # 0..1 spectral quality
    peaks: np.ndarray            # peak sample indices
    ibi_ms: np.ndarray           # inter-beat intervals (ms)


def _parabolic_interp(psd: np.ndarray, idx: int) -> tuple[float, float]:
    """Refine a spectral peak by parabolic interpolation; return (freq_bin, amplitude)."""
    if idx <= 0 or idx >= len(psd) - 1:
        return float(idx), float(psd[idx])
    a, b, c = psd[idx - 1], psd[idx], psd[idx + 1]
    denom = a - 2 * b + c
    if abs(denom) < 1e-12:
        return float(idx), float(b)
    delta = 0.5 * (a - c) / denom
    delta = max(-1.0, min(1.0, delta))
    amp = b - 0.25 * (a - c) * delta
    return float(idx + delta), float(amp)


def estimate(sig: np.ndarray, fs: float, band_hz=(0.7, 4.0),
             order: int = 4) -> HeartRateResult:
    """Estimate HR via band-pass + FFT with sub-bin interpolation.

    ``sig`` should be a normalized rPPG waveform (mean ~ 0).
    """
    sig = np.asarray(sig, dtype=np.float64)
    n = sig.size
    if n < 12 or fs <= 0:
        return HeartRateResult(0.0, 0.0, np.array([], dtype=int), np.array([], dtype=float))

    bp = filters.bandpass(filters.detrend(sig), fs, band_hz[0], band_hz[1], order)

    # --- spectral estimation (Welch-style with Hann window + zero padding) ---
    win = np.hanning(n)
    nfft = max(1 << int(np.ceil(np.log2(n * 4))), 1024)
    spec = np.abs(np.fft.rfft(bp * win, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    lo = max(0, int(np.searchsorted(freqs, band_hz[0])))
    hi = min(len(freqs), int(np.searchsorted(freqs, band_hz[1])) + 1)
    if hi - lo < 2:
        return HeartRateResult(0.0, 0.0, np.array([], dtype=int), np.array([], dtype=float))

    band = spec[lo:hi]
    k = int(np.argmax(band)) + lo
    f_bin, amp = _parabolic_interp(spec, k)

    # --- spectral SNR / confidence (fraction of band power near the peak) ---
    total = spec[lo:hi].sum()
    nb = max(0, k - 2)
    ne = min(len(spec), k + 3)
    peak_power = spec[nb:ne].sum()
    conf = float(np.clip(peak_power / max(total, 1e-12), 0.0, 1.0))

    f_hz = f_bin * fs / nfft
    bpm = f_hz * 60.0

    # --- peak detection for IBI / HRV ---
    peaks, ibi = _detect_peaks(bp, fs, expected_bpm=bpm)
    if peaks.size >= 2 and bpm <= 0:
        bpm = 60.0 / (np.mean(ibi) / 1000.0)

    return HeartRateResult(float(bpm), conf, peaks, ibi)


def _detect_peaks(sig: np.ndarray, fs: float, expected_bpm: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Detect systolic peaks; return (indices, inter-beat intervals in ms)."""
    if sig.size < 4:
        return np.array([], dtype=int), np.array([], dtype=float)
    from scipy import signal as sps

    # expected inter-beat distance: half the expected period (rejects dicrotic
    # notches / secondary bumps that would double-count each heartbeat)
    if expected_bpm > 0:
        min_dist = max(1, int(0.5 * (60.0 / expected_bpm) * fs))
    else:
        min_dist = max(1, int(fs * 60.0 / 240.0))

    # require peaks to stand out from the local noise floor
    prominence = max(0.2 * float(np.std(sig)), 1e-6)

    # light smoothing suppresses dicrotic notches / secondary bumps
    kernel = np.ones(5, dtype=np.float64) / 5.0
    sig = np.convolve(sig, kernel, mode="same")

    peaks, _ = sps.find_peaks(sig, distance=min_dist, prominence=prominence)
    peaks = np.asarray(peaks, dtype=int)

    if peaks.size >= 2:
        ibi = np.diff(peaks) / fs * 1000.0
        # reject physiologically implausible intervals (30 .. 220 bpm)
        ok = (ibi >= 272.0) & (ibi <= 2000.0)
        peaks = peaks[np.concatenate([[True], ok])]
        ibi = np.diff(peaks) / fs * 1000.0
    else:
        ibi = np.array([], dtype=float)
    return peaks, ibi
