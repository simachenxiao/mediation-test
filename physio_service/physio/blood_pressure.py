"""Cuffless blood-pressure estimation from rPPG waveform morphology.

True cuffless BP is an open research problem.  The pragmatic approach used here
estimates BP from a set of PPG waveform features combined through a linear
regression model that *must be calibrated* against a cuff sphygmomanometer for
each subject/device:

    SBP = bias + w_ptt*PTT + w_hr*HR + w_pir*PIR + w_aix*AIx
    DBP = bias + w_ptt*PTT + w_hr*HR + w_pir*PIR + w_aix*AIx

Feature definitions:
  PTT  - pulse-transit-time surrogate: lag between the forehead and cheek pulse
         waves (seconds, via cross-correlation).  Larger lag -> lower BP.
  HR   - heart rate (bpm)
  PIR  - photoplethysmogram intensity ratio  AC/DC  (vascular tone proxy)
  AIx  - augmentation index  (reflected-wave strength / pulse strength)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import filters


@dataclass
class BPFeatures:
    ptt_sec: float = 0.0
    hr_bpm: float = 0.0
    pir: float = 0.0
    aix: float = 0.0


@dataclass
class BloodPressureResult:
    sbp: float
    dbp: float
    features: BPFeatures = field(default_factory=BPFeatures)


def pulse_transit_time(sig_a: np.ndarray, sig_b: np.ndarray, fs: float) -> float:
    """Estimate the time lag of ``sig_b`` relative to ``sig_a`` in seconds.

    Positive value means ``sig_a`` leads ``sig_b``.
    """
    a = np.asarray(sig_a, dtype=np.float64)
    b = np.asarray(sig_b, dtype=np.float64)
    if a.size < 8 or b.size < 8 or fs <= 0:
        return 0.0
    a = filters.detrend(a)
    b = filters.detrend(b)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    # cross-correlation lag (in samples), peak must be within a heartbeat
    max_lag = min(n // 2, int(fs * 0.6))
    corr = np.correlate(a, b, mode="full")  # len 2n-1, index n-1 is zero lag
    zero = n - 1
    seg = corr[zero - max_lag:zero + max_lag + 1]
    lag = int(np.argmax(seg)) - max_lag
    return float(lag) / fs


def _extract_features(sig: np.ndarray, fs: float) -> tuple[float, float]:
    """Return (PIR, AIx) from a pulse waveform.

    PIR = AC/DC.  AIx is estimated from the asymmetry of the pulse: the ratio of
    the first derivative's positive energy around systole to the total pulse
    energy.
    """
    sig = np.asarray(sig, dtype=np.float64)
    if sig.size < 8 or fs <= 0:
        return 0.0, 0.0

    dc = float(np.mean(sig))
    if dc <= 1e-9:
        return 0.0, 0.0
    ac = filters.bandpass(filters.detrend(sig), fs, 0.7, 4.0, 4)
    ac_amp = float(np.std(ac))
    pir = ac_amp / dc

    # augmentation index: ratio of the late-systolic reflected wave to the main
    # systolic wave, approximated from the pulse's second derivative.
    d1 = np.gradient(ac)
    d2 = np.gradient(d1)
    up = ac > 0
    main_energy = float(np.sum(d2[up] ** 2)) if np.any(up) else 1e-12
    reflected = float(np.sum(d2[~up] ** 2))
    aix = reflected / (main_energy + 1e-12)

    return float(pir), float(aix)


def estimate(sig: np.ndarray, hr_bpm: float, fs: float,
             ref_sig: np.ndarray | None = None,
             coef: dict | None = None) -> BloodPressureResult:
    """Estimate SBP/DBP.

    ``sig``      - main pulse waveform (forehead)
    ``hr_bpm``   - heart rate
    ``ref_sig``  - secondary ROI waveform (cheek) for the PTT surrogate
    ``coef``     - calibrated regression coefficients (see config.yaml)
    """
    coef = coef or {}
    pir, aix = _extract_features(sig, fs)

    ptt = 0.0
    if ref_sig is not None:
        ptt = pulse_transit_time(sig, ref_sig, fs)

    feats = BPFeatures(ptt_sec=ptt, hr_bpm=float(hr_bpm), pir=pir, aix=aix)

    sbp = (
        float(coef.get("sbp_bias", 110.0))
        + float(coef.get("sbp_ptt", -0.8)) * ptt
        + float(coef.get("sbp_hr", 0.15)) * hr_bpm
        + float(coef.get("sbp_pir", 6.0)) * pir
        + float(coef.get("sbp_aix", 0.5)) * aix
    )
    dbp = (
        float(coef.get("dbp_bias", 70.0))
        + float(coef.get("dbp_ptt", -0.4)) * ptt
        + float(coef.get("dbp_hr", 0.05)) * hr_bpm
        + float(coef.get("dbp_pir", 3.0)) * pir
        + float(coef.get("dbp_aix", 0.2)) * aix
    )

    # physiological sanity clamps
    sbp = float(np.clip(sbp, 60.0, 250.0))
    dbp = float(np.clip(dbp, 40.0, 160.0))
    if sbp <= dbp:
        dbp = sbp * 0.62
    return BloodPressureResult(sbp, dbp, feats)
