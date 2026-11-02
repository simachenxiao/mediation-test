"""Per-person physiological estimation orchestration.

A :class:`SignalBuffer` accumulates, for a single tracked person, the mean RGB
values of each facial ROI over time.  Every ``step_sec`` the
:func:`estimate_all` function slides a window over the buffer and produces the
full set of physiological metrics.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import blood_pressure, heart_rate, hrv, respiratory, signal as rppg_signal, spo2
from ..utils import metrics as U


@dataclass
class SignalBuffer:
    """Rolling buffer of ROI colour traces for one person."""

    max_frames: int
    timestamps: deque = field(default_factory=deque)
    # roi_name -> deque of (R, G, B) float triples (or None if unavailable)
    rois: dict[str, deque] = field(default_factory=dict)

    def add(self, t: float, roi_means: dict[str, np.ndarray | None]) -> None:
        self.timestamps.append(float(t))
        for name, mean in roi_means.items():
            dq = self.rois.setdefault(name, deque(maxlen=self.max_frames))
            if mean is None:
                dq.append(np.full(3, np.nan))
            else:
                dq.append(np.asarray(mean, dtype=np.float64))
        while len(self.timestamps) > self.max_frames:
            self.timestamps.popleft()
            for dq in self.rois.values():
                if len(dq) > len(self.timestamps):
                    dq.popleft()

    def roi_array(self, name: str) -> np.ndarray:
        """Return (n, 3) array for an ROI; NaNs for missing frames."""
        dq = self.rois.get(name)
        if not dq:
            return np.zeros((0, 3))
        return np.vstack(list(dq))

    def times(self) -> np.ndarray:
        return np.asarray(list(self.timestamps), dtype=np.float64)

    def __len__(self) -> int:
        return len(self.timestamps)


def _clean_rgb(rgb: np.ndarray) -> np.ndarray:
    """Remove rows with any NaN channel and return a finite (n, 3) array."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.size == 0:
        return rgb.reshape(0, 3)
    ok = np.isfinite(rgb).all(axis=1)
    return rgb[ok]


def estimate_all(buf: SignalBuffer, cfg: dict, fs: float) -> dict[str, Any]:
    """Compute all physiological metrics from a signal buffer.

    Returns a dict with keys: heart_rate, spo2, respiratory_rate,
    blood_pressure, hrv, heart_rate_change, quality, n_samples.
    """
    hr_cfg = cfg.get("heart_rate", {})
    rr_cfg = cfg.get("respiratory_rate", {})
    spo2_cfg = cfg.get("spo2", {})
    bp_cfg = cfg.get("blood_pressure", {})
    hrv_cfg = cfg.get("hrv", {})
    algorithm = cfg.get("algorithm", "pos")

    forehead = _clean_rgb(buf.roi_array("forehead"))
    l_cheek = _clean_rgb(buf.roi_array("left_cheek"))
    r_cheek = _clean_rgb(buf.roi_array("right_cheek"))

    n_samples = len(forehead)
    if n_samples < cfg.get("min_frames", 60):
        return {"n_samples": n_samples, "ready": False}

    pulse = rppg_signal.extract(forehead, method=algorithm)

    hr = heart_rate.estimate(
        pulse, fs, band_hz=hr_cfg.get("band_hz", [0.7, 4.0]),
        order=hr_cfg.get("order", 4),
    )

    # SpO2 uses the full-face colour trace (mean of forehead + cheeks)
    all_rgb = np.vstack([a for a in (forehead, l_cheek, r_cheek) if a.size])
    if all_rgb.size:
        spo2_res = spo2.estimate(
            all_rgb, fs, a=spo2_cfg.get("a", 100.0), b=spo2_cfg.get("b", 5.0),
        )
    else:
        spo2_res = spo2.SpO2Result(0.0, 0.0, 0.0)

    resp = respiratory.estimate(
        forehead[:, 1], fs, band_hz=rr_cfg.get("band_hz", [0.1, 0.5]),
        order=rr_cfg.get("order", 4),
    )

    # PTT surrogate from forehead vs cheek when both are available.
    # The two cheeks may contain a different number of valid (non-NaN) frames,
    # so truncate both to the common length before averaging.
    cheek = None
    if l_cheek.size and r_cheek.size:
        n = min(l_cheek.shape[0], r_cheek.shape[0])
        cheek = (l_cheek[:n] + r_cheek[:n]) / 2.0
    elif l_cheek.size:
        cheek = l_cheek
    elif r_cheek.size:
        cheek = r_cheek
    cheek_pulse = None
    if cheek is not None and cheek.shape[0] >= 12:
        cheek_pulse = rppg_signal.extract(cheek, method=algorithm)

    bp = blood_pressure.estimate(pulse, hr.bpm, fs, ref_sig=cheek_pulse, coef=bp_cfg)

    hrv_res = hrv.estimate(hr.ibi_ms, bands=hrv_cfg)

    # ------------------------------------------------------------------
    # "heart rate change rate" (bpm/min): the (robust) slope of a smoothed
    # instantaneous-HR series versus time, expressed per minute.
    #
    # The naive version (least-squares slope of 60/IBI over a 10 s window) is
    # extremely noisy: one spurious/short IBI produces a huge instantaneous-HR
    # spike and thus absurd values (e.g. -2000 bpm/min).  We instead:
    #   1. reject implausible inter-beat intervals (30..220 bpm),
    #   2. remove single-sample spikes with a median filter,
    #   3. clip to a physiological range,
    #   4. fit the slope with the Theil-Sen estimator (median of pairwise
    #      slopes) over a longer trailing window, and
    #   5. clamp the final rate to +/- 60 bpm/min.
    # ------------------------------------------------------------------
    times = buf.times()
    hr_change = 0.0
    # primary: sliding-window FFT heart rate -> smooth HR(t) series -> robust slope
    if pulse.size >= int(6 * fs) and times.size == pulse.size:
        win = int(6 * fs)
        step = max(1, int(2 * fs))
        hrs: list[float] = []
        ts: list[float] = []
        for end in range(win, pulse.size + 1, step):
            seg = pulse[end - win:end]
            r = heart_rate.estimate(seg, fs,
                                    band_hz=hr_cfg.get("band_hz", [0.7, 4.0]),
                                    order=hr_cfg.get("order", 4))
            if r.confidence >= 0.15 and 30.0 < r.bpm < 220.0:
                hrs.append(r.bpm)
                ts.append(float(times[end - 1]))
        if len(hrs) >= 4:
            slope = U.theil_sen_slope(np.asarray(ts), np.asarray(hrs))  # bpm/s
            hr_change = float(np.clip(slope * 60.0, -60.0, 60.0))

    # fallback: robust slope of the (median-filtered) instantaneous HR from IBI
    if hr_change == 0.0 and hr.peaks.size >= 4:
        pk_t = times[hr.peaks]
        ibi_s = np.diff(pk_t)
        ok = (ibi_s >= 0.27) & (ibi_s <= 2.0)
        if ok.sum() >= 4:
            inst = U.median_filter(60.0 / ibi_s[ok], 3)
            inst = np.clip(inst, 30.0, 220.0)
            mid = 0.5 * (pk_t[:-1] + pk_t[1:])[ok]
            m = np.ones(mid.size, dtype=bool)
            for w in (30.0, 20.0, 15.0, 10.0):
                mm = mid >= (mid[-1] - w)
                if mm.sum() >= 5:
                    m = mm
                    break
            if m.sum() >= 5:
                hr_change = float(np.clip(
                    U.theil_sen_slope(mid[m], inst[m]) * 60.0, -60.0, 60.0))

    # aggregate quality
    quality = float(np.mean([hr.confidence, resp.confidence, spo2_res.confidence]))

    return {
        "ready": True,
        "n_samples": n_samples,
        "heart_rate": round(hr.bpm, 1),
        "heart_rate_confidence": round(hr.confidence, 3),
        "respiratory_rate": round(resp.breaths_per_min, 1),
        "respiratory_confidence": round(resp.confidence, 3),
        "spo2": round(spo2_res.spo2, 1),
        "spo2_confidence": round(spo2_res.confidence, 3),
        "blood_pressure": {"sbp": round(bp.sbp, 1), "dbp": round(bp.dbp, 1)},
        "ptt_ms": round(bp.features.ptt_sec * 1000.0, 1),
        "pir": round(bp.features.pir, 4),
        "hrv": {
            "sdnn_ms": round(hrv_res.sdnn_ms, 1),
            "rmssd_ms": round(hrv_res.rmssd_ms, 1),
            "pnn50": round(hrv_res.pnn50, 1),
            "lf_hf": round(hrv_res.lf_hf, 2),
        },
        "heart_rate_change_bpm_per_min": round(hr_change, 2),
        "quality": round(quality, 3),
    }
