"""HRV and respiration-rate estimators derived from Open-rppg BVP output."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, find_peaks, filtfilt, welch

from .common import clean_dict, clean_number


class HrvRespirationEstimator:
    def __init__(self) -> None:
        self.last_valid_hrv: dict[str, Any] = {}
        self.last_respiration_rate: float | None = None
        self.last_rr_source: str | None = None

    def reset(self) -> None:
        self.last_valid_hrv = {}
        self.last_respiration_rate = None
        self.last_rr_source = None

    def collect(self, model: Any, window_seconds: float) -> tuple[dict[str, Any], float | None, str | None]:
        hrv: dict[str, Any] = {}
        if model.now >= min(18.0, window_seconds * 0.5):
            try:
                metrics = model.hr(start=-window_seconds, return_hrv=True) or {}
                hrv = clean_dict(metrics.get("hrv") or {})
            except Exception:
                hrv = {}

        fallback_hrv = self._estimate_hrv_from_bvp(model, window_seconds)
        if fallback_hrv:
            hrv = {**fallback_hrv, **{key: value for key, value in hrv.items() if value is not None}}
        hrv = self._normalise_hrv_values(hrv)

        rr = self._normalise_breathing_rate(hrv.get("breathingrate"))
        rr_source = "hrv" if rr is not None else None

        if rr is None:
            rr = self._estimate_rr_from_bvp(model, window_seconds)
            rr_source = "bvp" if rr is not None else None

        if hrv:
            if rr is not None:
                hrv["breathingrate"] = rr
            self.last_valid_hrv = hrv
        elif self.last_valid_hrv:
            hrv = self.last_valid_hrv

        if rr is None and self.last_respiration_rate is not None:
            rr = self.last_respiration_rate
            rr_source = "cached"
        elif rr is not None:
            self.last_respiration_rate = rr
            self.last_rr_source = rr_source

        return hrv, rr, rr_source

    def _normalise_hrv_values(self, hrv: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(hrv)
        ranges = {
            "ibi": (330.0, 1500.0),
            "sdnn": (0.0, 180.0),
            "rmssd": (0.0, 220.0),
            "pnn50": (0.0, 100.0),
            "bpm": (40.0, 180.0),
            "LF/HF": (0.0, 20.0),
        }
        for key, (low, high) in ranges.items():
            value = clean_number(cleaned.get(key), 1)
            if value is None or not low <= value <= high:
                cleaned.pop(key, None)
            else:
                cleaned[key] = value
        return cleaned

    def _estimate_hrv_from_bvp(self, model: Any, window_seconds: float) -> dict[str, float]:
        try:
            bvp, ts = model.bvp(start=-window_seconds)
        except Exception:
            return {}
        if bvp is None or ts is None or len(bvp) < model.fps * 8:
            return {}

        peaks = self._pulse_peaks_from_bvp(np.asarray(bvp, dtype=float), float(model.fps))
        if len(peaks) < 4:
            return {}

        intervals = np.diff(peaks) / float(model.fps) * 1000.0
        intervals = self._clean_pulse_intervals(intervals)
        if len(intervals) < 4:
            return {}

        diffs = np.diff(intervals)
        sdnn = float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0.0
        rmssd = float(np.sqrt(np.mean(diffs**2))) if len(diffs) else 0.0
        if sdnn > 180.0 or rmssd > 220.0:
            return {}

        return {
            "bpm": round(60000.0 / float(np.mean(intervals)), 1),
            "ibi": round(float(np.mean(intervals)), 0),
            "sdnn": round(sdnn, 1),
            "rmssd": round(rmssd, 1),
            "pnn50": round(float(np.mean(np.abs(diffs) > 50.0) * 100.0) if len(diffs) else 0.0, 1),
        }

    def _clean_pulse_intervals(self, intervals: np.ndarray) -> np.ndarray:
        intervals = intervals[np.isfinite(intervals)]
        intervals = intervals[(intervals >= 330.0) & (intervals <= 1500.0)]
        if len(intervals) < 4:
            return np.array([], dtype=float)

        median_interval = float(np.median(intervals))
        if not 400.0 <= median_interval <= 1300.0:
            return np.array([], dtype=float)

        tolerance = max(120.0, median_interval * 0.22)
        intervals = intervals[np.abs(intervals - median_interval) <= tolerance]
        if len(intervals) < 4:
            return np.array([], dtype=float)

        q1, q3 = np.percentile(intervals, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            intervals = intervals[(intervals >= q1 - 1.5 * iqr) & (intervals <= q3 + 1.5 * iqr)]
        return intervals

    def _pulse_peaks_from_bvp(self, bvp: np.ndarray, fs: float) -> np.ndarray:
        if len(bvp) < fs * 4 or not np.isfinite(bvp).all():
            return np.array([], dtype=int)
        y = bvp - np.mean(bvp)
        std = np.std(y)
        if std < 1e-6:
            return np.array([], dtype=int)
        y = y / std

        distance = max(1, int(0.32 * fs))
        candidates = []
        for signal in (y, -y):
            peaks = find_peaks(signal, distance=distance, prominence=0.35)[0]
            if len(peaks) < 4:
                peaks = find_peaks(signal, distance=distance, prominence=0.2)[0]
            score = self._score_pulse_peaks(peaks, fs)
            candidates.append((score, peaks))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _score_pulse_peaks(self, peaks: np.ndarray, fs: float) -> float:
        if len(peaks) < 3:
            return 0.0
        intervals = np.diff(peaks) / fs
        valid = intervals[(intervals >= 0.33) & (intervals <= 1.5)]
        if len(valid) < 2:
            return 0.0
        consistency = 1.0 / (1.0 + float(np.std(valid) / (np.mean(valid) + 1e-6)))
        return len(valid) * consistency

    def _normalise_breathing_rate(self, value: Any) -> float | None:
        number = clean_number(value, 4)
        if number is None:
            return None
        if number <= 2.0:
            number *= 60.0
        if not 4.0 <= number <= 40.0:
            return None
        return round(number, 1)

    def _estimate_rr_from_bvp(self, model: Any, window_seconds: float) -> float | None:
        try:
            bvp, ts = model.bvp(start=-window_seconds, raw=True)
        except Exception:
            return None
        if bvp is None or ts is None or len(bvp) < model.fps * 12:
            return None

        y = np.asarray(bvp, dtype=float)
        if not np.isfinite(y).all() or np.std(y) < 1e-6:
            return None

        fs = float(model.fps)
        y = y - np.mean(y)
        try:
            b, a = butter(2, [0.08, 0.5], btype="bandpass", fs=fs)
            breathing_signal = filtfilt(b, a, y)
            freqs, power = welch(
                breathing_signal,
                fs=fs,
                nperseg=min(len(breathing_signal), int(fs * 24)),
                nfft=4096,
            )
        except Exception:
            return None

        mask = (freqs >= 0.08) & (freqs <= 0.5)
        if not np.any(mask):
            return None
        rr = float(freqs[mask][np.argmax(power[mask])] * 60.0)
        if not 4.0 <= rr <= 40.0:
            return None
        return round(rr, 1)
