#!/usr/bin/env python3
"""A+ 心率估计侧车（纯信号处理，不加载深度模型）。

A+ 不是 B 的替代品，而是在 A（POS + 单次 FFT）基础上补上 B 的后处理思路：

  1. 平滑先验去趋势，替代 A 的线性去趋势；
  2. 频带下界从 0.7Hz 提到 0.8Hz，避开慢漂移最容易造假的区域；
  3. 只保留 POS 方法，额头 / 脸颊多 ROI + 双频带交叉候选；
  4. Welch 多段平均，替代单次 FFT 取最大峰；
  5. 峰值置信度 + 自相关 SQI + 第一名/第二名差距 + ROI 一致性；
  6. 心率稳定器：限速 + EMA + 低质量保持；
  7. 返回 rawHr / hr / quality / reliable，页面可以同时看原始值和稳定值。

输入与 /physio/estimate 完全一致：
{
  "fs": 30,
  "timestamps": [ ... ],
  "forehead": [[r,g,b], ...] | [null, ...],
  "left_cheek": [...],
  "right_cheek": [...]
}

只依赖 numpy / scipy，启动不需要模型，通常几十毫秒内就绪。
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
from scipy import signal as sps
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

MODEL_NAME = "A+ (POS + smooth detrend + Welch + SQI)"
BAND_HZ = (0.8, 3.5)
LOW_BAND_HZ = (0.6, 3.5)
DEFAULT_FS = 30.0
EPS = 1e-12
RELIABLE_MIN_CONF = 0.01


# --------------------------------------------------------------------------- //
# 数值工具
# --------------------------------------------------------------------------- //
def _clean_rgb(rows: Any) -> np.ndarray:
    out: list[list[float]] = []
    for row in rows or []:
        if row is None:
            continue
        arr = np.asarray(row, dtype=np.float64)
        if arr.shape != (3,) or not np.isfinite(arr).all():
            continue
        out.append(arr.tolist())
    return np.asarray(out, dtype=np.float64).reshape(-1, 3) if out else np.zeros((0, 3))


def _mean_roi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size and b.size:
        n = min(len(a), len(b))
        return (a[:n] + b[:n]) / 2.0
    return a if a.size else b


def _temporal_normalize(ch: np.ndarray) -> np.ndarray:
    ch = np.asarray(ch, dtype=np.float64)
    m = float(np.mean(ch)) if ch.size else 0.0
    if abs(m) < 1e-9:
        return ch - m
    return ch / m - 1.0


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.mean(x)) if x.size else 0.0
    if abs(m) < 1e-12:
        return x - m
    return (x - m) / m


def _smooth_detrend(x: np.ndarray, fs: float, min_freq: float = 0.8) -> np.ndarray:
    """平滑先验去趋势（参考 open-rppg 的 detrend）。

    用二阶差分矩阵作为平滑惩罚，估计一条平滑基线并从信号中减掉。
    比线性 detrend 更能处理弯曲的灯光/运动慢漂移。
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 8:
        return x - float(np.mean(x)) if n else x

    lam = 50.0 * (fs / 30.0) ** 2 * (0.5 / max(0.1, min_freq)) ** 2
    d = [
        np.ones(n - 2),
        -2.0 * np.ones(n - 2),
        np.ones(n - 2),
    ]
    D = diags(d, [0, 1, 2], shape=(n - 2, n), format="csc")
    A = eye(n, format="csc") + (lam**2) * (D.T @ D)
    baseline = spsolve(A, x)
    return x - baseline


def _bandpass(x: np.ndarray, fs: float, low: float = BAND_HZ[0], high: float = BAND_HZ[1], order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3 * order:
        return x.copy()
    nyq = fs / 2.0
    low = max(0.01, float(low))
    high = min(nyq * 0.99, float(high))
    if high <= low:
        return x.copy()
    sos = sps.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def _pos(rgb: np.ndarray) -> np.ndarray:
    n = rgb.shape[0]
    if n < 8:
        return _normalize(rgb[:, 1])
    c = np.vstack([_temporal_normalize(rgb[:, i]) for i in range(3)])
    s1 = c[1] - c[2]
    s2 = -2.0 * c[0] + c[1] + c[2]
    std1, std2 = float(np.std(s1)), float(np.std(s2))
    s = s1 if std2 < EPS else s1 + (std1 / std2) * s2
    return _normalize(s)


METHODS = {
    "POS": _pos,
}


def _parabolic(psd: np.ndarray, idx: int) -> tuple[float, float]:
    if idx <= 0 or idx >= len(psd) - 1:
        return float(idx), float(psd[idx])
    a, b, c = float(psd[idx - 1]), float(psd[idx]), float(psd[idx + 1])
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-12:
        return float(idx), b
    delta = 0.5 * (a - c) / denom
    delta = max(-1.0, min(1.0, delta))
    return float(idx + delta), b - 0.25 * (a - c) * delta


def _welch_hr(x: np.ndarray, fs: float, band: tuple[float, float] = BAND_HZ) -> tuple[float, float, float]:
    """返回 (bpm, confidence, top2_margin)。"""
    n = int(x.size)
    if n < int(4 * fs):
        return 0.0, 0.0, 1.0

    nperseg = min(n, max(32, int(8 * fs)))
    noverlap = max(0, min(nperseg // 2, nperseg - 1))
    try:
        freqs, psd = sps.welch(
            x,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=4096,
            detrend=False,
            scaling="density",
        )
    except Exception:
        return 0.0, 0.0, 1.0

    lo = int(np.searchsorted(freqs, band[0]))
    hi = int(np.searchsorted(freqs, band[1])) + 1
    if hi - lo < 2:
        return 0.0, 0.0, 1.0

    band = psd[lo:hi]
    k_local = int(np.argmax(band))
    k = lo + k_local
    peak = float(psd[k])
    total = float(np.sum(band))
    conf = float(np.clip(peak / max(total, EPS), 0.0, 1.0))

    # 第一名与“离开主峰 ±0.2Hz 后的第二名”的比值。
    f_peak = float(freqs[k])
    mask = np.abs(freqs[lo:hi] - f_peak) >= 0.20
    alt = band[mask]
    second = float(np.max(alt)) if alt.size else 0.0
    margin = peak / max(second, EPS)

    f_bin, _ = _parabolic(psd, k)
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    bpm = float((freqs[k] + (f_bin - k) * df) * 60.0)
    if not (25.0 <= bpm <= 230.0):
        bpm = float(freqs[k] * 60.0)
    return bpm, conf, margin


def _sqi(x: np.ndarray, fs: float) -> float:
    """自相关 SQI：波形延迟一段时间后还像不像自己。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < int(2 * fs):
        return 0.0
    x = x - float(np.mean(x))
    sd = float(np.std(x))
    if sd < EPS:
        return 0.0
    x = x / sd
    ac = np.correlate(x, x, mode="full")[n - 1 :]
    if ac.size == 0 or abs(ac[0]) < EPS:
        return 0.0
    ac = ac / ac[0]
    min_lag = max(1, int(fs / 3.0))
    max_lag = min(ac.size - 1, int(fs / 0.5))
    if min_lag >= max_lag:
        return 0.0
    return float(np.clip(np.max(ac[min_lag : max_lag + 1]), 0.0, 1.0))


def _weighted_median(items: list[tuple[float, float]]) -> float:
    """items = [(value, weight), ...]"""
    if not items:
        return 0.0
    items = sorted(items, key=lambda p: p[0])
    total = sum(max(0.0, w) for _, w in items)
    if total <= EPS:
        return float(items[len(items) // 2][0])
    acc = 0.0
    for value, weight in items:
        acc += max(0.0, weight)
        if acc >= total / 2.0:
            return float(value)
    return float(items[-1][0])


def _candidate(
    rgb: np.ndarray,
    fs: float,
    roi: str,
    method: str,
    band_name: str,
    band: tuple[float, float],
    detrend_min_freq: float,
) -> dict[str, Any] | None:
    if rgb.shape[0] < int(6 * fs):
        return None
    fn = METHODS[method]
    pulse = fn(rgb)
    pulse = _smooth_detrend(pulse, fs, min_freq=detrend_min_freq)
    pulse = _bandpass(pulse, fs, low=band[0], high=band[1])
    hr, conf, margin = _welch_hr(pulse, fs, band=band)
    if not (30.0 <= hr <= 220.0):
        return None
    sqi = _sqi(pulse, fs)
    margin_score = float(np.clip((margin - 1.0) / 1.5, 0.0, 1.0))
    score = 0.45 * conf + 0.35 * sqi + 0.20 * margin_score
    return {
        "roi": roi,
        "method": method,
        "band": band_name,
        "bandLow": band[0],
        "bandHigh": band[1],
        "hr": round(hr, 1),
        "confidence": round(conf, 3),
        "sqi": round(sqi, 3),
        "margin": round(margin, 3),
        "score": round(score, 3),
    }



def _cluster_candidates(candidates: list[dict[str, Any]], tol: float = 6.0) -> list[dict[str, Any]]:
    """按心率把候选聚成簇；优先看不同方法/ROI 的共识，而不是单个最高分。"""
    remaining = sorted(candidates, key=lambda c: c["score"], reverse=True)
    clusters: list[dict[str, Any]] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        rest: list[dict[str, Any]] = []
        for c in remaining:
            if abs(c["hr"] - seed["hr"]) <= tol:
                group.append(c)
            else:
                rest.append(c)
        remaining = rest

        methods = {c["method"] for c in group}
        rois = {c["roi"] for c in group}
        weight = sum(float(c["score"]) for c in group)
        center = _weighted_median([(c["hr"], max(c["score"], 1e-6)) for c in group])
        clusters.append(
            {
                "hr": round(center, 1),
                "members": group,
                "supportMethods": len(methods),
                "supportRois": len(rois),
                "weight": round(weight, 3),
                "methods": sorted(methods),
                "rois": sorted(rois),
                "bandLow": min(float(c.get("bandLow", BAND_HZ[0])) for c in group),
                "bandHigh": max(float(c.get("bandHigh", BAND_HZ[1])) for c in group),
            }
        )
    return clusters


def _select_cluster(
    candidates: list[dict[str, Any]], prior_hr: float | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """返回主簇和全部簇。

    排序依据：
      1. 不同 ROI 数量（额头/脸颊）
      2. 候选分数总和
      3. 与上一轮稳定值的接近程度（防止在 76/111 两个簇之间来回跳）
    """
    clusters = _cluster_candidates(candidates)
    for cl in clusters:
        prior_bonus = 0.0
        if prior_hr is not None and abs(cl["hr"] - float(prior_hr)) <= 8.0:
            prior_bonus = 0.35
        cl["priorBonus"] = prior_bonus
        # POS-only 时 supportMethods 恒为 1，因此把 ROI 支持的权重提高。
        cl["rank"] = (
            cl["supportMethods"]
            + 0.75 * cl["supportRois"]
            + cl["weight"]
            + prior_bonus
        )

    primary = max(clusters, key=lambda cl: cl["rank"])

    # 次谐波检查：如果主簇约为另一簇的 2 倍，且另一簇有两个 ROI 支持，
    # 优先选低的那一簇。用于修复 45bpm 被报成 90bpm。
    if primary["hr"] >= 70.0:
        half = primary["hr"] / 2.0
        subs = [
            cl
            for cl in clusters
            if cl is not primary
            and abs(cl["hr"] - half) <= 6.0
            and cl["supportRois"] >= 2
        ]
        if subs:
            sub = max(subs, key=lambda cl: cl["rank"])
            if sub["weight"] >= 0.45 * primary["weight"]:
                primary = sub

    return primary, clusters


# --------------------------------------------------------------------------- //
# 心率稳定器（参考 B 的 HeartRateStabilizer，但阈值按 A+ 调整）
# --------------------------------------------------------------------------- //
class Stabilizer:
    def __init__(self) -> None:
        self.smoothed: float | None = None
        self.last_at: float | None = None
        self.last_good_at: float | None = None

    def reset(self) -> None:
        self.smoothed = None
        self.last_at = None
        self.last_good_at = None

    def update(
        self,
        raw: float | None,
        quality: float | None,
        reliable: bool = True,
        now: float | None = None,
    ) -> tuple[float | None, str, float | None]:
        """返回 (显示值, 状态, 已保持秒数)。

        now 用于确定性重放：传入视频时间而不是 wall clock，
        这样同一份 payload 序列每次得到相同的 dt / EMA / heldSec。
        """
        q = 0.0 if quality is None else max(0.0, min(1.0, float(quality)))
        clock = time.monotonic() if now is None else float(now)
        now = clock

        if raw is None or not (30.0 <= raw <= 220.0):
            held = round(now - self.last_good_at, 1) if self.last_good_at else None
            return self.smoothed, "holding" if self.smoothed is not None else "warming", held

        # 没有通过共识/边缘检查的候选，不参与初始化，也不更新。
        if not reliable:
            held = round(now - self.last_good_at, 1) if self.last_good_at else None
            return self.smoothed, "holding" if self.smoothed is not None else "warming", held

        if self.smoothed is None:
            if q < 0.28:
                return None, "warming", None
            self.smoothed = float(raw)
            self.last_at = now
            self.last_good_at = now
            return round(self.smoothed, 1), "live", None

        if q < 0.22:
            held = round(now - self.last_good_at, 1) if self.last_good_at else None
            return round(self.smoothed, 1), "holding", held

        dt = max(0.5, min(3.0, now - self.last_at)) if self.last_at else 1.0
        max_delta = 6.0 * dt
        delta = max(-max_delta, min(max_delta, float(raw) - self.smoothed))
        limited = self.smoothed + delta
        alpha = 0.18 + 0.20 * q
        self.smoothed = self.smoothed * (1.0 - alpha) + limited * alpha
        self.last_at = now
        self.last_good_at = now
        return round(self.smoothed, 1), "live", None


class AplusSession:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.stabilizer = Stabilizer()
        self.last_result: dict[str, Any] | None = None
        self.last_at: float | None = None

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.stabilizer.reset()
            self.last_result = None
            self.last_at = None
        return {"ok": True, "ready": True, "model": MODEL_NAME}

    def health(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ready": True,
                "loading": False,
                "error": None,
                "model": MODEL_NAME,
                "lastResultAt": self.last_at,
            }

    def estimate(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        fs = float(payload.get("fs") or DEFAULT_FS)
        if not (5.0 <= fs <= 120.0):
            fs = DEFAULT_FS
        timestamps = payload.get("timestamps") or []
        n = len(timestamps)
        if n < int(6 * fs):
            return {"ready": False, "error": "窗口不足 6 秒", "nSamples": n}

        try:
            t = np.asarray(timestamps, dtype=np.float64)
        except Exception:
            t = np.arange(n, dtype=np.float64) / fs
        span = float(t[-1] - t[0]) if t.size >= 2 else 0.0

        forehead = _clean_rgb(payload.get("forehead"))
        left = _clean_rgb(payload.get("left_cheek"))
        right = _clean_rgb(payload.get("right_cheek"))
        cheeks = _mean_roi(left, right)

        rois: list[tuple[str, np.ndarray]] = []
        if forehead.shape[0] >= int(6 * fs):
            rois.append(("forehead", forehead))
        if cheeks.shape[0] >= int(6 * fs):
            rois.append(("cheeks", cheeks))
        if not rois:
            return {"ready": False, "error": "没有足够的 ROI 数据", "nSamples": n}

        candidates: list[dict[str, Any]] = []
        for roi_name, rgb in rois:
            # 两边通道长度可能不同，统一截到共同长度
            if rgb.shape[0] != n:
                m = min(rgb.shape[0], n)
                rgb = rgb[:m]
            for method in METHODS:
                for band_name, band, detrend_min in (
                    ("std", BAND_HZ, BAND_HZ[0]),
                    ("low", LOW_BAND_HZ, LOW_BAND_HZ[0]),
                ):
                    got = _candidate(rgb, fs, roi_name, method, band_name, band, detrend_min)
                    if got:
                        candidates.append(got)

        if not candidates:
            return {"ready": False, "error": "所有方法都没有得到有效心率", "nSamples": n}

        primary, clusters = _select_cluster(candidates, self.stabilizer.smoothed)
        members = primary["members"]
        raw_hr = float(primary["hr"])
        support_methods = int(primary["supportMethods"])
        support_rois = int(primary["supportRois"])
        agreement = min(1.0, support_rois / 2.0)
        method_label = " + ".join(primary["methods"])
        if support_rois > 1:
            method_label += " / 多 ROI"

        near_conf = max(float(c["confidence"]) for c in members)
        near_sqi = max(float(c["sqi"]) for c in members)
        near_margin = max(float(c["margin"]) for c in members)
        margin_score = float(np.clip((near_margin - 1.0) / 1.5, 0.0, 1.0))
        base_quality = 0.45 * near_conf + 0.35 * near_sqi + 0.20 * margin_score
        quality = float(np.clip(base_quality * (0.70 + 0.30 * agreement), 0.0, 1.0))

        # 峰值太贴近所用频带下边界时，可能只是慢漂移。
        # POS-only 没有多方法可交叉验证，要求两个 ROI 同时支持 + 频谱置信度下限。
        edge = bool(abs(raw_hr - float(primary["bandLow"]) * 60.0) <= 4.0)
        reliable = bool(
            support_rois >= 2
            and not edge
            and near_conf >= RELIABLE_MIN_CONF
            and quality >= 0.30
            and near_sqi >= 0.22
            and near_margin >= 1.10
        )

        with self.lock:
            replay_time = None
            if payload.get("replay"):
                try:
                    replay_time = float(payload.get("replayTime"))
                except (TypeError, ValueError):
                    replay_time = None
            stable, status, held_sec = self.stabilizer.update(
                raw_hr, quality, reliable, now=replay_time
            )
            result = {
                "ready": True,
                "model": MODEL_NAME,
                "hr": round(float(stable), 1) if stable is not None else None,
                "rawHr": round(float(raw_hr), 1),
                "confidence": round(float(near_conf), 3),
                "sqi": round(float(near_sqi), 3),
                "margin": round(float(near_margin), 3),
                "quality": round(float(quality), 3),
                "reliable": reliable,
                "status": status,
                "heldSec": held_sec,
                "edge": edge,
                "supportMethods": support_methods,
                "supportRois": support_rois,
                "method": method_label,
                "agreement": round(float(agreement), 3),
                "bandHz": [float(primary["bandLow"]), float(primary["bandHigh"])],
                "candidates": candidates,
                "clusters": [
                    {
                        "hr": cl["hr"],
                        "supportMethods": cl["supportMethods"],
                        "supportRois": cl["supportRois"],
                        "weight": cl["weight"],
                        "methods": cl["methods"],
                        "rois": cl["rois"],
                    }
                    for cl in clusters
                ],
                "nSamples": int(n),
                "span": round(span, 2),
                "latencyMs": round((time.monotonic() - started) * 1000.0, 1),
                "replay": bool(payload.get("replay")),
            }
            self.last_result = result
            self.last_at = time.time()
            return result


SESSION = AplusSession()


# --------------------------------------------------------------------------- //
# HTTP
# --------------------------------------------------------------------------- //
class Handler(BaseHTTPRequestHandler):
    server_version = "lifemirror-aplus/1.0"

    def _json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        obj = json.loads(raw.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("body must be a JSON object")
        return obj

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, SESSION.health())
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/shutdown":
            self._json(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.path == "/reset":
            self._json(200, SESSION.reset())
            return
        if self.path != "/estimate":
            self._json(404, {"error": "not found"})
            return
        try:
            self._json(200, SESSION.estimate(self._read_json()))
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"ready": False, "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        # 避免每个请求都刷屏；错误仍由 server 输出。
        if args and str(args[1]).startswith("4"):
            print(f"[aplus] {fmt % args}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="A+ rPPG sidecar")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"APLUS_PORT={httpd.server_address[1]}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
