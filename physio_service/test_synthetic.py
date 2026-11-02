"""physio_service 自检：合成 rPPG 信号，验证搬运后的算法仍能正确恢复 HR/RR/SpO2。

运行（从 lifemirror 根目录）::

    python3 -m physio_service.test_synthetic

该测试不依赖摄像头、不依赖 Electron，可在无 GPU/NPU 环境下运行。
"""
from __future__ import annotations

import math
import sys

import numpy as np

try:
    from .physio.estimator import SignalBuffer, estimate_all
    from .server import compute
except ImportError:  # pragma: no cover
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from physio.estimator import SignalBuffer, estimate_all  # type: ignore
    from server import compute  # type: ignore


def synth_rgb(t, hr_bpm, rr_per_min, spo2_target):
    rng = np.random.default_rng(42)
    r_ratio = (100.0 - spo2_target) / 5.0
    pulse = np.sin(2 * math.pi * (hr_bpm / 60.0) * t)
    resp = np.sin(2 * math.pi * (rr_per_min / 60.0) * t)
    dc_r, dc_g, dc_b = 150.0, 110.0, 90.0
    ac_r = 0.006
    ac_b = ac_r / max(r_ratio, 1e-6)
    resp_base = 0.001 * resp
    r = dc_r * ((1 + ac_r * pulse) + resp_base)
    g = dc_g * ((1 + 0.8 * ac_r * pulse) + resp_base)
    b = dc_b * ((1 + ac_b * pulse) + resp_base)
    return np.stack([r, g, b], axis=1) + rng.normal(0, 0.05, size=(t.size, 3))


def main() -> int:
    fs, duration = 25.0, 60.0
    hr_true, rr_true, spo2_true = 75.0, 15.0, 97.0
    t = np.arange(0, duration, 1 / fs)
    rgb = synth_rgb(t, hr_true, rr_true, spo2_true)

    print(f"\n=== physio_service synthetic validation (60s @ {fs:.0f}fps) ===")
    print(f"ground truth -> HR={hr_true:.0f}, RR={rr_true:.0f}, SpO2={spo2_true:.0f}\n")
    print(f"{'path':16s} {'HR(bpm)':>9s} {'RR(/min)':>10s} {'SpO2(%)':>9s} {'quality':>9s}")
    print("-" * 58)

    ok = True
    for method in ("green", "chrom", "pos"):
        base = {"min_frames": 60,
                "heart_rate": {"band_hz": [0.7, 4.0], "order": 4},
                "respiratory_rate": {"band_hz": [0.1, 0.5], "order": 4},
                "spo2": {"a": 100.0, "b": 5.0},
                "blood_pressure": {}, "hrv": {}}
        cfg = dict(base, algorithm=method)
        buf = SignalBuffer(max_frames=int(fs * duration))
        for ti, row in zip(t, rgb):
            buf.add(ti, {"forehead": row, "left_cheek": row, "right_cheek": row})
        res = estimate_all(buf, cfg, fs)
        print(f"{method:16s} {res['heart_rate']:9.1f} {res['respiratory_rate']:10.1f} "
              f"{res['spo2']:9.1f} {res['quality']:9.3f}")
        if abs(res["heart_rate"] - hr_true) > 2.0:
            ok = False

    # 走 HTTP compute() 路径，确保序列化/接口逻辑与直接调用一致
    payload = {
        "fs": fs,
        "timestamps": t.tolist(),
        "forehead": rgb.tolist(),
        "left_cheek": rgb.tolist(),
        "right_cheek": rgb.tolist(),
    }
    res = compute(payload)
    print(f"\ncompute() via HTTP payload -> HR={res['heart_rate']} bpm, ready={res['ready']}")
    if not res.get("ready") or abs(res["heart_rate"] - hr_true) > 2.0:
        ok = False

    print("\n结果:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
