#!/usr/bin/env python3
"""FacePhys.rlap 心率侧车（仅 test 使用）。

为实时 A/B 对比提供 B 引擎：
  * 从 ../EmotiEffLib-main/rppg-demo/rppg-demo 的 vendor 中加载 Open-rppg 的
    FacePhys.rlap，模型输入 36x36、30fps、逐帧输出 BVP。
  * HTTP 接口由 test/server.mjs 代理，浏览器只和 Node 同源通信。

模型加载需要 30~40 秒，因此 /start 立即返回，/health 轮询加载状态。
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parent
DEMO = ROOT.parent / "EmotiEffLib-main" / "rppg-demo" / "rppg-demo"
VENDOR = DEMO / "vendor" / "open_rppg"

if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import numpy as np  # noqa: E402

MODEL_NAME = os.environ.get("DEEP_MODEL", "FacePhys.rlap")


class Stabilizer:
    """与 rppg-demo/vital_signs/heart_rate.py 相同的心率平滑/门控逻辑。"""

    def __init__(self) -> None:
        self.smoothed_hr: float | None = None
        self.last_update_at: float | None = None

    def reset(self) -> None:
        self.smoothed_hr = None
        self.last_update_at = None

    def update(
        self,
        raw_hr: float | None,
        sqi: float | None,
        now: float | None = None,
    ) -> tuple[float | None, str]:
        if raw_hr is None or not 35 <= raw_hr <= 190:
            return self.smoothed_hr, "holding" if self.smoothed_hr is not None else "warming"

        quality = 0.0 if sqi is None else max(0.0, min(1.0, sqi))
        now = time.time() if now is None else float(now)
        if self.smoothed_hr is None:
            if quality < 0.2:
                return None, "warming"
            self.smoothed_hr = raw_hr
            self.last_update_at = now
            return round(self.smoothed_hr, 1), "live"

        if quality < 0.18:
            return round(self.smoothed_hr, 1), "holding"

        dt = max(0.5, min(3.0, now - self.last_update_at)) if self.last_update_at else 1.0
        max_delta = 5.0 * dt
        delta = max(-max_delta, min(max_delta, raw_hr - self.smoothed_hr))
        limited_hr = self.smoothed_hr + delta
        alpha = 0.12 + quality * 0.18
        self.smoothed_hr = self.smoothed_hr * (1 - alpha) + limited_hr * alpha
        self.last_update_at = now
        return round(self.smoothed_hr, 1), "live"


class DeepSession:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.model: Any | None = None
        self.ready = False
        self.loading = False
        self.error: str | None = None
        self.load_started_at: float | None = None
        self.load_finished_at: float | None = None
        self.initial_state: Any | None = None
        self.stabilizer = Stabilizer()

    def start(self) -> dict[str, Any]:
        with self.lock:
            if self.ready or self.loading:
                return self.status()
            self.loading = True
            self.error = None
            self.load_started_at = time.time()
            self.load_finished_at = None

        def load() -> None:
            try:
                import rppg  # 延迟导入：让 HTTP 服务先起来

                model = rppg.Model(MODEL_NAME)
                model.__enter__()
                with self.lock:
                    self.model = model
                    self.initial_state = copy.deepcopy(model.state)
                    self.ready = True
                    self.loading = False
                    self.load_finished_at = time.time()
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    self.loading = False
                    self.ready = False
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.load_finished_at = time.time()

        threading.Thread(target=load, daemon=True).start()
        return self.status()

    def reset(self) -> dict[str, Any]:
        with self.lock:
            if self.loading:
                return {**self.status(), "ok": False, "error": "模型正在加载"}
            model = self.model
            if model is None:
                self.stabilizer.reset()
                return {**self.status(), "ok": True}

        # __exit__ 会等待剩余帧处理完，所以放在锁外执行；
        # 先标记 ready=False，避免 reset 期间 /frame、/result 继续访问模型。
        with self.lock:
            self.ready = False
            self.model = None
        try:
            model.__exit__(None, None, None)
            model.__enter__()
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.model = None
                self.ready = False
                self.error = f"reset 失败: {type(exc).__name__}: {exc}"
            return {**self.status(), "ok": False}

        with self.lock:
            if self.initial_state is not None:
                model.state = copy.deepcopy(self.initial_state)
            self.stabilizer.reset()
            self.model = model
            self.ready = True
        return {**self.status(), "ok": True}

    def push(self, image_data: str, timestamp: float | None) -> dict[str, Any]:
        with self.lock:
            model = self.model
            if model is None or not self.ready:
                raise RuntimeError("FacePhys 模型尚未加载完成")
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        raw = base64.b64decode(image_data, validate=True)
        import cv2

        frame_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("无法解码帧")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts = float(timestamp) if timestamp is not None else time.time()
        model.update_frame(frame_rgb, ts)
        return {"ok": True, "timestamp": ts}

    def result(
        self,
        now: float | None = None,
        wait_until: float | None = None,
        timeout: float = 20.0,
    ) -> dict[str, Any]:
        # 重放时先等模型把队列里的固定帧处理到目标时间点，
        # 否则 /result 可能读到上一秒的状态，导致不可重复。
        if wait_until is not None:
            deadline = time.time() + max(1.0, float(timeout))
            while time.time() < deadline:
                with self.lock:
                    model = self.model
                    ready = self.ready
                if model is None or not ready:
                    break
                try:
                    current = float(model.now)
                except Exception:  # noqa: BLE001
                    break
                if current >= float(wait_until) - 0.35:
                    break
                time.sleep(0.02)

        with self.lock:
            model = self.model
            ready = self.ready
            loading = self.loading
            error = self.error
            duration = float(model.now) if model is not None else 0.0
            latency = float(model.latency) if model is not None else 0.0
        if model is None or not ready:
            return {
                "ready": False,
                "loading": loading,
                "error": error,
                "hr": None,
                "rawHr": None,
                "sqi": None,
                "status": "loading" if loading else "warming",
                "duration": 0,
                "latency": 0,
            }

        raw_hr: float | None = None
        sqi: float | None = None
        err: str | None = None
        try:
            out = model.hr(start=-30, return_hrv=False) or {}
            value = out.get("hr")
            if value is not None and np.isfinite(float(value)):
                raw_hr = float(value)
            quality = out.get("SQI")
            if quality is not None and np.isfinite(float(quality)):
                sqi = float(quality)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"

        hr, status = self.stabilizer.update(raw_hr, sqi, now=now)
        return {
            "ready": True,
            "loading": False,
            "error": err or error,
            "hr": hr,
            "rawHr": raw_hr,
            "sqi": sqi,
            "status": status,
            "duration": round(duration, 2),
            "latency": round(latency, 3),
        }

    def stop(self) -> dict[str, Any]:
        with self.lock:
            model = self.model
            self.model = None
            self.ready = False
        if model is not None:
            try:
                model.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        return self.status()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ready": self.ready,
                "loading": self.loading,
                "error": self.error,
                "model": MODEL_NAME,
                "duration": round(float(self.model.now), 2) if self.model is not None else 0,
                "loadSec": (
                    round((self.load_finished_at or time.time()) - self.load_started_at, 1)
                    if self.load_started_at
                    else None
                ),
            }


SESSION = DeepSession()


class Handler(BaseHTTPRequestHandler):
    server_version = "TestDeepRppg/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("DEEP_VERBOSE") == "1":
            sys.stderr.write("[deep] " + (fmt % args) + "\n")

    def _json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=float).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 8_000_000:
            raise ValueError("请求体过大")
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, SESSION.status())
        elif self.path == "/result":
            self._json(200, SESSION.result())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/start":
                self._json(200, SESSION.start())
            elif self.path == "/reset":
                self._json(200, SESSION.reset())
            elif self.path == "/stop":
                self._json(200, SESSION.stop())
            elif self.path == "/result":
                payload = self._body()
                self._json(
                    200,
                    SESSION.result(
                        now=payload.get("replayTime"),
                        wait_until=payload.get("waitUntil"),
                        timeout=float(payload.get("timeout") or 20.0),
                    ),
                )
            elif self.path == "/frame":
                payload = self._body()
                self._json(
                    200,
                    SESSION.push(
                        str(payload.get("image") or ""),
                        payload.get("timestamp"),
                    ),
                )
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    ap = argparse.ArgumentParser(description="FacePhys.rlap sidecar for live A/B demo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    port = httpd.server_address[1]
    print(f"DEEP_PORT={port}", flush=True)
    print(f"[deep] model={MODEL_NAME} demo={DEMO}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SESSION.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
