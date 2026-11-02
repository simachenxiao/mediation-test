"""rPPG 生理指标计算侧车（本地微服务）。

职责边界：
  * JS 侧（Electron 渲染进程）：MediaPipe 人脸检测 + ROI 像素采样（YCbCr 肤色），
    产出每帧的 forehead / left_cheek / right_cheek 的 (R,G,B) 均值 + 时间戳。
  * 本服务：只做纯数字信号处理（numpy + scipy），直接复用 rppg 的
    ``estimate_all``，不做任何图像/模型处理。

设计为 **stateless**：前端每次把最近一个滑动窗口的三路 ROI 均值整段 POST 过来，
服务端构建 ``SignalBuffer`` 后调用 ``estimate_all``，返回指标 JSON。这样服务端
无会话状态，崩溃/重启无影响。

运行方式（从 lifemirror 根目录）::

    python3 -m physio_service.server            # 端口随机，stdout 打印 PHYSIO_PORT=<port>
    python3 -m physio_service.server --port 8765

HTTP 接口：
    GET  /health           -> {"status":"ok","pid":<pid>}
    POST /estimate         -> 见下方 payload
    POST /shutdown         -> 优雅退出（供 Electron 关闭时调用）

POST /estimate payload::

    {
      "fs": 30.0,
      "timestamps": [0.0, 0.0333, ...],
      "forehead":    [[r,g,b], ...],   # null 表示该帧 ROI 无效
      "left_cheek":  [[r,g,b], ...],
      "right_cheek": [[r,g,b], ...]
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# 兼容两种导入方式：`python -m physio_service.server`（推荐）与打包后的入口。
try:  # 作为包运行时
    from .physio.estimator import SignalBuffer, estimate_all
except ImportError:  # pragma: no cover - 直接脚本运行时兜底
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from physio.estimator import SignalBuffer, estimate_all  # type: ignore

_CONFIG_PATH = os.path.join(
    getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))),
    "config.json",
)
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    CONFIG = json.load(_f)

# 人员信息 dummy：后续会替换为轮询外部接口。结构尽量简单、扁平。
# 可在 config.json 的 "person" 字段覆盖。
DEFAULT_PERSON = {
    "name": "张某某",
    "age": 31,
    "gender": "男",
    "caseId": "A-2025-0915",
}
PERSON_INFO = {**DEFAULT_PERSON, **(CONFIG.get("person") or {})}


def _to_rgb(value):
    """None/非法 -> None（estimator 会当作 NaN 跳过）；否则转 float ndarray(3)。"""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    return arr


def compute(payload: dict):
    """把前端传来的滑动窗口转成 SignalBuffer 并计算全部指标。"""
    fs = float(payload.get("fs") or 0.0)
    timestamps = payload.get("timestamps") or []
    forehead = payload.get("forehead") or []
    l_cheek = payload.get("left_cheek") or []
    r_cheek = payload.get("right_cheek") or []

    n = len(timestamps)
    if n == 0:
        return {"n_samples": 0, "ready": False}

    buf = SignalBuffer(max_frames=n)
    for i in range(n):
        buf.add(
            float(timestamps[i]),
            {
                "forehead": _to_rgb(forehead[i]) if i < len(forehead) else None,
                "left_cheek": _to_rgb(l_cheek[i]) if i < len(l_cheek) else None,
                "right_cheek": _to_rgb(r_cheek[i]) if i < len(r_cheek) else None,
            },
        )
    return estimate_all(buf, CONFIG, fs)


class Handler(BaseHTTPRequestHandler):
    server_version = "lifemirror-physio/1.0"

    # ---- helpers ---------------------------------------------------------
    def _send_json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # 仅本机使用，允许同源/本地页面直接访问
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8"))

    def log_message(self, fmt, *args):  # 交给调用方（Electron）决定是否记录
        if os.environ.get("PHYSIO_VERBOSE") == "1":
            sys.stderr.write("[physio] " + (fmt % args) + "\n")

    # ---- routes ----------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "pid": os.getpid()})
        elif self.path == "/person":
            # dummy：固定返回，供前端轮询联调；将来替换为外部人员接口
            self._send_json(200, PERSON_INFO)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/shutdown":
            self._send_json(200, {"status": "bye"})
            # 在独立线程里关，确保响应先发出去
            import threading

            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.path != "/estimate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            payload = self._read_json()
            self._send_json(200, compute(payload))
        except Exception as exc:  # noqa: BLE001 - 兜底返回错误给前端
            self._send_json(400, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    ap = argparse.ArgumentParser(description="lifemirror rPPG physio sidecar")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = 由系统分配空闲端口")
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    port = httpd.server_address[1]
    # 端口握手：Electron 主进程读取这一行拿端口
    print(f"PHYSIO_PORT={port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
