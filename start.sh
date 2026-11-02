#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

if ! command -v node >/dev/null 2>&1; then
  echo "缺少 Node.js，请先安装 Node 18+：node --version" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c 'import numpy, scipy' >/dev/null 2>&1; then
  echo "缺少 Python 依赖，请先执行：" >&2
  echo "  python3 -m venv .venv" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  pip install -r requirements.txt" >&2
  exit 1
fi

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-3200}"
export PHYSIO_PYTHON="$PYTHON_BIN"
export APLUS_PYTHON="$PYTHON_BIN"
export DEEP_PYTHON="$PYTHON_BIN"

echo "启动离线分析实验室"
echo "  本机:   http://localhost:${PORT}/"
echo "  摄像头: http://localhost:${PORT}/camera"
echo "  局域网: http://<本机IP>:${PORT}/"
echo ""

exec node server.mjs
