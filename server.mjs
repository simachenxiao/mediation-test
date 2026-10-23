// 独立离线分析服务（仅依赖 Node 内置模块）。
//
// 功能：
//   * 提供 test/public 下的页面（微表情 + 生命体征离线分析）
//   * 以只读方式复用 ../code 里的 MediaPipe / onnxruntime-web / 模型文件
//   * 拉起 ../code/physio_service 的 Python 侧车，并代理 /physio/estimate
//
// 不会写入或修改 ../code 里的任何文件。
//
// 启动：node server.mjs    （或 npm start）
// 环境变量：PORT（默认 3200）、HOST（默认 0.0.0.0）、PHYSIO_PYTHON

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const CODE = path.resolve(ROOT, '..', 'code');
const PUBLIC = path.join(ROOT, 'public');
const NODE_MODULES = path.join(CODE, 'node_modules');

// 只读依赖检查：缺东西时给出明确提示，而不是 500
const REQUIRED = [
  path.join(NODE_MODULES, '@mediapipe/tasks-vision'),
  path.join(NODE_MODULES, 'onnxruntime-web/dist'),
  path.join(CODE, 'models'),
  path.join(CODE, 'physio_service'),
];

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.tflite': 'application/octet-stream',
  '.data': 'application/octet-stream',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
};

// --------------------------------------------------------------------------- //
// Python 侧车管理
// --------------------------------------------------------------------------- //
const physio = {
  child: null,
  port: null,
  starting: null,
  lastError: null,
  lastExit: null,
  command: null,
  logs: [],
};

function sidecarLog(line) {
  const s = String(line).replace(/\s+$/, '');
  if (!s) return;
  physio.logs.push(s);
  if (physio.logs.length > 80) physio.logs.shift();
  console.error('[physio] ' + s);
}

function stopPhysio() {
  if (physio.child) {
    try {
      physio.child.kill();
    } catch {
      /* ignore */
    }
  }
  physio.child = null;
  physio.port = null;
  physio.starting = null;
}

function launchPhysio() {
  const py =
    process.env.PHYSIO_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
  physio.command = { cmd: py, args: ['-m', 'physio_service.server'], cwd: CODE };
  physio.lastError = null;
  physio.lastExit = null;
  physio.port = null;
  sidecarLog(`启动: ${py} -m physio_service.server (cwd=${CODE})`);

  return new Promise((resolve, reject) => {
    let settled = false;
    const ok = (port) => {
      if (settled) return;
      settled = true;
      physio.port = port;
      resolve(port);
    };
    const fail = (err) => {
      physio.lastError = err.message;
      if (!settled) {
        settled = true;
        reject(err);
      }
    };

    let child;
    try {
      child = spawn(py, ['-m', 'physio_service.server'], {
        cwd: CODE,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: { ...process.env },
        windowsHide: true,
      });
    } catch (err) {
      fail(new Error(`无法启动 Python：${err.message}`));
      return;
    }
    physio.child = child;

    const timer = setTimeout(
      () => fail(new Error('侧车启动超时（20s 内未收到 PHYSIO_PORT）')),
      20000
    );

    child.stdout.on('data', (buf) => {
      for (const line of buf.toString().split(/\r?\n/)) {
        const m = /PHYSIO_PORT=(\d+)/.exec(line);
        if (m) {
          clearTimeout(timer);
          sidecarLog(`就绪 pid=${child.pid} port=${m[1]}`);
          ok(Number(m[1]));
        } else if (line.trim()) {
          sidecarLog('stdout: ' + line);
        }
      }
    });
    child.stderr.on('data', (buf) => {
      for (const line of buf.toString().split(/\r?\n/)) {
        if (line.trim()) sidecarLog('stderr: ' + line);
      }
    });
    child.on('error', (err) => {
      clearTimeout(timer);
      fail(new Error(`无法启动 Python（${py}）：${err.message}`));
    });
    child.on('exit', (code, signal) => {
      clearTimeout(timer);
      physio.lastExit = { code, signal, at: new Date().toISOString() };
      if (code !== 0 && code !== null) physio.lastError = `侧车退出 code=${code}`;
      sidecarLog(`侧车退出 code=${code} signal=${signal}`);
      physio.child = null;
      physio.port = null;
      physio.starting = null;
      if (!settled) fail(new Error(physio.lastError || `侧车退出 code=${code}`));
    });
  });
}

function ensurePhysio() {
  if (physio.port) return Promise.resolve(physio.port);
  if (!physio.starting) {
    physio.starting = launchPhysio().catch((err) => {
      physio.starting = null;
      throw err;
    });
  }
  return physio.starting;
}

async function restartPhysio() {
  stopPhysio();
  return ensurePhysio();
}

function physioStatus() {
  return {
    running: !!physio.child && physio.port != null,
    port: physio.port,
    pid: physio.child?.pid ?? null,
    command: physio.command,
    lastError: physio.lastError,
    lastExit: physio.lastExit,
    logs: physio.logs.slice(-50),
  };
}

// --------------------------------------------------------------------------- //
// HTTP 工具
// --------------------------------------------------------------------------- //
function setHeaders(res, cache = 'no-cache') {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('Cache-Control', cache);
}

function sendJson(res, code, obj) {
  setHeaders(res);
  const body = Buffer.from(JSON.stringify(obj), 'utf8');
  res.writeHead(code, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': body.length,
  });
  res.end(body);
}

function safeJoin(root, rel) {
  const target = path.normalize(path.join(root, rel));
  const base = path.normalize(root);
  if (target !== base && !target.startsWith(base + path.sep)) return null;
  return target;
}

function sendFile(res, filePath, cache) {
  let stat;
  try {
    stat = fs.statSync(filePath);
  } catch {
    return false;
  }
  if (!stat.isFile()) return false;
  setHeaders(res, cache);
  res.writeHead(200, {
    'Content-Type': MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream',
    'Content-Length': stat.size,
  });
  fs.createReadStream(filePath).pipe(res);
  return true;
}

function readBody(req, limit = 8 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (c) => {
      size += c.length;
      if (size > limit) {
        reject(new Error('请求体过大'));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

// 静态目录映射：/mediapipe /wasm /ort /models 都只读指向 ../code
const MOUNTS = [
  { prefix: '/mediapipe/', dir: path.join(NODE_MODULES, '@mediapipe/tasks-vision'), cache: 'public, max-age=86400' },
  { prefix: '/wasm/', dir: path.join(NODE_MODULES, '@mediapipe/tasks-vision/wasm'), cache: 'public, max-age=86400' },
  { prefix: '/ort/', dir: path.join(NODE_MODULES, 'onnxruntime-web/dist'), cache: 'public, max-age=86400' },
  { prefix: '/models/', dir: path.join(CODE, 'models'), cache: 'public, max-age=86400' },
];

async function handle(req, res) {
  const url = new URL(req.url, 'http://localhost');
  const pathname = decodeURIComponent(url.pathname);

  // ---- API ----
  if (pathname === '/health') {
    return sendJson(res, 200, { status: 'ok', mode: 'standalone-lab', codeDir: CODE });
  }

  if (pathname === '/physio/health') {
    return sendJson(res, 200, physioStatus());
  }

  if (pathname === '/physio/restart' && req.method === 'POST') {
    try {
      await restartPhysio();
      return sendJson(res, 200, { ok: true, ...physioStatus() });
    } catch (err) {
      return sendJson(res, 502, { ok: false, error: err.message, ...physioStatus() });
    }
  }

  if (pathname === '/physio/estimate' && req.method === 'POST') {
    try {
      const body = await readBody(req);
      const port = await ensurePhysio();
      const upstream = await fetch(`http://127.0.0.1:${port}/estimate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      });
      const text = await upstream.text();
      setHeaders(res);
      res.writeHead(upstream.status, {
        'Content-Type': 'application/json; charset=utf-8',
        'Content-Length': Buffer.byteLength(text),
      });
      res.end(text);
    } catch (err) {
      sendJson(res, 502, { error: err.message });
    }
    return;
  }

  if (req.method !== 'GET' && req.method !== 'HEAD') {
    return sendJson(res, 405, { error: 'method not allowed' });
  }

  // ---- 页面 ----
  if (pathname === '/' || pathname === '/lab' || pathname === '/lab.html') {
    return sendFile(res, path.join(PUBLIC, 'lab.html'), 'no-cache')
      ? undefined
      : sendJson(res, 404, { error: 'lab.html not found' });
  }

  // ---- 只读依赖 ----
  for (const m of MOUNTS) {
    if (pathname.startsWith(m.prefix)) {
      const rel = pathname.slice(m.prefix.length);
      const file = safeJoin(m.dir, rel);
      if (file && sendFile(res, file, m.cache)) return;
      return sendJson(res, 404, { error: 'not found', path: pathname });
    }
  }

  // ---- 页面自己的静态资源 ----
  const file = safeJoin(PUBLIC, pathname.replace(/^\/+/, ''));
  if (file && sendFile(res, file, 'no-cache')) return;

  sendJson(res, 404, { error: 'not found', path: pathname });
}

// --------------------------------------------------------------------------- //
// 启动
// --------------------------------------------------------------------------- //
const missing = REQUIRED.filter((p) => !fs.existsSync(p));
if (missing.length) {
  console.error('\n缺少只读依赖，请确认以下路径存在：');
  for (const p of missing) console.error('  - ' + p);
  console.error('\n本目录只读取 ../code，不会修改它。\n');
  process.exit(1);
}

const PORT = Number(process.env.PORT || 3200);
const HOST = process.env.HOST || '0.0.0.0';

const server = http.createServer((req, res) => {
  handle(req, res).catch((err) => {
    console.error('[lab] 请求失败:', err);
    if (!res.headersSent) sendJson(res, 500, { error: err.message });
    else res.end();
  });
});

server.listen(PORT, HOST, () => {
  console.log(`🧪 离线分析实验室: http://localhost:${PORT}/`);
  console.log(`   页面目录: ${PUBLIC}`);
  console.log(`   只读依赖: ${CODE}`);
  ensurePhysio().catch((err) => console.error('[physio] 预启动失败:', err.message));
});

function shutdown() {
  stopPhysio();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 1500).unref();
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
process.on('exit', stopPhysio);
