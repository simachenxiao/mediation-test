// 摄像头三路心率对比页（/camera）。
//
// 与 /live 的区别：
//   * 输入不是文件，而是 getUserMedia 拿到的本地摄像头流；
//   * 时间轴是自己的会话时钟（从开启那一刻算起），不是视频 currentTime；
//   * 结果不只有当前值，还保留全程历史，并画出滚动窗口 + 全程两条曲线；
//   * A / A+ / B 三路可独立开关，B 首次启用才加载模型。
//
// 链路：
//   A ：MediaPipe 人脸框 → ROI 平均 RGB → /physio/estimate（POS + FFT）
//   A+：复用 A 的采样 payload → /aplus/estimate（POS + Welch + SQI）
//   B ：画布 JPEG 帧 → /deep/frame → deep_service.py（FacePhys.rlap）→ /deep/result

import { getFaceDetector, detectFaces, largestFace, clampBox } from '/lib/face-detect.js';
import { LivePhysioA } from '/lib/physio-live.js';
import { roiRects } from '/lib/roi.js';
import { drawLineChart } from '/lib/charts.js';
import { summarizeSeries, compareSeries } from '/stat-live.js';

const $ = (id) => document.getElementById(id);

// 人脸检测间隔。ROI 每个采样帧都取，这里只决定框多久校正一次。
const DETECT_INTERVAL_SEC = 0.15;
// 框的有效期：超过这个时间没检测到，就不再采样，避免一直用旧框。
const BOX_HOLD_SEC = 1.0;
// 运动门控：框中心位移/尺寸变化超过阈值就判定为“运动中”。
const MOTION_CENTER_RATIO = 0.035;
const MOTION_SIZE_RATIO = 0.06;
const MOTION_WINDOW_SEC = 2.0;
// 历史序列上限。15Hz 采样、每次结果 1 条，长会话也不会无限增长。
const MAX_POINTS = 5000;
const ROLL_SEC = 60;
const ROI_COLORS = { forehead: '#f59e0b', left_cheek: '#06b6d4', right_cheek: '#a855f7' };
const ROI_ZH = { forehead: '额头', left_cheek: '左颊', right_cheek: '右颊' };

const els = {
  video: $('video'),
  overlay: $('overlay'),
  placeholder: $('placeholder'),
  camPill: $('camPill'),
  deepPill: $('deepPill'),
  metaDevice: $('metaDevice'),
  metaSize: $('metaSize'),
  metaElapsed: $('metaElapsed'),
  btnStart: $('btnStart'),
  btnStop: $('btnStop'),
  btnReset: $('btnReset'),
  btnExport: $('btnExport'),
  runB: $('runB'),
  showRoi: $('showRoi'),
  camSelect: $('camSelect'),
  fpsCap: $('fpsCap'),
  status: $('status'),
  log: $('log'),
  warnbar: $('warnbar'),
  badgeFps: $('badgeFps'),
  badgeFace: $('badgeFace'),
  badgeMotion: $('badgeMotion'),
  aHr: $('aHr'),
  aState: $('aState'),
  aConf: $('aConf'),
  aSpan: $('aSpan'),
  aFrames: $('aFrames'),
  aplusHr: $('aplusHr'),
  aplusState: $('aplusState'),
  aplusQuality: $('aplusQuality'),
  aplusSqi: $('aplusSqi'),
  aplusRoi: $('aplusRoi'),
  aplusRaw: $('aplusRaw'),
  aplusSpan: $('aplusSpan'),
  bHr: $('bHr'),
  bState: $('bState'),
  bSqi: $('bSqi'),
  bDur: $('bDur'),
  bRaw: $('bRaw'),
  statsBody: $('statsBody'),
  statsNote: $('statsNote'),
  diffNote: $('diffNote'),
  rollChart: $('rollChart'),
  allChart: $('allChart'),
  rollQChart: $('rollQChart'),
  rawChart: $('rawChart'),
  rollRange: $('rollRange'),
  allRange: $('allRange'),
  rawRange: $('rawRange'),
};

const overlayCtx = els.overlay.getContext('2d');
const frameCanvas = document.createElement('canvas');
const frameCtx = frameCanvas.getContext('2d');

const state = {
  stream: null,
  running: false,
  deviceLabel: '',
  elapsed: 0,
  frames: 0,
  fps: 0,
  detector: null,
  box: null,
  boxAt: -Infinity,
  detecting: false,
  lastDetectAt: -Infinity,
  lastSampleAt: -Infinity,
  motion: { level: 'ok', score: 0, lastBox: null, lastAt: 0 },
  motionSince: null,
  lastACallAt: -Infinity,
  lastAPlusCallAt: -Infinity,
  lastBSendAt: -Infinity,
  lastBErrorAt: 0,
  sendingB: false,
  deep: { ready: false, loading: false, error: null, requested: false, loadSec: null },
  deepTimer: null,
  uiTimer: null,
  resultTimer: null,
  rafId: 0,
  pendingStart: false,
  startedAt: 0,
  fpsHistory: [],
  a: new LivePhysioA({ windowSec: 30, stepSec: 1, minSpanSec: 8 }),
  aResult: null,
  aplusResult: null,
  bResult: null,
  aplusBusy: false,
  aplusError: null,
  aplusErrorAt: 0,
  aSeries: [],
  aplusSeries: [],
  bSeries: [],
  aQ: [],
  aplusQ: [],
  bQ: [],
  aRaw: [],
  aplusRaw: [],
  bRaw: [],
  faceHits: 0,
  faceTries: 0,
};

// --------------------------------------------------------------------------- //
// 小工具
// --------------------------------------------------------------------------- //
function log(msg) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  const old = els.log.textContent;
  const lines = (old === '就绪。' ? [] : old.split('\n')).concat(line);
  els.log.textContent = lines.slice(-120).join('\n');
  els.log.scrollTop = els.log.scrollHeight;
}

function setStatus(text) {
  els.status.textContent = text;
}

function num(v, digits = 0) {
  return Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : '—';
}

function pct(v) {
  return Number.isFinite(Number(v)) ? `${Math.round(Number(v) * 100)}%` : '—';
}

function setPill(el, mode, text) {
  el.className = `pill ${mode || ''}`.trim();
  el.textContent = text;
}

function pushSeries(list, x, y) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  list.push({ x, y });
  if (list.length > MAX_POINTS) list.shift();
}

function tail(list, seconds) {
  if (!list.length) return [];
  const end = list[list.length - 1].x;
  const from = end - seconds;
  let lo = 0;
  let hi = list.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid].x < from) lo = mid + 1;
    else hi = mid;
  }
  return list.slice(Math.max(0, lo - 1));
}

// --------------------------------------------------------------------------- //
// 摄像头
// --------------------------------------------------------------------------- //
async function listCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === 'videoinput');
    const current = els.camSelect.value;
    els.camSelect.innerHTML = '<option value="">默认设备</option>';
    cams.forEach((d, i) => {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = d.label || `摄像头 ${i + 1}`;
      els.camSelect.appendChild(opt);
    });
    if (cams.some((c) => c.deviceId === current)) els.camSelect.value = current;
  } catch (err) {
    log(`枚举摄像头失败：${err.message}`);
  }
}

async function openCamera(deviceId) {
  const size = { width: { ideal: 1280 }, height: { ideal: 720 } };
  let stream;
  if (deviceId) {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { deviceId: { exact: deviceId }, ...size } });
  } else {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: 'user', ...size } });
    } catch (err) {
      // 某些设备/虚拟摄像头没有 facingMode 语义，退回任意可用摄像头。
      if (err.name !== 'OverconstrainedError' && err.name !== 'NotFoundError') throw err;
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: size });
    }
  }
  closeCamera();
  state.stream = stream;
  els.video.srcObject = stream;

  await new Promise((resolve, reject) => {
    const ok = () => {
      cleanup();
      resolve();
    };
    const bad = () => {
      cleanup();
      reject(new Error('摄像头画面无法播放'));
    };
    const cleanup = () => {
      els.video.removeEventListener('loadedmetadata', ok);
      els.video.removeEventListener('error', bad);
    };
    els.video.addEventListener('loadedmetadata', ok);
    els.video.addEventListener('error', bad);
    setTimeout(ok, 2500);
  });

  await els.video.play().catch(() => {});
  state.deviceLabel = stream.getVideoTracks()[0]?.label || '摄像头';
  els.metaDevice.textContent = state.deviceLabel;
  els.placeholder.style.display = 'none';
  setPill(els.camPill, 'live', '摄像头已开启');
  await listCameras();
}

function closeCamera() {
  if (state.stream) {
    for (const track of state.stream.getTracks()) {
      try {
        track.stop();
      } catch {
        /* ignore */
      }
    }
  }
  state.stream = null;
  els.video.srcObject = null;
  els.placeholder.style.display = '';
  setPill(els.camPill, '', '摄像头未开启');
}

// --------------------------------------------------------------------------- //
// B 引擎状态
// --------------------------------------------------------------------------- //
async function pollDeep() {
  try {
    const res = await fetch('/deep/health', { cache: 'no-store' });
    const d = await res.json();
    const wasReady = state.deep.ready;
    state.deep.ready = !!d.ready;
    state.deep.loading = !!d.loading;
    state.deep.error = d.error || null;
    state.deep.loadSec = Number.isFinite(Number(d.loadSec)) ? Number(d.loadSec) : null;

    if (state.deep.error) setPill(els.deepPill, 'error', `B 不可用：${String(state.deep.error).slice(0, 28)}`);
    else if (state.deep.ready) setPill(els.deepPill, 'ready', 'B 模型就绪');
    else if (state.deep.loading)
      setPill(els.deepPill, 'loading', `B 加载中…${state.deep.loadSec ? ` ${Math.round(state.deep.loadSec)}s` : ''}`);
    else if (state.deep.requested) setPill(els.deepPill, 'loading', 'B 启动中…');
    else setPill(els.deepPill, '', 'B 未启用');

    if (state.deep.ready && !wasReady) {
      log(`B 模型就绪（${d.model || 'FacePhys.rlap'}）。`);
      updateStartButton();
      if (state.pendingStart && state.stream && !state.running) {
        state.pendingStart = false;
        startAnalysis();
      }
    }
    if (state.deep.error && els.runB.checked) {
      els.runB.checked = false;
      log('B 不可用，已关闭 B 通道。');
      updateStartButton();
    }
  } catch (err) {
    state.deep.ready = false;
    state.deep.loading = false;
    state.deep.error = err.message;
    setPill(els.deepPill, 'error', 'B 不可用');
  }
  state.deepTimer = setTimeout(pollDeep, state.deep.ready || state.deep.error ? 5000 : 1500);
}

async function requestDeepStart() {
  state.deep.requested = true;
  setPill(els.deepPill, 'loading', 'B 启动中…');
  try {
    await fetch('/deep/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });
    log('正在加载 B 模型（FacePhys.rlap，首次约 30–40s）。');
  } catch (err) {
    log(`B 启动失败：${err.message}`);
  }
  pollDeep();
}

async function resetDeep() {
  if (!state.deep.ready) return;
  setStatus('重置 B 模型状态…');
  try {
    await fetch('/deep/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });
    state.bResult = null;
    state.bSeries = [];
    state.bQ = [];
    state.bRaw = [];
    log('B 模型已重置。');
  } catch (err) {
    log(`B 重置失败：${err.message}`);
  }
}

async function resetAplus() {
  try {
    const res = await fetch('/aplus/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.aplusResult = null;
    state.aplusSeries = [];
    state.aplusQ = [];
    state.aplusRaw = [];
    state.aplusError = null;
    log('A+ 引擎已重置。');
  } catch (err) {
    state.aplusError = err.message;
    log(`A+ 重置失败：${err.message}`);
  }
}

function updateStartButton() {
  const bWaiting = els.runB.checked && !state.deep.ready && !state.deep.error;
  els.btnStart.disabled = state.running || bWaiting;
  els.btnStop.disabled = !state.running;
  els.btnReset.disabled = !state.aSeries.length && !state.aplusSeries.length && !state.bSeries.length;
  els.btnExport.disabled = els.btnReset.disabled;
  els.runB.disabled = state.running;
  els.camSelect.disabled = state.running;
  els.fpsCap.disabled = state.running;
}

// --------------------------------------------------------------------------- //
// 帧循环
// --------------------------------------------------------------------------- //
function scheduleFrame() {
  if (!state.running) return;
  state.rafId = requestAnimationFrame(() => {
    onFrame();
    scheduleFrame();
  });
}

function onFrame() {
  if (!state.running) return;
  const v = els.video;
  if (!v.videoWidth || !v.videoHeight) return;

  const now = performance.now() / 1000;
  state.elapsed = now - state.startedAt;

  // 采样帧率上限
  const fpsCap = Math.max(5, Math.min(30, Number(els.fpsCap.value) || 15));
  if (now - state.lastSampleAt < 1 / fpsCap) {
    drawOverlay();
    return;
  }
  state.lastSampleAt = now;

  const t = state.elapsed;

  if (state.detector && !state.detecting && now - state.lastDetectAt >= DETECT_INTERVAL_SEC) {
    state.lastDetectAt = now;
    state.detecting = true;
    detectFaces(state.detector, v)
      .then((dets) => {
        const hit = largestFace(dets);
        state.faceTries++;
        if (hit) {
          state.faceHits++;
          const prev = state.box;
          const box = clampBox(hit.box, v.videoWidth, v.videoHeight);
          updateMotion(prev, box, now);
          state.box = box;
          state.boxAt = now;
        }
      })
      .catch((err) => log(`人脸检测失败：${err.message}`))
      .finally(() => {
        state.detecting = false;
      });
  }

  const boxFresh = state.box && now - state.boxAt <= BOX_HOLD_SEC;
  if (boxFresh) {
    state.a.addFrame(v, state.box, t);
    state.frames++;
  }

  if (t - state.lastACallAt >= 1 && state.a.span() >= 8) {
    state.lastACallAt = t;
    state.a
      .maybeEstimate()
      .then((r) => {
        if (r) updateAResult(r, t);
      })
      .catch((err) => log(`A 估计失败：${err.message}`));
  }

  if (t - state.lastAPlusCallAt >= 1 && state.a.span() >= 12) {
    state.lastAPlusCallAt = t;
    const payload = state.a.buildPayload(state.a.latestTime());
    requestAplus(payload, t);
  }

  if (els.runB.checked && state.deep.ready && !state.sendingB && now - state.lastBSendAt >= 0.066) {
    state.lastBSendAt = now;
    sendBFrame(t);
  }

  updateFps(now);
  drawOverlay();
}

function updateFps(now) {
  state.fpsHistory.push(now);
  while (state.fpsHistory.length > 2 && now - state.fpsHistory[0] > 2) state.fpsHistory.shift();
  const span = now - state.fpsHistory[0];
  state.fps = span > 0.2 ? (state.fpsHistory.length - 1) / span : 0;
}

function updateMotion(prev, next, now) {
  if (!prev) {
    state.motion.lastBox = next;
    state.motion.lastAt = now;
    return;
  }
  const dt = Math.max(0.02, now - state.motion.lastAt);
  const cx0 = prev.originX + prev.width / 2;
  const cy0 = prev.originY + prev.height / 2;
  const cx1 = next.originX + next.width / 2;
  const cy1 = next.originY + next.height / 2;
  const centerShift = Math.hypot(cx1 - cx0, cy1 - cy0) / Math.max(1, prev.width);
  const sizeShift = Math.abs(next.width - prev.width) / Math.max(1, prev.width);
  const rate = (centerShift + sizeShift) / dt;
  const score = Math.max(centerShift / MOTION_CENTER_RATIO, sizeShift / MOTION_SIZE_RATIO);

  if (score >= 1.0) {
    if (state.motionSince == null) state.motionSince = now;
  } else if (state.motionSince != null && now - state.motionSince > MOTION_WINDOW_SEC * 0.5) {
    state.motionSince = null;
  }

  const moving = state.motionSince != null;
  const level = score >= 2.5 ? 'high' : moving ? 'moving' : score >= 0.6 ? 'slight' : 'ok';
  state.motion = { level, score, rate, lastBox: next, lastAt: now };
}

function motionLabel() {
  const m = state.motion;
  if (!state.box) return '运动 —';
  if (m.level === 'high') return '运动 过大';
  if (m.level === 'moving') return '运动 中';
  if (m.level === 'slight') return '运动 轻微';
  return '运动 小';
}

async function sendBFrame(t) {
  if (!els.runB.checked || !state.deep.ready || state.sendingB) return;
  state.sendingB = true;
  try {
    const w = Math.min(1280, els.video.videoWidth || 1280);
    const h = Math.round((w * (els.video.videoHeight || 720)) / (els.video.videoWidth || 1280));
    if (frameCanvas.width !== w || frameCanvas.height !== h) {
      frameCanvas.width = w;
      frameCanvas.height = h;
    }
    frameCtx.drawImage(els.video, 0, 0, frameCanvas.width, frameCanvas.height);
    const image = frameCanvas.toDataURL('image/jpeg', 0.7);
    const res = await fetch('/deep/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image, timestamp: t }),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      if (Date.now() - state.lastBErrorAt > 5000) {
        state.lastBErrorAt = Date.now();
        log(`B 帧发送失败 HTTP ${res.status}${text ? ': ' + text.slice(0, 120) : ''}`);
      }
    }
  } catch (err) {
    if (Date.now() - state.lastBErrorAt > 5000) {
      state.lastBErrorAt = Date.now();
      log(`B 帧发送失败：${err.message}`);
    }
  } finally {
    state.sendingB = false;
  }
}

// --------------------------------------------------------------------------- //
// A / A+ 结果
// --------------------------------------------------------------------------- //
async function requestAplus(payload, t) {
  if (state.aplusBusy) return null;
  state.aplusBusy = true;
  try {
    const res = await fetch('/aplus/estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status}${text ? ': ' + text.slice(0, 160) : ''}`);
    }
    const r = await res.json();
    state.aplusError = null;
    updateAplusResult(r, t);
    return r;
  } catch (err) {
    state.aplusError = err.message;
    if (Date.now() - state.aplusErrorAt > 5000) {
      state.aplusErrorAt = Date.now();
      log(`A+ 估计失败：${err.message}`);
    }
    return null;
  } finally {
    state.aplusBusy = false;
  }
}

function updateAResult(r, t) {
  state.aResult = r;
  if (!r || !r.ready) return;
  const hr = Number(r.heart_rate);
  const conf = Number(r.heart_rate_confidence ?? r.quality);
  if (Number.isFinite(hr) && hr > 0 && hr < 240) {
    pushSeries(state.aSeries, t, hr);
    pushSeries(state.aQ, t, Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0);
  }
}

function updateAplusResult(r, t) {
  state.aplusResult = r;
  if (!r || !r.ready) return;
  const stable = Number(r.hr);
  const raw = Number(r.rawHr);
  const shown = Number.isFinite(stable) && stable > 0 ? stable : raw;
  if (Number.isFinite(shown) && shown > 0 && shown < 240) {
    pushSeries(state.aplusSeries, t, shown);
    pushSeries(state.aplusQ, t, Number.isFinite(Number(r.quality)) ? Number(r.quality) : 0);
  }
  if (Number.isFinite(raw) && raw > 0 && raw < 240) pushSeries(state.aplusRaw, t, raw);
}

async function pollBResult() {
  if (!state.running || !els.runB.checked || !state.deep.ready) return;
  try {
    const res = await fetch('/deep/result', { cache: 'no-store' });
    const r = await res.json();
    updateBResult(r);
  } catch (err) {
    log(`B 结果获取失败：${err.message}`);
  }
}

function updateBResult(r) {
  if (!r) return;
  state.bResult = r;
  const hr = Number(r.rawHr ?? r.hr);
  const dur = Number(r.duration) || 0;
  const t = state.elapsed;
  if (r.ready && dur >= 28 && Number.isFinite(hr) && hr > 0 && hr < 240) {
    const shown = Number.isFinite(Number(r.hr)) && Number(r.hr) > 0 ? Number(r.hr) : hr;
    pushSeries(state.bSeries, t, shown);
    pushSeries(state.bQ, t, Number.isFinite(Number(r.sqi)) ? Number(r.sqi) : 0);
    pushSeries(state.bRaw, t, hr);
  }
  if (r.error && Date.now() - state.lastBErrorAt > 5000) {
    state.lastBErrorAt = Date.now();
    log(`B 结果错误：${r.error}`);
  }
}

// --------------------------------------------------------------------------- //
// 叠框
// --------------------------------------------------------------------------- //
function drawOverlay() {
  const v = els.video;
  const w = v.videoWidth || 0;
  const h = v.videoHeight || 0;
  const cssW = els.overlay.clientWidth;
  const cssH = els.overlay.clientHeight;
  if (!w || !h || !cssW || !cssH) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (els.overlay.width !== Math.round(cssW * dpr) || els.overlay.height !== Math.round(cssH * dpr)) {
    els.overlay.width = Math.round(cssW * dpr);
    els.overlay.height = Math.round(cssH * dpr);
  }
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  overlayCtx.clearRect(0, 0, cssW, cssH);
  if (!els.showRoi.checked || !state.box) return;

  // 摄像头画面 object-fit: cover，竖屏设备会左右裁切，需要按同一规则换算。
  const scale = Math.max(cssW / w, cssH / h);
  const offsetX = (cssW - w * scale) / 2;
  const offsetY = (cssH - h * scale) / 2;
  // video 用 transform: scaleX(-1) 做了镜像自拍效果，叠框必须做同样的镜像，
  // 否则框会左右反着画。
  const mapX = (x) => cssW - (offsetX + x * scale);
  const mapY = (y) => offsetY + y * scale;

  const b = state.box;
  const moving = state.motion.level === 'high' || state.motion.level === 'moving';
  const color = moving ? '#f59e0b' : '#22c55e';
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeStyle = color;
  overlayCtx.strokeRect(mapX(b.originX + b.width), mapY(b.originY), b.width * scale, b.height * scale);

  const rects = roiRects(b);
  for (const [name, r] of Object.entries(rects)) {
    overlayCtx.strokeStyle = ROI_COLORS[name] || '#94a3b8';
    overlayCtx.lineWidth = 1.5;
    const rx = mapX(r.x1);
    const ry = mapY(r.y0);
    overlayCtx.strokeRect(rx, ry, (r.x1 - r.x0) * scale, (r.y1 - r.y0) * scale);
    overlayCtx.fillStyle = ROI_COLORS[name] || '#94a3b8';
    overlayCtx.font = '11px ui-monospace, Consolas, monospace';
    overlayCtx.fillText(ROI_ZH[name] || name, rx + 2, ry - 3);
  }
}

// --------------------------------------------------------------------------- //
// UI
// --------------------------------------------------------------------------- //
function showHr(el, value) {
  if (Number.isFinite(Number(value)) && Number(value) > 0) {
    el.innerHTML = `${Math.round(Number(value))}<small>bpm</small>`;
  } else {
    el.innerHTML = '—<small>bpm</small>';
  }
}

function updateUI() {
  if (!state.running && !state.elapsed) {
    els.metaElapsed.textContent = '0.0s';
  } else {
    els.metaElapsed.textContent = `${state.elapsed.toFixed(1)}s`;
  }
  els.badgeFps.textContent = `${state.fps.toFixed(1)} fps`;
  const hitRate = state.faceTries ? state.faceHits / state.faceTries : NaN;
  els.badgeFace.textContent = state.box
    ? `人脸命中 ${pct(hitRate)}`
    : state.detector
      ? '未检测到人脸'
      : '检测器未加载';
  els.badgeMotion.textContent = motionLabel();
  els.metaSize.textContent =
    els.video.videoWidth && els.video.videoHeight
      ? `${els.video.videoWidth} × ${els.video.videoHeight} @ ${state.fps.toFixed(0)}fps`
      : '—';

  const a = state.aResult;
  const aConf = a ? Number(a.heart_rate_confidence ?? a.quality) : NaN;
  if (a && a.ready && Number.isFinite(Number(a.heart_rate))) {
    showHr(els.aHr, a.heart_rate);
    els.aState.textContent =
      Number.isFinite(aConf) && aConf >= 0.2 ? '实时' : `低质量（置信度 ${pct(aConf)}）`;
    els.aConf.textContent = pct(aConf);
  } else {
    showHr(els.aHr, null);
    els.aState.textContent = state.a.span() > 0 ? `预热中（${state.a.span().toFixed(1)}s）` : '等待人脸';
    els.aConf.textContent = '—';
  }
  els.aSpan.textContent = `${Math.min(30, state.a.span()).toFixed(1)}s`;
  els.aFrames.textContent = `${state.frames}`;

  const ap = state.aplusResult;
  if (state.aplusError) {
    showHr(els.aplusHr, null);
    els.aplusState.textContent = 'A+ 不可用';
    els.aplusQuality.textContent = '—';
    els.aplusSqi.textContent = '—';
    els.aplusRoi.textContent = '—';
    els.aplusRaw.textContent = '—';
  } else if (ap && ap.ready) {
    const stable = Number(ap.hr);
    const raw = Number(ap.rawHr);
    const shown = Number.isFinite(stable) && stable > 0 ? stable : Number.isFinite(raw) && raw > 0 ? raw : NaN;
    const quality = Number(ap.quality);
    showHr(els.aplusHr, shown);
    if (!Number.isFinite(shown)) {
      els.aplusState.textContent =
        state.a.span() < 15 ? `预热（质量 ${pct(quality)}）` : `信号不足（质量 ${pct(quality)}）`;
    } else if (!ap.reliable) {
      const held = Number.isFinite(Number(ap.heldSec)) ? `（${Math.round(Number(ap.heldSec))}s 前）` : '';
      if (ap.status === 'holding') els.aplusState.textContent = `低质量·保持${held}`;
      else if (ap.status === 'live') els.aplusState.textContent = '低质量（候选值）';
      else els.aplusState.textContent = '低质量（预热候选）';
    } else if (ap.status === 'holding') {
      const held = Number.isFinite(Number(ap.heldSec)) ? `（${Math.round(Number(ap.heldSec))}s 前）` : '';
      els.aplusState.textContent = `可信 · 保持${held}`;
    } else if (ap.status === 'live') {
      els.aplusState.textContent = '可信 · 实时';
    } else {
      els.aplusState.textContent = '可信 · 预热';
    }
    els.aplusQuality.textContent = pct(quality);
    els.aplusSqi.textContent = Number.isFinite(Number(ap.sqi)) ? Number(ap.sqi).toFixed(3) : '—';
    els.aplusRoi.textContent = ap.activeRoi ? `${ap.activeRoi}${ap.fallback ? '（回退）' : ''}` : '—';
    els.aplusRaw.textContent = Number.isFinite(raw) && raw > 0 ? `${num(raw, 1)} bpm` : '—';
  } else {
    showHr(els.aplusHr, null);
    els.aplusState.textContent = state.a.span() > 0 ? `预热中（${state.a.span().toFixed(1)}s）` : '等待人脸';
    els.aplusQuality.textContent = '—';
    els.aplusSqi.textContent = '—';
    els.aplusRoi.textContent = '—';
    els.aplusRaw.textContent = '—';
  }
  els.aplusSpan.textContent = `${Math.min(30, state.a.span()).toFixed(1)}s`;

  const b = state.bResult;
  if (!els.runB.checked) {
    showHr(els.bHr, null);
    els.bState.textContent = '未启用';
    els.bSqi.textContent = '—';
    els.bDur.textContent = '0s / 30s';
    els.bRaw.textContent = '—';
  } else if (state.deep.error) {
    showHr(els.bHr, null);
    els.bState.textContent = 'B 不可用';
    els.bSqi.textContent = '—';
    els.bDur.textContent = '—';
    els.bRaw.textContent = '—';
  } else if (b && b.ready) {
    const dur = Number(b.duration) || 0;
    showHr(els.bHr, dur >= 28 ? b.hr ?? b.rawHr : null);
    els.bState.textContent = dur < 28 ? `累计信号 ${dur.toFixed(1)}s` : b.status || '—';
    els.bSqi.textContent = Number.isFinite(Number(b.sqi)) ? Number(b.sqi).toFixed(3) : '—';
    els.bDur.textContent = `${dur.toFixed(1)}s / 30s`;
    els.bRaw.textContent = Number.isFinite(Number(b.rawHr)) ? `${num(b.rawHr, 1)} bpm` : '—';
  } else {
    showHr(els.bHr, null);
    els.bState.textContent = state.deep.loading ? '模型加载中…' : '等待模型';
    els.bSqi.textContent = '—';
    els.bDur.textContent = '0s / 30s';
    els.bRaw.textContent = '—';
  }

  updateWarn();
  updateStats();
  drawCharts();
}

function updateWarn() {
  const msgs = [];
  if (state.running && !state.box && state.detector) msgs.push('<b>没有检测到人脸。</b>请让脸完整出现在画面里，正面朝向摄像头，避免逆光。');
  if (state.running && (state.motion.level === 'high' || state.motion.level === 'moving'))
    msgs.push('<b>检测到头部/身体运动。</b>此期间 A 与 A+ 的 ROI 会被运动伪影污染，数字只能当参考；运动停止后请再观察一段。');
  if (state.aplusResult?.ready && !state.aplusResult.reliable)
    msgs.push('<b>A+ 当前不是可信值。</b>A+ 只有 confidence/quality/SQI/margin 同时达标才会置 reliable，低质量时显示的是稳定候选。');
  if (state.aplusResult?.fallback) msgs.push('<b>A+ 正在回退到脸颊 ROI。</b>说明额头区域当前取不到有效候选。');
  if (msgs.length) {
    els.warnbar.innerHTML = msgs.join('<br />');
    els.warnbar.classList.add('show');
  } else {
    els.warnbar.classList.remove('show');
  }
}

function statsCell(s, fmt) {
  return s && s.n ? fmt(s) : '—';
}

function updateStats() {
  const a = summarizeSeries(state.aSeries);
  const ap = summarizeSeries(state.aplusSeries);
  const b = summarizeSeries(state.bSeries);
  const fmt1 = (v) => (Number.isFinite(v) ? v.toFixed(1) : '—');

  els.statsBody.innerHTML = `
    <tr><td>样本数</td><td class="a">${a.n}</td><td class="aplus">${ap.n}</td><td class="b">${b.n}</td></tr>
    <tr><td>均值</td>
      <td class="a">${statsCell(a, (s) => fmt1(s.mean))}</td>
      <td class="aplus">${statsCell(ap, (s) => fmt1(s.mean))}</td>
      <td class="b">${statsCell(b, (s) => fmt1(s.mean))}</td></tr>
    <tr><td>中位数</td>
      <td class="a">${statsCell(a, (s) => fmt1(s.median))}</td>
      <td class="aplus">${statsCell(ap, (s) => fmt1(s.median))}</td>
      <td class="b">${statsCell(b, (s) => fmt1(s.median))}</td></tr>
    <tr><td>最小 / 最大</td>
      <td class="a">${statsCell(a, (s) => `${fmt1(s.min)} / ${fmt1(s.max)}`)}</td>
      <td class="aplus">${statsCell(ap, (s) => `${fmt1(s.min)} / ${fmt1(s.max)}`)}</td>
      <td class="b">${statsCell(b, (s) => `${fmt1(s.min)} / ${fmt1(s.max)}`)}</td></tr>
    <tr><td>标准差</td>
      <td class="a">${statsCell(a, (s) => fmt1(s.std))}</td>
      <td class="aplus">${statsCell(ap, (s) => fmt1(s.std))}</td>
      <td class="b">${statsCell(b, (s) => fmt1(s.std))}</td></tr>`;

  const total = a.n + ap.n + b.n;
  els.statsNote.textContent = total ? `共 ${total} 个检查点` : '尚无数据';

  const pairs = [
    ['A+ − A', compareSeries(state.aplusSeries, state.aSeries)],
    ['B − A', compareSeries(state.bSeries, state.aSeries)],
    ['B − A+', compareSeries(state.bSeries, state.aplusSeries)],
  ];
  const ready = pairs.filter(([, c]) => c.n > 0);
  if (!ready.length) {
    els.diffNote.textContent = '两路都有数据后显示相互偏差（平均绝对差 / 偏差）。';
    updateStartButton();
    return;
  }
  els.diffNote.innerHTML =
    '相互偏差：' +
    ready
      .map(([label, c]) => `${label} 平均绝对差 <b>${c.meanAbsDiff.toFixed(1)}bpm</b>（偏差 ${c.bias >= 0 ? '+' : ''}${c.bias.toFixed(1)}，n=${c.n}）`)
      .join('；');
  updateStartButton();
}

function drawCharts() {
  const end = Math.max(1, state.elapsed);
  const base = {
    yRange: [40, 140],
    yLabel: 'bpm',
    xFormat: (v) => `${Math.round(v)}s`,
    yFormat: (v) => `${Math.round(v)}`,
  };
  const series = [
    { label: 'A · POS+FFT', color: '#22d3ee', width: 2, points: state.aSeries },
    { label: 'A+ · POS+Welch', color: '#34d399', width: 2, points: state.aplusSeries },
    { label: 'B · FacePhys', color: '#a78bfa', width: 2, points: state.bSeries },
  ];

  const rollFrom = Math.max(0, end - ROLL_SEC);
  drawLineChart(els.rollChart, {
    ...base,
    series: series.map((s) => ({ ...s, points: tail(s.points, ROLL_SEC) })),
    xRange: end > ROLL_SEC ? [rollFrom, end] : [0, Math.max(ROLL_SEC, end)],
  });
  els.rollRange.textContent = end > ROLL_SEC ? `最近 ${ROLL_SEC}s` : `0–${ROLL_SEC}s`;

  drawLineChart(els.allChart, {
    ...base,
    series,
    xRange: [0, end],
  });
  els.allRange.textContent = `${Math.round(end)}s`;

  drawLineChart(els.rollQChart, {
    series: [
      { label: 'A 置信度', color: '#22d3ee', width: 2, points: tail(state.aQ, ROLL_SEC) },
      { label: 'A+ 质量', color: '#34d399', width: 2, points: tail(state.aplusQ, ROLL_SEC) },
      { label: 'B SQI', color: '#a78bfa', width: 2, points: tail(state.bQ, ROLL_SEC) },
    ],
    xRange: end > ROLL_SEC ? [rollFrom, end] : [0, Math.max(ROLL_SEC, end)],
    yRange: [0, 1],
    yLabel: 'score',
    xFormat: (v) => `${Math.round(v)}s`,
    yFormat: (v) => `${Math.round(v * 100)}%`,
  });

  drawLineChart(els.rawChart, {
    series: [
      { label: 'A+ 原始 HR', color: '#34d399', width: 1.6, points: tail(state.aplusRaw, ROLL_SEC) },
      { label: 'B 原始 HR', color: '#a78bfa', width: 1.6, points: tail(state.bRaw, ROLL_SEC) },
    ],
    xRange: end > ROLL_SEC ? [rollFrom, end] : [0, Math.max(ROLL_SEC, end)],
    yRange: [40, 160],
    yLabel: 'bpm',
    xFormat: (v) => `${Math.round(v)}s`,
    yFormat: (v) => `${Math.round(v)}`,
  });
  els.rawRange.textContent = end > ROLL_SEC ? `最近 ${ROLL_SEC}s` : `0–${ROLL_SEC}s`;
}

// --------------------------------------------------------------------------- //
// 启停
// --------------------------------------------------------------------------- //
async function start() {
  if (state.running) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('当前浏览器不支持摄像头，请用 Chrome / Edge 并确保页面是 http://localhost 或 https。');
    return;
  }
  els.btnStart.disabled = true;
  setStatus('请求摄像头权限…');
  try {
    await openCamera(els.camSelect.value || '');
  } catch (err) {
    setStatus(`摄像头打开失败：${err.message}`);
    log(`摄像头打开失败：${err.message}`);
    updateStartButton();
    return;
  }

  if (els.runB.checked && !state.deep.ready && !state.deep.error) {
    state.pendingStart = true;
    setStatus('摄像头已就绪，等待 B 模型加载完成后自动开始分析…');
    log('摄像头已就绪，等待 B 模型。');
    await requestDeepStart();
    return;
  }

  await startAnalysis();
}

async function startAnalysis() {
  setStatus('加载人脸检测器…');
  try {
    state.detector = await getFaceDetector();
  } catch (err) {
    setStatus(`人脸检测器加载失败：${err.message}`);
    log(`人脸检测器加载失败：${err.message}`);
    return;
  }

  state.running = true;
  state.startedAt = performance.now() / 1000;
  state.elapsed = 0;
  state.frames = 0;
  state.fpsHistory = [];
  state.faceHits = 0;
  state.faceTries = 0;
  state.box = null;
  state.boxAt = -Infinity;
  state.lastDetectAt = -Infinity;
  state.lastSampleAt = -Infinity;
  state.lastACallAt = -Infinity;
  state.lastAPlusCallAt = -Infinity;
  state.lastBSendAt = -Infinity;
  state.motion = { level: 'ok', score: 0, lastBox: null, lastAt: 0 };
  state.motionSince = null;
  state.aResult = null;
  state.aplusResult = null;
  state.bResult = null;
  state.aplusError = null;
  state.aplusBusy = false;
  state.a.reset();

  await resetAplus();
  if (els.runB.checked && state.deep.ready) await resetDeep();

  setPill(els.camPill, 'live', '正在分析');
  setStatus('实时分析中：请正面朝向摄像头，保持头部尽量稳定。');
  log('开始实时分析。');

  scheduleFrame();
  state.uiTimer = setInterval(updateUI, 300);
  state.resultTimer = setInterval(pollBResult, 1000);
  updateStartButton();
  updateUI();
}

function stop(reason = '已停止。') {
  state.running = false;
  state.pendingStart = false;
  if (state.rafId) cancelAnimationFrame(state.rafId);
  state.rafId = 0;
  if (state.uiTimer) clearInterval(state.uiTimer);
  if (state.resultTimer) clearInterval(state.resultTimer);
  state.uiTimer = null;
  state.resultTimer = null;
  closeCamera();
  setStatus(reason);
  updateStartButton();
  updateUI();
}

function clearHistory() {
  state.aSeries = [];
  state.aplusSeries = [];
  state.bSeries = [];
  state.aQ = [];
  state.aplusQ = [];
  state.bQ = [];
  state.aRaw = [];
  state.aplusRaw = [];
  state.bRaw = [];
  state.a.reset();
  updateUI();
  log('已清空历史曲线。');
}

function exportJson() {
  const payload = {
    tool: 'camera-aplus-b-live',
    generatedAt: new Date().toISOString(),
    device: state.deviceLabel,
    durationSec: Number(state.elapsed.toFixed(2)),
    frames: state.frames,
    fpsCap: Number(els.fpsCap.value) || 15,
    series: {
      a: state.aSeries,
      aplus: state.aplusSeries,
      b: state.bSeries,
      aQuality: state.aQ,
      aplusQuality: state.aplusQ,
      bSqi: state.bQ,
      aplusRaw: state.aplusRaw,
      bRaw: state.bRaw,
    },
    stats: {
      a: summarizeSeries(state.aSeries),
      aplus: summarizeSeries(state.aplusSeries),
      b: summarizeSeries(state.bSeries),
      diff: {
        aplusVsA: compareSeries(state.aplusSeries, state.aSeries),
        bVsA: compareSeries(state.bSeries, state.aSeries),
        bVsAplus: compareSeries(state.bSeries, state.aplusSeries),
      },
    },
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `camera-${Date.now()}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// --------------------------------------------------------------------------- //
// 事件
// --------------------------------------------------------------------------- //
els.btnStart.addEventListener('click', start);
els.btnStop.addEventListener('click', () => stop('已停止。'));
els.btnReset.addEventListener('click', clearHistory);
els.btnExport.addEventListener('click', exportJson);

els.runB.addEventListener('change', async () => {
  updateStartButton();
  if (els.runB.checked && !state.deep.ready && !state.deep.error) await requestDeepStart();
  if (state.running && els.runB.checked && state.deep.ready) await resetDeep();
});

els.camSelect.addEventListener('change', async () => {
  if (state.running) return;
  if (!state.stream) return;
  setStatus('切换摄像头…');
  try {
    await openCamera(els.camSelect.value || '');
    setStatus('已切换摄像头，点击「开启摄像头」开始分析。');
  } catch (err) {
    setStatus(`切换失败：${err.message}`);
  }
});

window.addEventListener('beforeunload', () => {
  closeCamera();
});

navigator.mediaDevices?.addEventListener?.('devicechange', () => {
  listCameras();
});

updateStartButton();
updateUI();
listCameras();
pollDeep();
requestDeepStart();
