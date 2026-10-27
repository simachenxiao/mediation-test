// 实时 A / A+ / B 心率对比页。
//
// A ：MediaPipe 人脸框 → ROI 平均 RGB → 30s 滑动窗口 → /physio/estimate（POS+FFT）
// A+：复用 A 的采样 payload → /aplus/estimate（POS + Welch + SQI）
// B ：浏览器把播放帧发给 /deep/frame → deep_service.py（FacePhys.rlap）→ /deep/result
//
// 视频按 1× 播放，三路结果边播边出；不做整段离线分析。

import { getFaceDetector, detectFaces, largestFace, clampBox } from '/lib/face-detect.js';
import { LivePhysioA } from '/lib/physio-live.js';
import { drawLineChart } from '/lib/charts.js';
import { formatBytes, formatTime } from '/lib/video-frames.js';

const $ = (id) => document.getElementById(id);

const els = {
  drop: $('drop'),
  file: $('file'),
  video: $('video'),
  meta: $('meta'),
  playTime: $('playTime'),
  playHint: $('playHint'),
  deepPill: $('deepPill'),
  btnStart: $('btnStart'),
  btnStop: $('btnStop'),
  btnRestart: $('btnRestart'),
  status: $('status'),
  log: $('log'),
  aHr: $('aHr'),
  aState: $('aState'),
  aConf: $('aConf'),
  aSpan: $('aSpan'),
  aplusHr: $('aplusHr'),
  aplusState: $('aplusState'),
  aplusQuality: $('aplusQuality'),
  aplusSqi: $('aplusSqi'),
  aplusMethod: $('aplusMethod'),
  aplusRaw: $('aplusRaw'),
  aplusSpan: $('aplusSpan'),
  bHr: $('bHr'),
  bState: $('bState'),
  bSqi: $('bSqi'),
  bDur: $('bDur'),
  bRaw: $('bRaw'),
  diffAPlus: $('diffAPlus'),
  diffB: $('diffB'),
  diffBA: $('diffBA'),
  diffNote: $('diffNote'),
  hrChart: $('hrChart'),
  qChart: $('qChart'),
};

const state = {
  file: null,
  url: null,
  duration: 0,
  analyzing: false,
  detector: null,
  box: null,
  boxAt: -Infinity,
  detecting: false,
  lastDetectAt: -Infinity,
  lastACallAt: -Infinity,
  lastAPlusCallAt: -Infinity,
  lastBSendAt: -Infinity,
  lastBErrorAt: 0,
  sendingB: false,
  deep: { ready: false, loading: true, error: null, sidecar: false, loadSec: null },
  deepTimer: null,
  resultTimer: null,
  uiTimer: null,
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
  internalSeek: false,
  startWall: 0,
};

const frameCanvas = document.createElement('canvas');
frameCanvas.width = 1280;
frameCanvas.height = 720;
const frameCtx = frameCanvas.getContext('2d');
const MAX_POINTS = 420;

function log(msg) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  const old = els.log.textContent;
  const lines = (old === '就绪。' ? [] : old.split('\n')).concat(line);
  els.log.textContent = lines.slice(-80).join('\n');
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

function pushSeries(list, x, y) {
  if (!Number.isFinite(x) || !Number.isFinite(y) || y <= 0) return;
  list.push({ x, y });
  if (list.length > MAX_POINTS) list.shift();
}

function setDeepState(next) {
  state.deep = { ...state.deep, ...next };
  const d = state.deep;
  els.deepPill.className = 'pill ' + (d.error ? 'error' : d.ready ? 'ready' : 'loading');
  if (d.error) els.deepPill.textContent = 'B 不可用';
  else if (d.ready) els.deepPill.textContent = 'B 模型就绪';
  else if (d.loading)
    els.deepPill.textContent = `B 模型加载中…${d.loadSec ? ` ${Math.round(d.loadSec)}s` : ''}`;
  else els.deepPill.textContent = 'B 未启动';
  updateStartButton();
}

function updateStartButton() {
  const bWaiting = !state.deep.ready && !state.deep.error;
  els.btnStart.disabled = !state.file || state.analyzing || bWaiting;
  els.btnStop.disabled = !state.analyzing;
  els.btnRestart.disabled = !state.file;
}

function updateMeta() {
  if (!state.file) {
    els.meta.innerHTML = '<dt>文件</dt><dd>—</dd><dt>时长</dt><dd>—</dd><dt>分辨率</dt><dd>—</dd>';
    return;
  }
  els.meta.innerHTML = `
    <dt>文件</dt><dd title="${state.file.name}">${state.file.name}</dd>
    <dt>大小</dt><dd>${formatBytes(state.file.size)}</dd>
    <dt>时长</dt><dd>${formatTime(state.duration)}</dd>
    <dt>分辨率</dt><dd>${els.video.videoWidth} × ${els.video.videoHeight}</dd>`;
}

async function loadFile(file) {
  if (!file) return;
  if (file.type && !/^video\//.test(file.type) && !/\.(mp4|webm|mov|mkv|avi|m4v)$/i.test(file.name)) {
    setStatus('不是视频文件。');
    return;
  }
  stopAnalysis();
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = file;
  state.url = URL.createObjectURL(file);
  state.duration = 0;
  state.aResult = null;
  state.aplusResult = null;
  state.bResult = null;
  state.aplusError = null;
  state.aplusBusy = false;
  state.aplusErrorAt = 0;
  state.aSeries = [];
  state.aplusSeries = [];
  state.bSeries = [];
  state.aQ = [];
  state.aplusQ = [];
  state.bQ = [];
  state.a.reset();

  els.video.src = state.url;
  els.video.controls = true;
  els.video.load();
  setStatus('读取视频元数据…');
  try {
    await new Promise((resolve, reject) => {
      const ok = () => {
        cleanup();
        resolve();
      };
      const bad = () => {
        cleanup();
        reject(new Error('视频解码失败'));
      };
      const cleanup = () => {
        els.video.removeEventListener('loadedmetadata', ok);
        els.video.removeEventListener('error', bad);
      };
      els.video.addEventListener('loadedmetadata', ok);
      els.video.addEventListener('error', bad);
    });
    state.duration = Number.isFinite(els.video.duration) ? els.video.duration : 0;
    updateMeta();
    els.playHint.textContent = '可选择开始实时分析';
    setStatus('就绪：A 约 8s 出结果，A+ 约 12s 出结果；A/A+ 之后每秒更新，B 需要先加载模型并累计约 30s。');
    log(`已选择 ${file.name}，${formatTime(state.duration)}`);
  } catch (err) {
    setStatus(`读取失败：${err.message}`);
    log(`读取失败：${err.message}`);
  }
  updateStartButton();
  updateUI();
}

async function pollDeep() {
  try {
    const res = await fetch('/deep/health', { cache: 'no-store' });
    const d = await res.json();
    const wasReady = state.deep.ready;
    setDeepState({
      ready: !!d.ready,
      loading: !!d.loading,
      error: d.error || null,
      sidecar: d.sidecar !== false,
      loadSec: Number.isFinite(Number(d.loadSec)) ? Number(d.loadSec) : null,
    });
    if (d.ready && !wasReady) {
      log(`B 模型就绪（${d.model || 'FacePhys.rlap'}）。`);
      setStatus('A / A+ / B 就绪，可以开始实时分析。');
    }
    if (d.error && !state.deep.error) {
      log(`B 模型错误：${d.error}`);
      setStatus(`B 不可用：${d.error}`);
    }
  } catch (err) {
    setDeepState({ ready: false, loading: false, error: err.message, sidecar: false });
  }
  const interval = state.deep.ready || state.deep.error ? 5000 : 1500;
  state.deepTimer = setTimeout(pollDeep, interval);
}

async function startDeep() {
  try {
    await fetch('/deep/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });
    log('正在加载 B 模型（FacePhys.rlap，首次约 30–40s）…');
  } catch (err) {
    setDeepState({ ready: false, loading: false, error: err.message });
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
    log('B 模型已重置。');
  } catch (err) {
    log(`B 重置失败：${err.message}`);
  }
}

async function warmAplus() {
  try {
    const res = await fetch('/aplus/health', { cache: 'no-store' });
    const d = await res.json();
    if (d.ready) log('A+ 引擎就绪（POS + Welch + SQI）。');
    else if (d.error) log(`A+ 引擎不可用：${d.error}`);
  } catch (err) {
    state.aplusError = err.message;
    log(`A+ 引擎预热失败：${err.message}`);
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
    state.aplusError = null;
    log('A+ 引擎已重置。');
  } catch (err) {
    state.aplusError = err.message;
    log(`A+ 重置失败：${err.message}`);
  }
}

async function requestAplus(payload) {
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
    updateAplusResult(r);
    return r;
  } catch (err) {
    state.aplusError = err.message;
    const now = Date.now();
    if (now - state.aplusErrorAt > 5000) {
      state.aplusErrorAt = now;
      log(`A+ 估计失败：${err.message}`);
    }
    return null;
  } finally {
    state.aplusBusy = false;
  }
}

async function ensureDetector() {
  if (state.detector) return state.detector;
  setStatus('加载人脸检测器…');
  state.detector = await getFaceDetector();
  return state.detector;
}

function scheduleFrame() {
  if (!state.analyzing) return;
  const v = els.video;
  if (typeof v.requestVideoFrameCallback === 'function') {
    v.requestVideoFrameCallback(() => {
      onPlaybackFrame();
      scheduleFrame();
    });
  } else {
    setTimeout(() => {
      onPlaybackFrame();
      scheduleFrame();
    }, 33);
  }
}

function onPlaybackFrame() {
  if (!state.analyzing) return;
  const v = els.video;
  const t = v.currentTime;

  if (state.detector && !state.detecting && t - state.lastDetectAt >= 0.3) {
    state.lastDetectAt = t;
    state.detecting = true;
    detectFaces(state.detector, v)
      .then((dets) => {
        const hit = largestFace(dets);
        if (hit) {
          state.box = clampBox(hit.box, v.videoWidth, v.videoHeight);
          state.boxAt = t;
        }
      })
      .catch((err) => log(`人脸检测失败：${err.message}`))
      .finally(() => {
        state.detecting = false;
      });
  }

  if (state.box && t - state.boxAt <= 1.5) {
    state.a.addFrame(v, state.box, t);
  }

  if (t - state.lastACallAt >= 1 && state.a.span() >= 8) {
    state.lastACallAt = t;
    state.a
      .maybeEstimate()
      .then((r) => {
        if (r) updateAResult(r);
      })
      .catch((err) => log(`A 估计失败：${err.message}`));
  }

  if (t - state.lastAPlusCallAt >= 1 && state.a.span() >= 12) {
    state.lastAPlusCallAt = t;
    const payload = state.a.buildPayload(state.a.latestTime());
    requestAplus(payload);
  }

  if (
    state.deep.ready &&
    !state.sendingB &&
    t - state.lastBSendAt >= 0.066
  ) {
    state.lastBSendAt = t;
    sendBFrame(t);
  }
}

async function sendBFrame(t) {
  if (!state.deep.ready || state.sendingB) return;
  state.sendingB = true;
  try {
    frameCtx.drawImage(els.video, 0, 0, frameCanvas.width, frameCanvas.height);
    const image = frameCanvas.toDataURL('image/jpeg', 0.7);
    const res = await fetch('/deep/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image, timestamp: t }),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      const now = Date.now();
      if (now - state.lastBErrorAt > 5000) {
        state.lastBErrorAt = now;
        log(`B 帧发送失败 HTTP ${res.status}${text ? ': ' + text.slice(0, 120) : ''}`);
      }
    }
  } catch (err) {
    const now = Date.now();
    if (now - state.lastBErrorAt > 5000) {
      state.lastBErrorAt = now;
      log(`B 帧发送失败：${err.message}`);
    }
  } finally {
    state.sendingB = false;
  }
}

async function fetchBResult() {
  const res = await fetch('/deep/result', { cache: 'no-store' });
  const r = await res.json();
  updateBResult(r);
  return r;
}

async function pollBResult() {
  if (!state.analyzing || !state.deep.ready) return;
  try {
    await fetchBResult();
  } catch (err) {
    log(`B 结果获取失败：${err.message}`);
  }
}

function updateAResult(r) {
  state.aResult = r;
  if (!r || !r.ready) return;
  const hr = Number(r.heart_rate);
  const conf = Number(r.heart_rate_confidence ?? r.quality);
  if (Number.isFinite(hr) && hr > 0 && hr < 240) {
    pushSeries(state.aSeries, els.video.currentTime, hr);
    pushSeries(state.aQ, els.video.currentTime, Number.isFinite(conf) ? conf : 0);
  }
}

function updateAplusResult(r) {
  state.aplusResult = r;
  if (!r || !r.ready) return;
  const hr = Number(r.hr ?? r.rawHr);
  const quality = Number(r.quality);
  if (Number.isFinite(hr) && hr > 0 && hr < 240) {
    pushSeries(state.aplusSeries, els.video.currentTime, hr);
    pushSeries(state.aplusQ, els.video.currentTime, Number.isFinite(quality) ? quality : 0);
  }
}

function updateBResult(r) {
  if (!r) return;
  state.bResult = r;
  const hr = Number(r.hr);
  const dur = Number(r.duration) || 0;
  if (r.ready && dur >= 28 && Number.isFinite(hr) && hr > 0 && hr < 240) {
    pushSeries(state.bSeries, els.video.currentTime, hr);
    pushSeries(state.bQ, els.video.currentTime, Number.isFinite(Number(r.sqi)) ? Number(r.sqi) : 0);
  }
  if (r.error) {
    const now = Date.now();
    if (now - state.lastBErrorAt > 5000) {
      state.lastBErrorAt = now;
      log(`B 结果错误：${r.error}`);
    }
  }
}

function updateUI() {
  const v = els.video;
  const t = v.currentTime || 0;
  els.playTime.textContent = `${formatTime(t)} / ${formatTime(state.duration)}`;

  const a = state.aResult;
  const aConf = a ? Number(a.heart_rate_confidence ?? a.quality) : NaN;
  if (a && a.ready && Number.isFinite(Number(a.heart_rate))) {
    els.aHr.innerHTML = `${Math.round(Number(a.heart_rate))}<small>bpm</small>`;
    els.aState.textContent =
      Number.isFinite(aConf) && aConf >= 0.2
        ? '实时'
        : `低质量（置信度 ${pct(aConf)}）`;
    els.aConf.textContent = pct(aConf);
  } else {
    els.aHr.innerHTML = '—<small>bpm</small>';
    els.aState.textContent = state.a.span() > 0 ? `预热中（${state.a.span().toFixed(1)}s）` : '等待人脸';
    els.aConf.textContent = '—';
  }
  els.aSpan.textContent = `${Math.min(30, state.a.span()).toFixed(1)}s`;

  const ap = state.aplusResult;
  if (state.aplusError) {
    els.aplusHr.innerHTML = '—<small>bpm</small>';
    els.aplusState.textContent = 'A+ 不可用';
    els.aplusQuality.textContent = '—';
    els.aplusSqi.textContent = '—';
    els.aplusMethod.textContent = '—';
    els.aplusRaw.textContent = '—';
  } else if (ap && ap.ready) {
    const stable = Number(ap.hr);
    const raw = Number(ap.rawHr);
    const shown = Number.isFinite(stable) && stable > 0
      ? stable
      : Number.isFinite(raw) && raw > 0
        ? raw
        : NaN;
    const quality = Number(ap.quality);
    const sqi = Number(ap.sqi);

    if (Number.isFinite(shown)) {
      els.aplusHr.innerHTML = `${Math.round(shown)}<small>bpm</small>`;
      if (!ap.reliable) {
        els.aplusState.textContent =
          ap.status === 'holding'
            ? `低质量·保持（${Number.isFinite(Number(ap.heldSec)) ? `${Math.round(Number(ap.heldSec))}s 前` : '质量低'}）`
            : ap.status === 'live'
              ? '低质量（候选值）'
              : '低质量（预热候选）';
      } else if (ap.status === 'holding') {
        els.aplusState.textContent =
          `保持（${Number.isFinite(Number(ap.heldSec)) ? `${Math.round(Number(ap.heldSec))}s 前` : '质量低'}）`;
      } else if (ap.status === 'live') {
        els.aplusState.textContent = '实时';
      } else {
        els.aplusState.textContent = '预热';
      }
    } else {
      els.aplusHr.innerHTML = '—<small>bpm</small>';
      els.aplusState.textContent =
        state.a.span() < 15
          ? `预热（质量 ${pct(quality)}）`
          : `信号不足（质量 ${pct(quality)}）`;
    }
    els.aplusQuality.textContent = pct(quality);
    els.aplusSqi.textContent = Number.isFinite(sqi) ? sqi.toFixed(3) : '—';
    els.aplusMethod.textContent = ap.method || '—';
    els.aplusMethod.title = ap.method || '';
    els.aplusRaw.textContent = Number.isFinite(raw) && raw > 0 ? `${num(raw, 1)} bpm` : '—';
  } else {
    els.aplusHr.innerHTML = '—<small>bpm</small>';
    els.aplusState.textContent = state.a.span() > 0 ? `预热中（${state.a.span().toFixed(1)}s）` : '等待人脸';
    els.aplusQuality.textContent = '—';
    els.aplusSqi.textContent = '—';
    els.aplusMethod.textContent = '—';
    els.aplusRaw.textContent = '—';
  }
  els.aplusSpan.textContent = `${Math.min(30, state.a.span()).toFixed(1)}s`;

  const b = state.bResult;
  if (state.deep.error) {
    els.bHr.innerHTML = '—<small>bpm</small>';
    els.bState.textContent = 'B 不可用';
    els.bSqi.textContent = '—';
    els.bDur.textContent = '—';
    els.bRaw.textContent = '—';
  } else if (state.deep.loading) {
    els.bHr.innerHTML = '—<small>bpm</small>';
    els.bState.textContent = '模型加载中…';
    els.bSqi.textContent = '—';
    els.bDur.textContent = '—';
    els.bRaw.textContent = '—';
  } else if (b && b.ready) {
    const hr = Number(b.hr);
    const dur = Number(b.duration) || 0;
    const show = dur >= 28 && Number.isFinite(hr) && hr > 0;
    els.bHr.innerHTML = `${show ? Math.round(hr) : '—'}<small>bpm</small>`;
    if (dur < 28) {
      els.bState.textContent = `预热（${num(dur, 1)}s / 30s）`;
    } else if (b.status === 'live') {
      els.bState.textContent = '实时';
    } else if (b.status === 'holding') {
      els.bState.textContent = '保持（SQI 低）';
    } else {
      els.bState.textContent = '预热';
    }
    els.bSqi.textContent = Number.isFinite(Number(b.sqi)) ? Number(b.sqi).toFixed(3) : '—';
    els.bDur.textContent = `${num(dur, 1)}s / 30s`;
    els.bRaw.textContent =
      dur >= 28 && Number.isFinite(Number(b.rawHr)) && Number(b.rawHr) > 0
        ? `${num(b.rawHr, 1)} bpm`
        : '—';
  } else {
    els.bHr.innerHTML = '—<small>bpm</small>';
    els.bState.textContent = '等待信号';
    els.bSqi.textContent = '—';
    els.bDur.textContent = '0s / 30s';
    els.bRaw.textContent = '—';
  }

  const aHr = a && a.ready ? Number(a.heart_rate) : NaN;
  const apHr = ap && ap.ready ? Number(ap.hr ?? ap.rawHr) : NaN;
  const bHr = b && b.ready && Number(b.duration) >= 28 ? Number(b.hr) : NaN;

  const fmtDiff = (x) =>
    Number.isFinite(x) ? `${x >= 0 ? '+' : ''}${x.toFixed(1)}<small>bpm</small>` : '—<small>bpm</small>';
  els.diffAPlus.innerHTML = fmtDiff(Number.isFinite(aHr) && Number.isFinite(apHr) ? apHr - aHr : NaN);
  els.diffB.innerHTML = fmtDiff(Number.isFinite(aHr) && Number.isFinite(bHr) ? bHr - aHr : NaN);
  els.diffBA.innerHTML = fmtDiff(Number.isFinite(apHr) && Number.isFinite(bHr) ? bHr - apHr : NaN);

  const aGood = Number.isFinite(aConf) && aConf >= 0.2;
  const apGood = !!(ap && ap.ready && ap.reliable);
  const bGood = !!(b && b.status === 'live' && Number(b.duration) >= 28);
  if (Number.isFinite(aHr) && Number.isFinite(apHr) && Number.isFinite(bHr)) {
    const vals = [aHr, apHr, bHr];
    const spread = Math.max(...vals) - Math.min(...vals);
    const bits = [];
    if (!aGood) bits.push(`A 置信度 ${pct(aConf)}`);
    if (!apGood) bits.push(`A+ 质量 ${pct(ap?.quality)}`);
    if (!bGood) bits.push(`B SQI ${Number.isFinite(Number(b?.sqi)) ? Number(b.sqi).toFixed(3) : '—'}`);
    if (bits.length) {
      els.diffNote.textContent = `质量不足（${bits.join('，')}），差异仅供参考。`;
    } else if (spread <= 8) {
      els.diffNote.textContent = '三路基本一致（差值 ≤ 8bpm）。';
    } else {
      els.diffNote.textContent = '三路偏差较大：优先看 A+ 质量、A 置信度和 B SQI，低质量数字不可信。';
    }
  } else if (state.aplusError) {
    els.diffNote.textContent = `A+ 不可用：${state.aplusError}`;
  } else {
    els.diffNote.textContent = '等待三路结果（A 约 8s，A+ 约 12s，B 约 30s）。';
  }

  drawCharts();
}

function drawCharts() {
  drawLineChart(els.hrChart, {
    series: [
      { label: 'A POS+FFT', color: '#22d3ee', width: 2, points: state.aSeries },
      { label: 'A+ POS+Welch+SQI', color: '#34d399', width: 2, points: state.aplusSeries },
      { label: 'B FacePhys', color: '#a78bfa', width: 2, points: state.bSeries },
    ],
    xRange: [0, Math.max(1, state.duration)],
    yRange: [30, 140],
    xFormat: (v) => `${Math.round(v)}s`,
    yFormat: (v) => `${Math.round(v)}`,
  });

  drawLineChart(els.qChart, {
    series: [
      { label: 'A 置信度', color: '#22d3ee', width: 2, points: state.aQ },
      { label: 'A+ 质量', color: '#34d399', width: 2, points: state.aplusQ },
      { label: 'B SQI', color: '#a78bfa', width: 2, points: state.bQ },
    ],
    xRange: [0, Math.max(1, state.duration)],
    yRange: [0, 1],
    xFormat: (v) => `${Math.round(v)}s`,
    yFormat: (v) => `${Math.round(v * 100)}%`,
  });
}

async function startAnalysis() {
  if (state.analyzing || !state.file) return;
  if (!state.deep.ready && !state.deep.error) {
    setStatus('B 模型还在加载，请稍候…');
    return;
  }

  state.analyzing = true;
  state.aResult = null;
  state.aplusResult = null;
  state.bResult = null;
  state.aplusError = null;
  state.aplusBusy = false;
  state.aplusErrorAt = 0;
  state.aSeries = [];
  state.aplusSeries = [];
  state.bSeries = [];
  state.aQ = [];
  state.aplusQ = [];
  state.bQ = [];
  state.box = null;
  state.boxAt = -Infinity;
  state.lastDetectAt = -Infinity;
  state.lastACallAt = -Infinity;
  state.lastAPlusCallAt = -Infinity;
  state.lastBSendAt = -Infinity;
  state.a.reset();
  state.startWall = performance.now();
  els.video.controls = false;
  updateStartButton();
  setStatus('实时分析中：视频 1× 播放，A / A+ / B 边播边算…');
  log('开始实时分析。');

  if (state.deep.ready) {
    await resetDeep();
  } else {
    log('B 不可用，本次运行 A / A+。');
  }
  await resetAplus();

  try {
    state.internalSeek = true;
    els.video.currentTime = 0;
    await new Promise((resolve) => {
      const done = () => {
        els.video.removeEventListener('seeked', done);
        resolve();
      };
      els.video.addEventListener('seeked', done);
      setTimeout(done, 800);
    });
    state.internalSeek = false;
    await ensureDetector();
    setStatus('实时分析中：视频 1× 播放，A / A+ / B 边播边算…');
    await els.video.play();
    els.playHint.textContent = '实时分析中（已禁用拖动）';
    scheduleFrame();
    state.resultTimer = setInterval(pollBResult, 1000);
    state.uiTimer = setInterval(updateUI, 250);
    updateUI();
  } catch (err) {
    log(`启动失败：${err.message}`);
    stopAnalysis('启动失败：' + err.message);
  }
}

function stopAnalysis(reason = '已停止。') {
  state.analyzing = false;
  if (state.resultTimer) clearInterval(state.resultTimer);
  if (state.uiTimer) clearInterval(state.uiTimer);
  state.resultTimer = null;
  state.uiTimer = null;
  try {
    els.video.pause();
  } catch {
    /* ignore */
  }
  els.video.controls = true;
  els.playHint.textContent = '已停止，可重新开始';
  updateStartButton();
  setStatus(reason);
  updateUI();
}

async function onEnded() {
  if (!state.analyzing) return;
  state.analyzing = false;
  if (state.resultTimer) clearInterval(state.resultTimer);
  if (state.uiTimer) clearInterval(state.uiTimer);
  state.resultTimer = null;
  state.uiTimer = null;
  els.video.controls = true;
  els.playHint.textContent = '播放结束';
  setStatus('播放结束，正在取最后结果…');
  log('视频播放结束。');
  try {
    const last = await state.a.maybeEstimate(true);
    if (last) updateAResult(last);
  } catch (err) {
    log(`A 最后估计失败：${err.message}`);
  }
  try {
    const payload = state.a.buildPayload(state.a.latestTime());
    await requestAplus(payload);
  } catch (err) {
    log(`A+ 最后估计失败：${err.message}`);
  }
  if (state.deep.ready) {
    // B 的稳定器有 5bpm/s 限速：结束时多取几次，让显示值收敛到最后的原始 HR。
    setStatus('播放结束，B 正在收敛最后结果…');
    for (let i = 0; i < 6; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      if (state.analyzing) return;
      try {
        await fetchBResult();
      } catch (err) {
        log(`B 最后结果获取失败：${err.message}`);
        break;
      }
      updateUI();
    }
  }
  if (state.analyzing) return;
  updateStartButton();
  setStatus('播放结束。可查看 A / A+ / B 最终结果，或重新开始。');
  updateUI();
}

function bind() {
  els.drop.addEventListener('click', () => els.file.click());
  els.file.addEventListener('change', () => {
    const f = els.file.files?.[0];
    els.file.value = '';
    if (f) loadFile(f);
  });
  for (const ev of ['dragenter', 'dragover']) {
    els.drop.addEventListener(ev, (e) => {
      e.preventDefault();
      els.drop.classList.add('over');
    });
  }
  for (const ev of ['dragleave', 'drop']) {
    els.drop.addEventListener(ev, (e) => {
      e.preventDefault();
      els.drop.classList.remove('over');
    });
  }
  els.drop.addEventListener('drop', (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) loadFile(f);
  });

  els.video.addEventListener('ended', onEnded);
  els.video.addEventListener('seeking', () => {
    if (state.analyzing && !state.internalSeek) {
      stopAnalysis('检测到拖动进度，已停止实时分析。');
    }
  });

  els.btnStart.addEventListener('click', startAnalysis);
  els.btnStop.addEventListener('click', () => stopAnalysis());
  els.btnRestart.addEventListener('click', async () => {
    stopAnalysis('重新开始…');
    await startAnalysis();
  });

  updateStartButton();
  updateUI();
}

bind();
startDeep();
warmAplus();
