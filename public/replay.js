// 确定性重放：固定帧号 + 固定时间戳 + 固定人脸框 + 固定窗口。
//
// 与 /live 的区别：
//   * 不按 requestVideoFrameCallback 采样，而是按 15fps 逐点 seek；
//   * 人脸框只由固定帧序列产生，并缓存；
//   * A/A+ 在固定检查点构造同一份 payload；
//   * A+/B 的稳定器使用视频时间（replayTime），不用 wall clock；
//   * A/A+ 自动复跑一次并逐点比较，B 可点“再跑一次”验证。

import {
  probeVideoFile,
  ensureVideoReady,
  seekTo,
  planFrameTimes,
  createFrameGrabber,
  formatBytes,
  formatTime,
} from '/lib/video-frames.js';
import { getFaceDetector, detectFaces, largestFace, clampBox } from '/lib/face-detect.js';
import { sampleRoiMeans } from '/lib/roi.js';
import { drawLineChart } from '/lib/charts.js';

const $ = (id) => document.getElementById(id);
const FPS = 15;
const DETECT_EVERY = 5;          // 固定帧序列里每 5 帧检测一次
const WINDOW_SEC = 30;
const STEP_SEC = 1;
const A_START = 8;
const APLUS_START = 12;
const B_START = 30;
const B_W = 1280;
const B_H = 720;

const ROI_NAMES = ['forehead', 'left_cheek', 'right_cheek'];

const state = {
  file: null,
  url: null,
  meta: null,
  preview: null,
  video: null,
  frames: [],
  extractedEnd: 0,
  running: false,
  stopping: false,
  last: null,
  previous: null,
  detector: null,
};

// --------------------------------------------------------------------------- //
// 基础工具
// --------------------------------------------------------------------------- //
function log(msg) {
  const el = $('log');
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  el.textContent = el.textContent === '就绪。' ? line : `${el.textContent}\n${line}`;
  el.scrollTop = el.scrollHeight;
  console.info('[replay]', msg);
}

function setStatus(msg) { $('status').textContent = msg; }

function setProgress(pct, text = '', busy = false) {
  $('progressBar').style.width = `${Math.max(0, Math.min(1, pct)) * 100}%`;
  $('progressBar').classList.toggle('busy', !!busy);
  $('progressText').textContent = text;
}

function checkStop() {
  if (state.stopping) {
    const err = new Error('已停止');
    err.name = 'AbortError';
    throw err;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, { cache: 'no-store', ...options });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
  if (!res.ok) {
    const msg = data?.error || data?.raw || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function postJson(url, body) {
  return fetchJson(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function round(v, digits = 1) {
  return Number.isFinite(Number(v)) ? Math.round(Number(v) * 10 ** digits) / 10 ** digits : null;
}

function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

function sameNum(a, b, tol = 0.05) {
  const x = a == null ? null : Number(a);
  const y = b == null ? null : Number(b);
  if (x == null && y == null) return true;
  if (x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y)) return false;
  return Math.abs(x - y) <= tol;
}

// --------------------------------------------------------------------------- //
// 文件 / 预览
// --------------------------------------------------------------------------- //
async function loadFile(file) {
  if (!file || state.running) return;
  setStatus('读取视频…');
  setProgress(0, '读取视频…', true);
  try {
    cleanupVideo();
    const probe = await probeVideoFile(file);
    state.file = file;
    state.url = probe.url;
    state.meta = probe.meta;
    state.video = probe.video;
    state.preview = $('video');

    $('video').src = probe.url;
    $('video').muted = true;
    await ensureVideoReady($('video')).catch(() => {});
    $('video').currentTime = Math.min(0.04, Math.max(0.001, probe.meta.duration / 1000));
    await ensureVideoReady(state.video).catch(() => {});

    const m = state.meta;
    const dds = $('meta').querySelectorAll('dd');
    dds[0].textContent = m.name;
    dds[1].textContent = `${formatTime(m.duration)} (${m.duration.toFixed(2)}s)`;
    dds[2].textContent = `${m.width} × ${m.height}`;
    dds[2].title = m.type || '';
    $('playTime').textContent = `00:00 / ${formatTime(m.duration)}`;
    $('playHint').textContent = `${formatBytes(m.size)} · 固定 15fps 重放`;
    $('file').value = '';

    state.frames = [];
    state.extractedEnd = 0;
    state.last = null;
    state.previous = null;
    $('btnStart').disabled = false;
    $('btnRepeat').disabled = true;
    $('btnExport').disabled = true;
    resetResultUI();
    setProgress(0, '视频已就绪，可开始固定重放', false);
    setStatus('固定采样：15fps；窗口 30s；每秒一个检查点。');
    log(`视频就绪：${m.width}×${m.height}，${formatTime(m.duration)}，${formatBytes(m.size)}`);
  } catch (err) {
    setProgress(0, `读取失败：${err.message}`, false);
    setStatus(`读取失败：${err.message}`);
    log(`读取失败：${err.message}`);
  }
}

function cleanupVideo() {
  if (state.preview) {
    try { state.preview.pause(); } catch {}
    state.preview.removeAttribute('src');
    state.preview.load();
  }
  if (state.video) {
    try { state.video.pause(); } catch {}
    state.video.removeAttribute('src');
    state.video.load();
  }
  if (state.url) {
    URL.revokeObjectURL(state.url);
    state.url = null;
  }
  state.video = null;
  state.preview = null;
  state.frames = [];
  state.extractedEnd = 0;
}

// --------------------------------------------------------------------------- //
// B 模型健康度
// --------------------------------------------------------------------------- //
function setDeepPill(mode, text) {
  const el = $('deepPill');
  el.className = `pill ${mode}`;
  el.textContent = text;
}

async function refreshDeepPill() {
  try {
    const h = await fetchJson('/deep/health');
    if (h.ready) setDeepPill('ready', 'B 已就绪');
    else if (h.loading) setDeepPill('loading', 'B 加载中…');
    else if (h.error) setDeepPill('error', `B 错误：${String(h.error).slice(0, 40)}`);
    else setDeepPill('', 'B 未启动');
    return h;
  } catch (err) {
    setDeepPill('error', 'B 不可用');
    return { ready: false, error: err.message };
  }
}

async function ensureB() {
  let h = await refreshDeepPill();
  if (h.ready) return;
  if (!h.loading) {
    log('启动 B 侧车…');
    await postJson('/deep/start', {}).catch((err) => log(`B 启动失败：${err.message}`));
  }
  const started = Date.now();
  while (Date.now() - started < 180000) {
    checkStop();
    await sleep(1000);
    h = await refreshDeepPill();
    if (h.ready) return;
    if (h.error) throw new Error(`B 模型错误：${h.error}`);
    if (Math.floor((Date.now() - started) / 1000) % 5 === 0) {
      setStatus(`等待 B 模型加载… ${Math.round((Date.now() - started) / 1000)}s`);
    }
  }
  throw new Error('B 模型加载超时');
}

// --------------------------------------------------------------------------- //
// 固定帧提取（只做一次，重复运行直接复用 ROI / 人脸框）
// --------------------------------------------------------------------------- //
async function extractFrames(endSec) {
  const video = state.video;
  const meta = state.meta;
  if (!video || !meta) throw new Error('视频未就绪');
  const times = planFrameTimes(meta.duration, FPS, { maxSeconds: endSec });
  const grabber = createFrameGrabber(video, { maxWidth: 960 });

  if (!state.detector) {
    setStatus('加载人脸检测器…');
    state.detector = await getFaceDetector();
  }

  state.frames = [];
  state.extractedEnd = times.length ? times[times.length - 1] : 0;
  let box = null;
  let detected = 0;

  for (let i = 0; i < times.length; i++) {
    checkStop();
    const t = times[i];
    await grabber.grab(t);

    if (i % DETECT_EVERY === 0) {
      const dets = await detectFaces(state.detector, grabber.canvas);
      const hit = largestFace(dets);
      if (hit) {
        box = clampBox(hit.box, grabber.width, grabber.height);
        detected++;
      }
    }

    const roi = box
      ? sampleRoiMeans(grabber.canvas, box)
      : { forehead: null, left_cheek: null, right_cheek: null };
    state.frames.push({
      t: round(t, 4),
      roi,
      box: box ? { ...box } : null,
    });

    if (i % 15 === 0 || i === times.length - 1) {
      const pct = 0.04 + 0.36 * ((i + 1) / times.length);
      setProgress(pct, `固定帧 ${i + 1}/${times.length} · t=${t.toFixed(2)}s`, true);
      if (i % 60 === 0) {
        setStatus(`抽帧与人脸检测：${i + 1}/${times.length}（命中 ${detected} 次）`);
      }
    }
  }

  state.extractedEnd = state.frames.length ? state.frames[state.frames.length - 1].t : 0;
  log(`固定帧完成：${state.frames.length} 帧 @ ${FPS}fps，${state.extractedEnd.toFixed(2)}s；人脸检测 ${detected} 次`);
}

// --------------------------------------------------------------------------- //
// payload：固定窗口，不插值、不跨断档
// --------------------------------------------------------------------------- //
function windowFrames(tEnd) {
  const t0 = Math.max(0, tEnd - WINDOW_SEC);
  return state.frames.filter((f) => f.t >= t0 - 1e-9 && f.t <= tEnd + 1e-9);
}

function buildPayload(tEnd) {
  const win = windowFrames(tEnd);
  const payload = {
    fs: FPS,
    timestamps: win.map((f) => f.t),
    forehead: [],
    left_cheek: [],
    right_cheek: [],
  };
  for (const name of ROI_NAMES) {
    payload[name] = win.map((f) => (f.roi && f.roi[name] ? f.roi[name] : null));
  }
  return { payload, frames: win };
}

function payloadHash(payload) {
  return fnv1a(JSON.stringify(payload));
}

function checkpoints(endSec) {
  const out = [];
  for (let t = A_START; t <= endSec + 1e-9; t += STEP_SEC) out.push(round(t, 3));
  return out;
}

// --------------------------------------------------------------------------- //
// A / A+：固定 payload + 视频时间
// --------------------------------------------------------------------------- //
async function runAplusPass(endSec, passLabel) {
  checkStop();
  await postJson('/aplus/reset', {});
  const a = [];
  const ap = [];
  const hashes = [];
  const points = checkpoints(endSec);

  for (let i = 0; i < points.length; i++) {
    checkStop();
    const t = points[i];
    const { payload } = buildPayload(t);
    if (!payload.timestamps.length || payload.timestamps.length < 6 * FPS) continue;
    const hash = payloadHash(payload);
    hashes.push(hash);

    try {
      const r = await postJson('/physio/estimate', payload);
      a.push({ t, hash, ...r });
    } catch (err) {
      a.push({ t, hash, error: err.message });
    }

    if (t >= APLUS_START) {
      try {
        const r = await postJson('/aplus/estimate', {
          ...payload,
          replay: true,
          replayTime: t,
        });
        ap.push({ t, hash, ...r });
      } catch (err) {
        ap.push({ t, hash, error: err.message });
      }
    }

    const pct = 0.42 + 0.20 * ((i + 1) / points.length);
    setProgress(pct, `${passLabel} ${i + 1}/${points.length} · t=${t.toFixed(1)}s`, true);
  }
  return { a, ap, hashes };
}

function compareSeries(first, second, fields) {
  const byT = new Map((second || []).map((r) => [r.t, r]));
  let total = 0;
  let same = 0;
  let firstT = null;
  let firstDiff = null;
  for (const r of first || []) {
    const o = byT.get(r.t);
    if (!o) continue;
    total++;
    let ok = true;
    for (const f of fields) {
      if (!sameNum(r[f], o[f])) { ok = false; break; }
    }
    if (ok) same++;
    else if (!firstT) { firstT = r.t; firstDiff = fields.find((f) => !sameNum(r[f], o[f])); }
  }
  return { total, same, firstT, firstDiff };
}

function compareHashes(first, second) {
  const n = Math.min(first?.length || 0, second?.length || 0);
  let same = 0;
  for (let i = 0; i < n; i++) if (first[i] === second[i]) same++;
  return { total: n, same };
}

// --------------------------------------------------------------------------- //
// B：固定帧 + 固定时间戳，reset 后顺序送入
// --------------------------------------------------------------------------- //
async function replayB(endSec) {
  if (!state.frames.length) throw new Error('没有固定帧');
  await ensureB();
  setStatus('B：重置模型…');
  await postJson('/deep/reset', {});
  checkStop();

  const canvas = document.createElement('canvas');
  canvas.width = B_W;
  canvas.height = B_H;
  const ctx = canvas.getContext('2d', { willReadFrequently: false });
  const results = [];
  let next = B_START;

  for (let i = 0; i < state.frames.length; i++) {
    checkStop();
    const f = state.frames[i];
    if (f.t > endSec + 1e-9) break;

    await seekTo(state.video, f.t);
    ctx.drawImage(state.video, 0, 0, B_W, B_H);
    const image = canvas.toDataURL('image/jpeg', 0.7);
    await postJson('/deep/frame', { image, timestamp: f.t });

    while (next <= f.t + 1e-9 && next <= endSec + 1e-9) {
      try {
        const r = await postJson('/deep/result', {
          replayTime: next,
          waitUntil: next,
          timeout: 10,
        });
        results.push({ t: next, ...r });
      } catch (err) {
        results.push({ t: next, ready: false, error: err.message });
      }
      next = round(next + 1, 3);
    }

    if (i % 15 === 0 || i === state.frames.length - 1) {
      const pct = 0.64 + 0.20 * ((i + 1) / state.frames.length);
      setProgress(pct, `B 固定帧 ${i + 1}/${state.frames.length} · 检查点 ${results.length}`, true);
      if (i % 60 === 0) setStatus(`B 重放：${i + 1}/${state.frames.length} 帧，已记录 ${results.length} 个检查点`);
    }
  }

  // 最后一个检查点若正好等于末尾，补一次
  while (next <= endSec + 1e-9) {
    const r = await postJson('/deep/result', { replayTime: next, waitUntil: next, timeout: 10 });
    results.push({ t: next, ...r });
    next = round(next + 1, 3);
  }
  log(`B 重放完成：${results.length} 个检查点`);
  return results;
}

// --------------------------------------------------------------------------- //
// 主流程
// --------------------------------------------------------------------------- //
async function startReplay({ repeat = false } = {}) {
  if (state.running || !state.file) return;
  state.running = true;
  state.stopping = false;
  $('btnStart').disabled = true;
  $('btnStop').disabled = false;
  $('btnRepeat').disabled = true;
  $('btnExport').disabled = true;
  $('maxSec').disabled = true;
  $('runB').disabled = true;

  const maxSec = Number($('maxSec').value) || 0;
  const endSec = maxSec > 0 ? Math.min(state.meta.duration, maxSec) : state.meta.duration;
  const includeB = $('runB').checked && endSec >= B_START + 1;

  try {
    if (!repeat || !state.frames.length || endSec > state.extractedEnd + 0.2) {
      setStatus('固定抽帧：15fps，逐点 seek…');
      await extractFrames(endSec);
    } else {
      log(`复用固定帧：${state.frames.length} 帧，截止 ${state.extractedEnd.toFixed(2)}s`);
    }

    setProgress(0.42, 'A / A+ 第一遍（固定 payload）…', true);
    setStatus('A / A+ 第一遍…');
    const first = await runAplusPass(endSec, 'A/A+ 第一遍');

    setProgress(0.62, 'A / A+ 第二遍（验证复跑）…', true);
    setStatus('A / A+ 第二遍：同一 payload、reset 后复跑…');
    const second = await runAplusPass(endSec, 'A/A+ 第二遍');

    const hashCmp = compareHashes(first.hashes, second.hashes);
    const aCmp = compareSeries(first.a, second.a, ['heart_rate', 'heart_rate_confidence']);
    const apCmp = compareSeries(first.ap, second.ap, ['hr', 'rawHr', 'quality', 'reliable', 'sqi']);

    let b = null;
    if (includeB) {
      setProgress(0.64, 'B 固定帧重放…', true);
      setStatus('B 固定帧重放：reset 后按视频时间顺序送入…');
      try {
        b = await replayB(endSec);
      } catch (err) {
        if (err.name !== 'AbortError') {
          log(`B 重放失败：${err.message}`);
          setStatus(`B 重放失败：${err.message}`);
        }
        b = null;
      }
    } else if ($('runB').checked) {
      log(`最大秒数 ${endSec.toFixed(1)}s < ${B_START + 1}s，本次跳过 B。`);
    }

    const run = {
      time: new Date().toISOString(),
      meta: { ...state.meta },
      params: { fps: FPS, windowSec: WINDOW_SEC, stepSec: STEP_SEC, endSec, includeB },
      frames: state.frames.length,
      first,
      second,
      comparison: { hashCmp, aCmp, apCmp },
      b,
    };

    state.previous = state.last;
    state.last = run;
    state.running = false;
    $('btnStop').disabled = true;
    $('btnStart').disabled = false;
    $('btnRepeat').disabled = false;
    $('btnExport').disabled = false;
    $('maxSec').disabled = false;
    $('runB').disabled = false;
    setProgress(1, '完成', false);
    setStatus('固定重放完成。A/A+ 已自动复跑；B 可点“再跑一次并对比”。');
    renderAll(run, { repeat });
    log(`完成：${first.a.length} 个 A 点，${first.ap.length} 个 A+ 点，${b?.length || 0} 个 B 点。`);
  } catch (err) {
    state.running = false;
    $('btnStop').disabled = true;
    $('btnStart').disabled = false;
    $('btnRepeat').disabled = !state.last;
    $('btnExport').disabled = !state.last;
    $('maxSec').disabled = false;
    $('runB').disabled = false;
    if (err.name === 'AbortError') {
      setProgress(0, '已停止', false);
      setStatus('已停止。');
      log('用户停止。');
    } else {
      setProgress(0, `失败：${err.message}`, false);
      setStatus(`失败：${err.message}`);
      log(`失败：${err.message}`);
    }
  }
}

// --------------------------------------------------------------------------- //
// 渲染
// --------------------------------------------------------------------------- //
function resetResultUI() {
  for (const id of ['aHr', 'aplusHr', 'bHr']) $(id).innerHTML = '—<small>bpm</small>';
  $('aState').textContent = '未运行';
  $('aplusState').textContent = '未运行';
  $('bState').textContent = '未运行';
  for (const id of ['aConf', 'aSpan', 'aplusQuality', 'aplusSqi', 'aplusRoi', 'aplusRaw', 'bSqi', 'bRaw']) {
    $(id).textContent = '—';
  }
  for (const id of ['detHash', 'detA', 'detAplus', 'detB']) {
    $(id).textContent = '—';
    $(id).className = 'muted';
  }
  $('detNote').textContent = '首次运行后会自动复跑 A/A+；B 需要点“再跑一次并对比”验证。';
  $('tableBody').innerHTML = '<tr><td colspan="8" style="text-align:center;color:#8c9aad">尚未运行</td></tr>';
  drawCharts(null);
}

function lastOf(list) {
  if (!Array.isArray(list)) return null;
  for (let i = list.length - 1; i >= 0; i--) {
    const r = list[i];
    if (!r || r.error) continue;
    if (r.heart_rate != null || r.hr != null || r.rawHr != null) return r;
  }
  return null;
}

function showHr(el, value, digits = 0) {
  if (value == null || !Number.isFinite(Number(value))) {
    el.innerHTML = '—<small>bpm</small>';
  } else {
    el.innerHTML = `${Number(value).toFixed(digits)}<small>bpm</small>`;
  }
}

function setDet(id, cmp, note) {
  const el = $(id);
  if (!cmp) {
    el.textContent = '未验证';
    el.className = 'muted';
    return;
  }
  el.textContent = `${cmp.same}/${cmp.total}`;
  el.className = cmp.total > 0 && cmp.same === cmp.total ? 'ok' : 'bad';
  el.title = note || '';
}

function renderAll(run, { repeat = false } = {}) {
  const aLast = lastOf(run.first.a);
  const apLast = lastOf(run.first.ap);
  const bLast = lastOf(run.b);

  showHr($('aHr'), aLast?.heart_rate, 0);
  $('aState').textContent = aLast ? (aLast.heart_rate_confidence != null ? '固定帧结果' : '无有效值') : '无结果';
  $('aConf').textContent = aLast?.heart_rate_confidence != null ? aLast.heart_rate_confidence.toFixed(3) : '—';
  $('aSpan').textContent = `${run.first.a.length} 个`;

  const apShown = apLast ? (apLast.hr ?? apLast.rawHr) : null;
  showHr($('aplusHr'), apShown, 0);
  if (apLast) {
    $('aplusState').textContent = apLast.reliable
      ? (apLast.status === 'live' ? '可信 / 实时' : `可信 / ${apLast.status}`)
      : apLast.status === 'holding'
        ? `低质量·保持${apLast.heldSec != null ? `（${Math.round(apLast.heldSec)}s）` : ''}`
        : '低质量候选';
    $('aplusQuality').textContent = apLast.quality != null ? apLast.quality.toFixed(3) : '—';
    $('aplusSqi').textContent = apLast.sqi != null ? apLast.sqi.toFixed(3) : '—';
    $('aplusRoi').textContent = apLast.activeRoi ? `${apLast.activeRoi}${apLast.fallback ? '（回退）' : ''}` : '—';
    $('aplusRaw').textContent = apLast.rawHr != null ? `${apLast.rawHr.toFixed(1)} bpm` : '—';
  } else {
    $('aplusState').textContent = '无结果';
    $('aplusQuality').textContent = '—';
    $('aplusSqi').textContent = '—';
    $('aplusRoi').textContent = '—';
    $('aplusRaw').textContent = '—';
  }

  showHr($('bHr'), bLast ? (bLast.hr ?? bLast.rawHr) : null, 0);
  $('bState').textContent = bLast ? (bLast.status || '—') : (run.b === null ? '本次未运行' : '无结果');
  $('bSqi').textContent = bLast?.sqi != null ? bLast.sqi.toFixed(3) : '—';
  $('bRaw').textContent = bLast?.rawHr != null ? `${bLast.rawHr.toFixed(1)} bpm` : '—';

  setDet('detHash', run.comparison?.hashCmp);
  setDet('detA', run.comparison?.aCmp);
  setDet('detAplus', run.comparison?.apCmp);

  const prevB = state.previous?.b;
  if (repeat && prevB && run.b) {
    const bCmp = compareSeries(prevB, run.b, ['hr', 'rawHr', 'sqi']);
    setDet('detB', bCmp);
    $('detNote').textContent = bCmp.same === bCmp.total
      ? `B 两次固定重放逐点一致（${bCmp.same}/${bCmp.total}）。`
      : `B 仍有差异：第一个不一致检查点 t=${bCmp.firstT ?? '?'}s，字段 ${bCmp.firstDiff || '?'}。`;
  } else if (run.b) {
    $('detB').textContent = '单次';
    $('detB').className = 'muted';
    const cmp = run.comparison?.apCmp;
    $('detNote').textContent = cmp
      ? `A/A+ 已用同一 payload 复跑；A+ ${cmp.same}/${cmp.total} 一致。B 本次只跑一次，点“再跑一次并对比”验证。`
      : 'B 本次只跑一次，点“再跑一次并对比”验证。';
  } else {
    $('detB').textContent = '未运行';
    $('detB').className = 'muted';
    $('detNote').textContent = '本次没有运行 B。';
  }

  renderTable(run);
  drawCharts(run);
}

function fmtVal(v, digits = 0) {
  return v == null || !Number.isFinite(Number(v)) ? '—' : Number(v).toFixed(digits);
}

function renderTable(run) {
  const rows = new Map();
  for (const r of run.first.a) rows.set(r.t, { t: r.t, hash: r.hash, a: r });
  for (const r of run.first.ap) {
    const cur = rows.get(r.t) || { t: r.t, hash: r.hash };
    cur.ap = r;
    rows.set(r.t, cur);
  }
  for (const r of run.b || []) {
    const cur = rows.get(r.t) || { t: r.t };
    cur.b = r;
    rows.set(r.t, cur);
  }
  const list = [...rows.values()].sort((x, y) => x.t - y.t).slice(-16);
  if (!list.length) {
    $('tableBody').innerHTML = '<tr><td colspan="8" style="text-align:center;color:#8c9aad">没有检查点</td></tr>';
    return;
  }
  $('tableBody').innerHTML = list.map((row) => {
    const a = row.a?.heart_rate;
    const apRaw = row.ap ? (row.ap.hr ?? row.ap.rawHr) : null;
    const bRaw = row.b ? (row.b.hr ?? row.b.rawHr) : null;
    return `<tr>
      <td class="mono">${row.t.toFixed(1)}</td>
      <td class="mono">${row.hash || row.ap?.hash || '—'}</td>
      <td class="mono">${fmtVal(a)}</td>
      <td class="mono">${fmtVal(apRaw)}</td>
      <td class="mono">${row.ap?.quality != null ? row.ap.quality.toFixed(3) : '—'}</td>
      <td>${row.ap ? (row.ap.reliable ? '是' : '否') : '—'}</td>
      <td class="mono">${fmtVal(bRaw)}</td>
      <td class="mono">${row.b?.sqi != null ? row.b.sqi.toFixed(3) : '—'}</td>
    </tr>`;
  }).join('');
}

function drawCharts(run) {
  if (!run) {
    drawLineChart($('hrChart'), { series: [], xRange: [0, 1], yRange: [0, 1] });
    drawLineChart($('qChart'), { series: [], xRange: [0, 1], yRange: [0, 1] });
    return;
  }
  const aPts = run.first.a.filter((r) => Number.isFinite(Number(r.heart_rate))).map((r) => ({ x: r.t, y: r.heart_rate }));
  const apPts = run.first.ap
    .map((r) => ({ x: r.t, y: Number(r.hr ?? r.rawHr) }))
    .filter((p) => Number.isFinite(p.y));
  const bPts = (run.b || [])
    .map((r) => ({ x: r.t, y: Number(r.hr ?? r.rawHr) }))
    .filter((p) => Number.isFinite(p.y));
  drawLineChart($('hrChart'), {
    series: [
      { label: 'A', color: '#22d3ee', points: aPts },
      { label: 'A+', color: '#34d399', points: apPts },
      { label: 'B', color: '#a78bfa', points: bPts },
    ],
    yLabel: 'bpm',
  });

  const aQ = run.first.a.filter((r) => Number.isFinite(Number(r.heart_rate_confidence))).map((r) => ({ x: r.t, y: r.heart_rate_confidence }));
  const apQ = run.first.ap.filter((r) => Number.isFinite(Number(r.quality))).map((r) => ({ x: r.t, y: r.quality }));
  const bQ = (run.b || []).filter((r) => Number.isFinite(Number(r.sqi))).map((r) => ({ x: r.t, y: r.sqi }));
  drawLineChart($('qChart'), {
    series: [
      { label: 'A 置信度', color: '#22d3ee', points: aQ },
      { label: 'A+ 质量', color: '#34d399', points: apQ },
      { label: 'B SQI', color: '#a78bfa', points: bQ },
    ],
    yRange: [0, 1],
    yLabel: 'score',
  });
}

// --------------------------------------------------------------------------- //
// 导出
// --------------------------------------------------------------------------- //
function exportJson() {
  if (!state.last) return;
  const data = {
    tool: 'deterministic-replay',
    generatedAt: new Date().toISOString(),
    meta: state.last.meta,
    params: state.last.params,
    frames: state.last.frames,
    first: state.last.first,
    second: state.last.second,
    comparison: state.last.comparison,
    b: state.last.b,
    previousB: state.previous?.b || null,
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `replay-${Date.now()}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// --------------------------------------------------------------------------- //
// 事件
// --------------------------------------------------------------------------- //
$('file').addEventListener('change', (e) => {
  const f = e.target.files?.[0];
  if (f) loadFile(f);
});
$('drop').addEventListener('click', () => $('file').click());
$('drop').addEventListener('dragover', (e) => {
  e.preventDefault();
  $('drop').classList.add('over');
});
$('drop').addEventListener('dragleave', () => $('drop').classList.remove('over'));
$('drop').addEventListener('drop', (e) => {
  e.preventDefault();
  $('drop').classList.remove('over');
  const f = e.dataTransfer?.files?.[0];
  if (f) loadFile(f);
});
$('btnStart').addEventListener('click', () => startReplay({ repeat: false }));
$('btnRepeat').addEventListener('click', () => startReplay({ repeat: true }));
$('btnStop').addEventListener('click', () => {
  state.stopping = true;
  setStatus('正在停止…');
  log('请求停止。');
});
$('btnExport').addEventListener('click', exportJson);

$('video').addEventListener('timeupdate', () => {
  const v = $('video');
  $('playTime').textContent = `${formatTime(v.currentTime)} / ${formatTime(v.duration || state.meta?.duration || 0)}`;
});

window.addEventListener('resize', () => drawCharts(state.last));

refreshDeepPill();
setInterval(() => {
  if (!state.running) refreshDeepPill();
}, 5000);
setProgress(0, '等待选择视频。', false);
