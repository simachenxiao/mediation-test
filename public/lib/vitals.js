// 生命体征（rPPG 非接触式生理指标）离线分析模块——可独立运行。
//
// 流程：逐帧 seek → 周期人脸检测（框跟随）→ 三路面部 ROI 像素采样
//       → 均匀重采样 → 分窗口 POST /physio/estimate（Python 侧车）
//       → 心率/呼吸/血氧/血压/HRV 汇总。
//
// 与微表情链路共享的只有“人脸检测 / 抽帧 / ROI 采样”这些基础工具，
// 算法与结果完全独立。

import { getFaceDetector, detectFaces, largestFace, clampBox } from './face-detect.js';
import { ROI_DEFS, sampleRoiMeans } from './roi.js';
import {
  createFrameGrabber,
  planFrameTimes,
  interpLinear,
  decimate,
  abortError,
  throwIfAborted,
} from './video-frames.js';

const EMPTY_ROI = { forehead: null, left_cheek: null, right_cheek: null };

// --------------------------------------------------------------------------- //
// 重采样 / 请求
// --------------------------------------------------------------------------- //
function prepareCurves(samples) {
  const curves = {};
  for (const name of Object.keys(ROI_DEFS)) {
    const ts = [];
    const rs = [];
    const gs = [];
    const bs = [];
    for (const s of samples) {
      const v = s.roi?.[name];
      if (!v) continue;
      ts.push(s.t);
      rs.push(v[0]);
      gs.push(v[1]);
      bs.push(v[2]);
    }
    curves[name] = { ts, rs, gs, bs };
  }
  return curves;
}

/** 把一个时间窗口重采样到固定 fs 的均匀网格，拼成侧车需要的 payload。 */
export function buildPhysioPayload(samples, curves, t0, t1, fs) {
  const n = Math.max(2, Math.floor((t1 - t0) * fs + 1e-6) + 1);
  const timestamps = new Array(n);
  for (let i = 0; i < n; i++) timestamps[i] = t0 + i / fs;

  const payload = { fs, timestamps, forehead: [], left_cheek: [], right_cheek: [] };
  for (const name of Object.keys(ROI_DEFS)) {
    const c = curves[name];
    const out = payload[name];
    out.length = 0;
    if (!c.ts.length) {
      for (let i = 0; i < n; i++) out.push(null);
      continue;
    }
    for (let i = 0; i < n; i++) {
      const t = timestamps[i];
      out.push([
        interpLinear(c.ts, c.rs, t),
        interpLinear(c.ts, c.gs, t),
        interpLinear(c.ts, c.bs, t),
      ]);
    }
  }
  return payload;
}

async function postEstimate(payload, signal) {
  const ctrl = new AbortController();
  const onAbort = () => ctrl.abort();
  if (signal) signal.addEventListener('abort', onAbort, { once: true });
  const timer = setTimeout(() => ctrl.abort(), 45000);
  try {
    const res = await fetch('/physio/estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      const hint =
        res.status === 502
          ? '（生理计算侧车未就绪：请确认 Python 侧车已启动，可访问 /physio/health 查看）'
          : '';
      throw new Error(`physio HTTP ${res.status}${hint}${body ? ': ' + body.slice(0, 300) : ''}`);
    }
    return await res.json();
  } catch (err) {
    if (signal?.aborted) throw abortError();
    if (err.name === 'AbortError') throw new Error('生理估计请求超时（Python 侧车可能未启动或过慢）');
    throw err;
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', onAbort);
  }
}

function median(values) {
  if (!values.length) return null;
  const arr = [...values].sort((a, b) => a - b);
  const mid = arr.length >> 1;
  return arr.length % 2 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;
}

const round1 = (v) => (Number.isFinite(v) ? Math.round(v * 10) / 10 : null);

function summarize(samples, windows) {
  const ready = windows.filter((w) => w.result && w.result.ready);
  const hrs = ready.map((w) => w.result.heart_rate).filter(Number.isFinite);
  const rrs = ready.map((w) => w.result.respiratory_rate).filter(Number.isFinite);
  const spo2s = ready.map((w) => w.result.spo2).filter(Number.isFinite);
  const quals = ready.map((w) => w.result.quality).filter(Number.isFinite);
  const last = ready.length ? ready[ready.length - 1].result : null;

  const roiValid = {};
  for (const name of Object.keys(ROI_DEFS)) {
    roiValid[name] = samples.length
      ? samples.filter((s) => s.roi?.[name]).length / samples.length
      : 0;
  }

  return {
    duration: samples.length ? samples[samples.length - 1].t : 0,
    frameCount: samples.length,
    faceHitRate: samples.length ? samples.filter((s) => s.box).length / samples.length : 0,
    roiValid,
    windows: windows.length,
    readyWindows: ready.length,
    hrMedian: round1(median(hrs)),
    hrLatest: last ? last.heart_rate : null,
    rrMedian: round1(median(rrs)),
    spo2Median: round1(median(spo2s)),
    qualityMedian: round1(median(quals)),
    bloodPressure: last ? last.blood_pressure : null,
    hrv: last ? last.hrv : null,
    heartRateChange: last ? last.heart_rate_change_bpm_per_min : null,
    lastResult: last,
  };
}

// --------------------------------------------------------------------------- //
// 主流程
// --------------------------------------------------------------------------- //
/**
 * @param {HTMLVideoElement} video
 * @param {{
 *   fps?:number, maxSeconds?:number, detectEverySec?:number, boxHoldSec?:number,
 *   windowSec?:number, stepSec?:number, minWindowSec?:number, signal?:AbortSignal
 * }} opts
 * @param {{onProgress?:Function,onSample?:Function,onWindow?:Function,onLog?:Function}} hooks
 */
export async function analyzeVitals(video, opts = {}, hooks = {}) {
  const fps = Number(opts.fps) > 0 ? Number(opts.fps) : 15;
  const maxSeconds = Number(opts.maxSeconds) || 0;
  const detectEverySec = Number(opts.detectEverySec) > 0 ? Number(opts.detectEverySec) : 0.5;
  const boxHoldSec = Number(opts.boxHoldSec) > 0 ? Number(opts.boxHoldSec) : 1.5;
  const windowSec = Number(opts.windowSec) > 0 ? Number(opts.windowSec) : 20;
  const stepSec = Number(opts.stepSec) > 0 ? Number(opts.stepSec) : 5;
  const minWindowSec = Number(opts.minWindowSec) > 0 ? Number(opts.minWindowSec) : 6;
  const signal = opts.signal;
  const log = (m) => hooks.onLog?.(m);

  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  const times = planFrameTimes(duration, fps, { maxSeconds });
  if (!times.length) throw new Error('视频时长为 0，无法分析');

  log('加载人脸检测器…');
  const detector = await getFaceDetector();
  throwIfAborted(signal);

  const grabber = createFrameGrabber(video);
  const total = times.length;
  const samples = [];
  let lastDetectAt = -Infinity;
  let lastBox = null;
  let lastBoxAt = -Infinity;
  let detectCount = 0;
  let detectHit = 0;

  log(`开始 ROI 采样：${total} 帧 @ ${fps}fps（每 ${detectEverySec}s 检测一次人脸）`);

  for (let i = 0; i < total; i++) {
    throwIfAborted(signal);
    const t = times[i];
    const frame = await grabber.grab(t);

    if (t - lastDetectAt >= detectEverySec || !lastBox) {
      lastDetectAt = t;
      detectCount++;
      try {
        const dets = await detectFaces(detector, frame.canvas);
        const hit = largestFace(dets);
        if (hit) {
          lastBox = clampBox(hit.box, frame.width, frame.height);
          lastBoxAt = t;
          detectHit++;
        }
      } catch (err) {
        log(`t=${t.toFixed(1)}s 人脸检测失败：${err.message}`);
      }
    }

    let box = lastBox;
    if (box && t - lastBoxAt > boxHoldSec) box = null;
    const roi = box ? sampleRoiMeans(frame.canvas, box) : { ...EMPTY_ROI };
    const sample = { index: i, t, box, roi, valid: !!(roi.forehead || roi.left_cheek || roi.right_cheek) };
    samples.push(sample);

    hooks.onSample?.(sample, { total, fps, frameCanvas: frame.canvas });
    hooks.onProgress?.({ phase: 'sample', done: i + 1, total });
    if (i % 10 === 9) await new Promise((r) => setTimeout(r, 0));
  }

  throwIfAborted(signal);
  log(`采样完成：人脸命中 ${samples.filter((s) => s.box).length}/${total} 帧，开始分窗口估计…`);

  // ---- 分窗口估计 ----
  const curves = prepareCurves(samples);
  const dur = samples.length ? samples[samples.length - 1].t : 0;
  const win = Math.min(windowSec, dur);
  const starts = [];
  if (dur > 0) {
    for (let s = 0; s + win <= dur + 1e-6; s += stepSec) starts.push(s);
    const endStart = Math.max(0, dur - win);
    if (!starts.length) starts.push(endStart);
    else if (starts[starts.length - 1] < endStart - 1e-6) starts.push(endStart);
  }
  if (dur < minWindowSec) {
    log(
      `视频有效时长 ${dur.toFixed(1)}s 小于建议窗口 ${minWindowSec}s，` +
        '生理指标可能因子窗口样本不足而无结果'
    );
  }
  const uniqStarts = [...new Set(starts.map((s) => Math.round(s * 1000) / 1000))];

  const windows = [];
  for (let i = 0; i < uniqStarts.length; i++) {
    throwIfAborted(signal);
    const t0 = uniqStarts[i];
    const t1 = Math.min(dur, t0 + win);
    const payload = buildPhysioPayload(samples, curves, t0, t1, fps);
    let result = null;
    let error = null;
    try {
      result = await postEstimate(payload, signal);
    } catch (err) {
      if (err.name === 'AbortError') throw err;
      error = err.message;
    }
    const winRec = { index: i, t0, t1, samples: payload.timestamps.length, result, error };
    windows.push(winRec);
    hooks.onWindow?.(winRec, { total: uniqStarts.length });
    hooks.onProgress?.({ phase: 'estimate', done: i + 1, total: uniqStarts.length });
  }

  const summary = summarize(samples, windows);
  if (windows.length && !windows.some((w) => w.result?.ready)) {
    const firstErr = windows.find((w) => w.error)?.error;
    if (firstErr) log(`生理估计未成功：${firstErr}`);
    else log('生理估计窗口样本不足（需要至少 60 帧有效 ROI）');
  } else {
    log(`生命体征分析完成：${summary.readyWindows}/${windows.length} 个窗口出结果`);
  }

  const picked = decimate(
    samples.map((_, i) => i),
    1500
  );
  const traces = {
    t: picked.map((i) => samples[i].t),
    forehead: picked.map((i) => samples[i].roi.forehead),
    left_cheek: picked.map((i) => samples[i].roi.left_cheek),
    right_cheek: picked.map((i) => samples[i].roi.right_cheek),
  };

  return {
    kind: 'vitals',
    meta: {
      fps,
      maxSeconds,
      detectEverySec,
      windowSec: win,
      stepSec,
      duration: dur,
      analyzedFrames: total,
      frameWidth: grabber.width,
      frameHeight: grabber.height,
      detectCount,
      detectHit,
    },
    samples,
    traces,
    windows,
    summary,
  };
}
