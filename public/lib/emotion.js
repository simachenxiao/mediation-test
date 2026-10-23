// 微表情（面部情绪）分析模块——可独立运行，不依赖生命体征链路。
//
// 流程：逐帧 seek → MediaPipe 人脸检测 → 人脸裁剪/归一化 → ONNX 8 类情绪推理
//       → 汇总时间线、微表情事件。
//
// 用法：
//   const res = await analyzeEmotion(video, { fps: 5, forceEp: 'wasm' }, {
//     onProgress({done,total}) {},
//     onFrame({ index, t, box, label, zh, confidence }) {},
//   });

import * as ort from '/ort/ort.webgpu.bundle.min.mjs';
import { getFaceDetector, detectFaces, largestFace, clampBox } from './face-detect.js';
import {
  createFrameGrabber,
  planFrameTimes,
  throwIfAborted,
} from './video-frames.js';

export const EMOTION_MODEL = '/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx';
export const EMOTION_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export const EMOTION_LABELS = [
  'Anger',
  'Contempt',
  'Disgust',
  'Fear',
  'Happiness',
  'Neutral',
  'Sadness',
  'Surprise',
];

export const EMOTION_ZH = {
  Anger: '生气',
  Contempt: '轻蔑',
  Disgust: '厌恶',
  Fear: '害怕',
  Happiness: '开心',
  Neutral: '平静',
  Sadness: '难过',
  Surprise: '惊讶',
};

export const EMOTION_COLORS = {
  Anger: '#ef4444',
  Contempt: '#a855f7',
  Disgust: '#84cc16',
  Fear: '#f59e0b',
  Happiness: '#22c55e',
  Neutral: '#64748b',
  Sadness: '#3b82f6',
  Surprise: '#06b6d4',
};

export const NEGATIVE_LABELS = new Set(['Anger', 'Disgust', 'Fear', 'Sadness']);

// --------------------------------------------------------------------------- //
// ONNX 会话（缓存；不同引擎参数各自一份）
// --------------------------------------------------------------------------- //
const sessionCache = new Map();

function isSoftwareAdapter(adapter) {
  if (adapter.isFallbackAdapter) return true;
  const info = adapter.info || {};
  const s = `${info.vendor || ''} ${info.architecture || ''} ${info.description || ''}`.toLowerCase();
  return (
    s.includes('swiftshader') ||
    s.includes('software') ||
    s.includes('llvmpipe') ||
    s.includes('basic render')
  );
}

async function hasHardwareGpu() {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;
  for (let i = 0; i < 3; i++) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter && !isSoftwareAdapter(adapter)) return true;
    } catch {
      /* 继续重试 */
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}

async function createSession(forceEp) {
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(navigator.hardwareConcurrency || 4, 8)
    : 1;

  let candidates;
  if (forceEp === 'wasm') candidates = ['wasm'];
  else if (forceEp === 'webgpu') candidates = ['webgpu', 'wasm'];
  else candidates = (await hasHardwareGpu()) ? ['webgpu', 'wasm'] : ['wasm'];

  let lastErr = null;
  for (const ep of candidates) {
    try {
      const session = await ort.InferenceSession.create(EMOTION_MODEL, {
        executionProviders: [ep],
      });
      // 预热一次，避免第一帧被着色器编译拖慢
      const zero = new Float32Array(3 * EMOTION_SIZE * EMOTION_SIZE);
      await session.run({ input: new ort.Tensor('float32', zero, [1, 3, EMOTION_SIZE, EMOTION_SIZE]) });
      const engine = ep === 'webgpu' ? 'WebGPU (GPU)' : 'WASM (CPU)';
      return { session, engine, ep };
    } catch (err) {
      lastErr = err;
      console.warn(`[emotion] 推理引擎 ${ep} 不可用，尝试下一个`, err);
    }
  }
  throw lastErr || new Error('无法创建情绪 ONNX 会话');
}

/** 获取（并缓存）情绪推理会话。 */
export function getEmotionSession(forceEp = '') {
  const key = forceEp || 'auto';
  if (!sessionCache.has(key)) {
    sessionCache.set(
      key,
      createSession(forceEp).catch((err) => {
        sessionCache.delete(key);
        throw err;
      })
    );
  }
  return sessionCache.get(key);
}

// --------------------------------------------------------------------------- //
// 预处理 / 推理
// --------------------------------------------------------------------------- //
let cropCanvas = null;
let cropCtx = null;
function getCropCtx() {
  if (!cropCtx) {
    cropCanvas = document.createElement('canvas');
    cropCanvas.width = EMOTION_SIZE;
    cropCanvas.height = EMOTION_SIZE;
    cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });
    cropCtx.imageSmoothingEnabled = true;
    cropCtx.imageSmoothingQuality = 'high';
  }
  return cropCtx;
}

let thumbCanvas = null;
function makeThumb(sourceCanvas) {
  if (!thumbCanvas) {
    thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = 96;
    thumbCanvas.height = 96;
  }
  const ctx = thumbCanvas.getContext('2d');
  ctx.drawImage(sourceCanvas, 0, 0, 96, 96);
  return thumbCanvas.toDataURL('image/jpeg', 0.72);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/**
 * 按人脸框裁出正方形人脸并归一化到模型输入。
 * @returns {{input:Float32Array, thumb:string, side:number, rgbMean:number[]}}
 */
export function prepareFaceInput(source, box, { margin = 1.3 } = {}) {
  const side = Math.max(box.width, box.height) * margin;
  const cx = box.originX + box.width / 2;
  const cy = box.originY + box.height / 2;

  const ctx = getCropCtx();
  ctx.clearRect(0, 0, EMOTION_SIZE, EMOTION_SIZE);
  ctx.drawImage(
    source,
    cx - side / 2,
    cy - side / 2,
    side,
    side,
    0,
    0,
    EMOTION_SIZE,
    EMOTION_SIZE
  );

  const { data } = ctx.getImageData(0, 0, EMOTION_SIZE, EMOTION_SIZE);
  const plane = EMOTION_SIZE * EMOTION_SIZE;
  const input = new Float32Array(3 * plane);
  let sr = 0;
  let sg = 0;
  let sb = 0;
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    sr += r;
    sg += g;
    sb += b;
    input[i] = (r / 255 - MEAN[0]) / STD[0];
    input[plane + i] = (g / 255 - MEAN[1]) / STD[1];
    input[2 * plane + i] = (b / 255 - MEAN[2]) / STD[2];
  }
  return {
    input,
    thumb: makeThumb(cropCanvas),
    side,
    rgbMean: [sr / plane, sg / plane, sb / plane],
  };
}

/** 对一张已预处理的人脸执行 8 类情绪推理。 */
export async function predictEmotion(session, input) {
  const t0 = performance.now();
  const feeds = {
    input: new ort.Tensor('float32', input, [1, 3, EMOTION_SIZE, EMOTION_SIZE]),
  };
  const out = await session.run(feeds);
  const tensor = out[session.outputNames?.[0]] || out.output;
  const probs = softmax(tensor.data);
  const t1 = performance.now();

  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  const label = EMOTION_LABELS[best];
  return {
    label,
    zh: EMOTION_ZH[label],
    confidence: probs[best],
    probs,
    timing: { runMs: t1 - t0, totalMs: t1 - t0 },
  };
}

// --------------------------------------------------------------------------- //
// 微表情事件 / 汇总
// --------------------------------------------------------------------------- //
function buildTimeline(frames, fps) {
  const segments = [];
  let cur = null;
  for (const f of frames) {
    if (!f.label) {
      cur = null;
      continue;
    }
    const gap = cur ? f.t - cur.t1 : Infinity;
    if (cur && cur.label === f.label && gap <= 1.6 / fps) {
      cur.t1 = f.t;
      cur.n += 1;
      cur.confSum += f.confidence;
      continue;
    }
    cur = {
      label: f.label,
      zh: f.zh,
      t0: f.t,
      t1: f.t,
      n: 1,
      confSum: f.confidence,
    };
    segments.push(cur);
  }
  return segments.map((s) => ({
    label: s.label,
    zh: s.zh,
    t0: s.t0,
    t1: s.t1,
    n: s.n,
    duration: s.t1 - s.t0,
    meanConfidence: s.confSum / s.n,
  }));
}

function findMicroEvents(segments, { microMaxSec = 1.2, microMinConf = 0.4, fps = 5 } = {}) {
  const frameDur = 1 / Math.max(1, fps);
  return segments
    .filter(
      (s) =>
        s.label !== 'Neutral' &&
        s.duration <= microMaxSec + frameDur * 0.51 &&
        s.meanConfidence >= microMinConf
    )
    .map((s) => ({
      ...s,
      // 单帧事件的 duration 为 0，用一帧时长表示
      duration: Math.max(s.duration, frameDur),
    }));
}

function summarize(frames, timeline, events, engine, analyzedFrames) {
  const labeled = frames.filter((f) => f.label);
  const counts = {};
  for (const l of EMOTION_LABELS) counts[l] = 0;
  let confSum = 0;
  let runMs = 0;
  let neg = 0;
  for (const f of labeled) {
    counts[f.label] = (counts[f.label] || 0) + 1;
    confSum += f.confidence;
    runMs += f.timing?.runMs || 0;
    if (NEGATIVE_LABELS.has(f.label)) neg += 1;
  }
  let dominant = 'Neutral';
  for (const l of EMOTION_LABELS) if (counts[l] > counts[dominant]) dominant = l;
  const n = labeled.length;
  return {
    engine,
    analyzedFrames,
    faceFrames: n,
    detectionRate: analyzedFrames ? n / analyzedFrames : 0,
    counts,
    dominant,
    dominantZh: EMOTION_ZH[dominant],
    negativeRatio: n ? neg / n : 0,
    meanConfidence: n ? confSum / n : 0,
    avgInferMs: n ? runMs / n : 0,
    eventCount: events.length,
    timelineSegments: timeline.length,
  };
}

// --------------------------------------------------------------------------- //
// 主流程
// --------------------------------------------------------------------------- //
/**
 * @param {HTMLVideoElement} video 已加载元数据的视频元素
 * @param {{fps?:number,maxSeconds?:number,margin?:number,forceEp?:string,microMaxSec?:number,microMinConf?:number,signal?:AbortSignal}} opts
 * @param {{onProgress?:Function,onFrame?:Function,onLog?:Function}} hooks
 */
export async function analyzeEmotion(video, opts = {}, hooks = {}) {
  const fps = Number(opts.fps) > 0 ? Number(opts.fps) : 5;
  const maxSeconds = Number(opts.maxSeconds) || 0;
  const margin = Number(opts.margin) || 1.3;
  const signal = opts.signal;
  const log = (m) => hooks.onLog?.(m);

  log('加载人脸检测器…');
  const detector = await getFaceDetector();
  throwIfAborted(signal);
  log('加载情绪推理模型…');
  const { session, engine } = await getEmotionSession(opts.forceEp);
  throwIfAborted(signal);

  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  const times = planFrameTimes(duration, fps, { maxSeconds });
  if (!times.length) throw new Error('视频时长为 0，无法分析');

  const grabber = createFrameGrabber(video);
  const frames = [];
  const total = times.length;
  log(`开始逐帧分析：${total} 帧 @ ${fps}fps，引擎 ${engine}`);

  for (let i = 0; i < total; i++) {
    throwIfAborted(signal);
    const t = times[i];
    const frame = await grabber.grab(t);

    let detections = [];
    try {
      detections = await detectFaces(detector, frame.canvas);
    } catch (err) {
      log(`第 ${i + 1} 帧人脸检测失败：${err.message}`);
    }
    const hit = largestFace(detections);
    const item = {
      index: i,
      t,
      face: !!hit,
      detCount: detections.length,
      box: hit ? clampBox(hit.box, frame.width, frame.height) : null,
      detScore: hit?.score ?? null,
      label: null,
      zh: null,
      confidence: 0,
      probs: null,
      thumb: null,
      timing: null,
    };

    if (hit && item.box) {
      const prep = prepareFaceInput(frame.canvas, item.box, { margin });
      const pred = await predictEmotion(session, prep.input);
      item.label = pred.label;
      item.zh = pred.zh;
      item.confidence = pred.confidence;
      item.probs = pred.probs;
      item.thumb = prep.thumb;
      item.timing = pred.timing;
      item.faceRGB = prep.rgbMean;
    }

    frames.push(item);
    hooks.onFrame?.(item, { total, fps, frameCanvas: frame.canvas });
    hooks.onProgress?.({ done: i + 1, total, phase: 'emotion' });

    // 无脸帧不会 await 推理，主动让出主线程，保证 UI 能刷新
    if (i % 5 === 4) await new Promise((r) => setTimeout(r, 0));
  }

  throwIfAborted(signal);
  const timeline = buildTimeline(frames, fps);
  const events = findMicroEvents(timeline, {
    microMaxSec: opts.microMaxSec,
    microMinConf: opts.microMinConf,
    fps,
  });
  const summary = summarize(frames, timeline, events, engine, total);
  log(
    `微表情分析完成：${summary.faceFrames}/${total} 帧检出人脸，` +
      `主情绪=${summary.dominantZh}，微表情事件=${events.length}`
  );

  return {
    kind: 'emotion',
    meta: {
      fps,
      maxSeconds,
      margin,
      engine,
      duration,
      analyzedFrames: total,
      frameWidth: grabber.width,
      frameHeight: grabber.height,
    },
    frames,
    samples: frames.filter((f) => f.label),
    timeline,
    events,
    summary,
  };
}
