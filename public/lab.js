// 离线视频分析实验室：上传视频 → 分步查看微表情 / 生命体征结果。
//
// 两条流水线完全独立：
//   * /lib/emotion.js  微表情（人脸检测 → 裁剪 → 8 类情绪 → 时间线）
//   * /lib/vitals.js   生命体征（人脸跟踪 → ROI 采样 → 分窗口 rPPG）
// 本文件只负责 UI：文件管理、步骤状态、进度、图表与导出。

import {
  probeVideoFile,
  createVideoElement,
  ensureVideoReady,
  seekTo,
  planFrameTimes,
  formatBytes,
  formatTime,
  decimate,
} from '/lib/video-frames.js';
import {
  analyzeEmotion,
  EMOTION_LABELS,
  EMOTION_ZH,
  EMOTION_COLORS,
  NEGATIVE_LABELS,
} from '/lib/emotion.js';
import { analyzeVitals } from '/lib/vitals.js';
import { ROI_DEFS, roiRects } from '/lib/roi.js';
import { drawLineChart, drawBarChart, drawBand } from '/lib/charts.js';

const $ = (id) => document.getElementById(id);
const ANALYSIS_MAX_SIDE = 960; // 与各模块内的 createFrameGrabber 默认值一致

const ROI_COLORS = {
  forehead: '#f59e0b',
  left_cheek: '#06b6d4',
  right_cheek: '#a855f7',
};
const ROI_ZH = {
  forehead: '额头',
  left_cheek: '左颊',
  right_cheek: '右颊',
};

const state = {
  file: null,
  url: null,
  meta: null,
  emoVideo: null,
  vitVideo: null,
  emo: null,
  vit: null,
  running: false,
  abort: null,
  emoSelected: 0,
  emoSampleByFrame: new Map(),
  emoRenderToken: 0,
  vitRenderToken: 0,
  vitChannel: 'G',
  vitTraces: null,
  liveTraces: null,
  plan: { emotion: false, vitals: false },
};

// --------------------------------------------------------------------------- //
// 基础 UI
// --------------------------------------------------------------------------- //
function log(msg, level = 'info') {
  const el = $('log');
  const time = new Date().toLocaleTimeString();
  const line = `[${time}] ${msg}`;
  el.textContent = el.textContent === '就绪。' ? line : `${el.textContent}\n${line}`;
  el.scrollTop = el.scrollHeight;
  if (level === 'error') console.error('[lab]', msg);
  else console.info('[lab]', msg);
}

function setProgress(pct, text, busy = false) {
  const bar = $('progressBar');
  bar.style.width = `${Math.max(0, Math.min(1, pct)) * 100}%`;
  bar.classList.toggle('busy', !!busy);
  $('progressText').textContent = text;
}

function moduleProgress(kind, done, total, text) {
  const active = Object.keys(state.plan).filter((k) => state.plan[k]);
  if (!active.length) return;
  const idx = active.indexOf(kind);
  const weight = 1 / active.length;
  const pct = (idx + Math.max(0, Math.min(1, done / Math.max(1, total)))) * weight;
  setProgress(pct, text, true);
}

const BADGE_TEXT = {
  pending: '待运行',
  running: '进行中',
  done: '完成',
  error: '失败',
  skip: '未选择',
};

function setStepState(id, kind, text) {
  const article = $(id);
  if (!article) return;
  const badge = article.querySelector('[data-badge]');
  badge.className = `badge ${kind === 'pending' ? '' : kind}`.trim();
  badge.textContent = text ?? BADGE_TEXT[kind] ?? kind;
}

function statChips(items) {
  return items
    .map(
      ([label, value, cls = '']) =>
        `<span class="stat ${cls}"><b>${value}</b>${label}</span>`
    )
    .join('');
}

function card(label, value, unit = '') {
  return `<div class="card-metric"><div class="label">${label}</div>
    <div class="value">${value ?? '—'}${unit ? `<span class="unit">${unit}</span>` : ''}</div></div>`;
}

function setKV(dl, values) {
  const dds = dl.querySelectorAll('dd');
  values.forEach((v, i) => {
    if (dds[i]) dds[i].textContent = v;
  });
}

function pct(v, digits = 1) {
  return Number.isFinite(v) ? `${(v * 100).toFixed(digits)}%` : '—';
}

function fixed(v, digits = 1, fallback = '—') {
  return Number.isFinite(v) ? v.toFixed(digits) : fallback;
}

function downloadText(name, text, mime = 'application/json') {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// 折叠步骤
document.querySelectorAll('[data-toggle]').forEach((header) => {
  header.addEventListener('click', () => header.parentElement.classList.toggle('collapsed'));
});

// --------------------------------------------------------------------------- //
// 文件选择
// --------------------------------------------------------------------------- //
const drop = $('drop');
const fileInput = $('file');
drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', (e) => {
  e.preventDefault();
  drop.classList.add('over');
});
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  const f = e.dataTransfer?.files?.[0];
  if (f) pickFile(f);
});
fileInput.addEventListener('change', () => {
  const f = fileInput.files?.[0];
  if (f) pickFile(f);
  fileInput.value = '';
});

async function pickFile(file) {
  if (state.running) {
    log('正在分析中，请先停止或等待完成后再换文件', 'error');
    return;
  }
  if (!/^video\//.test(file.type) && !/\.(mp4|webm|mov|mkv|avi|m4v)$/i.test(file.name)) {
    log(`不是视频文件：${file.name}`, 'error');
    return;
  }

  log(`读取视频：${file.name} (${formatBytes(file.size)})`);
  setProgress(0, '正在解码视频元数据…', true);

  disposeVideos();
  if (state.url) URL.revokeObjectURL(state.url);
  state.url = null;
  state.meta = null;
  state.file = null;
  resetAnalysis();

  let probe;
  try {
    probe = await probeVideoFile(file);
  } catch (err) {
    log(`视频读取失败：${err.message}`, 'error');
    setProgress(0, `视频读取失败：${err.message}`, false);
    setStepState('emoDecode', 'error', '解码失败');
    return;
  }

  state.file = file;
  state.url = probe.url;
  state.meta = probe.meta;
  // probe 用的临时 video 不再需要，保留 objectURL 给 preview 与分析 video
  probe.video.removeAttribute('src');
  probe.video.load();

  const preview = $('preview');
  preview.src = probe.url;
  preview.muted = true;
  await ensureVideoReady(preview).catch(() => {});

  fillFileMeta();
  refreshPlan();
  await drawFirstFrame(preview);

  $('previewHint').textContent = `${probe.meta.name} · 可直接播放预览，分析使用独立解码实例`;
  setStepState('emoDecode', 'done', '已解码');
  setProgress(0, '视频已就绪，可开始分析', false);
  setRunning(false);
  log(`视频就绪：${probe.meta.width}×${probe.meta.height}，${formatTime(probe.meta.duration)}`);
}

function fillFileMeta() {
  const m = state.meta;
  const dd = $('fileMeta').querySelectorAll('dd');
  dd[0].textContent = m.name;
  dd[1].textContent = formatBytes(m.size);
  dd[2].textContent = `${formatTime(m.duration)} (${m.duration.toFixed(2)}s)`;
  dd[3].textContent = `${m.width} × ${m.height}`;
  dd[3].title = m.type || '未知容器';
}

function refreshPlan() {
  const m = state.meta;
  if (!m) return;
  const fps = Number($('emoFps').value);
  const maxSeconds = Number($('maxSec').value);
  const times = planFrameTimes(m.duration, fps, { maxSeconds });
  setKV($('emoDecodeMeta'), [
    `${formatTime(m.duration)} (${m.duration.toFixed(2)}s)`,
    `${m.width} × ${m.height}`,
    `${fps} fps`,
    `${times.length} 帧`,
    '已解码',
  ]);
}

async function drawFirstFrame(preview) {
  try {
    await ensureVideoReady(preview);
    // seek 到一个非零时间点，强制解码器真正呈现一帧；
    // 否则 currentTime 恒为 0 时可能只拿到黑帧。
    const t = Math.min(0.04, Math.max(0.001, (preview.duration || 1) / 1000));
    await seekTo(preview, t).catch(() => {});
    if (typeof preview.requestVideoFrameCallback === 'function') {
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, 400);
        preview.requestVideoFrameCallback(() => {
          clearTimeout(timer);
          resolve();
        });
      });
    } else {
      await new Promise((r) => setTimeout(r, 80));
    }
    const vw = preview.videoWidth;
    const vh = preview.videoHeight;
    if (!vw || !vh) return;
    const scale = Math.min(1, 320 / vw);
    const canvas = $('emoFirstFrame');
    canvas.width = Math.round(vw * scale);
    canvas.height = Math.round(vh * scale);
    canvas.getContext('2d').drawImage(preview, 0, 0, canvas.width, canvas.height);
  } catch (err) {
    log(`首帧绘制失败：${err.message}`, 'error');
  }
}

async function ensureAnalysisVideo(kind) {
  const key = kind === 'emotion' ? 'emoVideo' : 'vitVideo';
  if (state[key]) return state[key];
  const video = createVideoElement(state.url);
  video.style.display = 'none';
  document.body.appendChild(video);
  await ensureVideoReady(video);
  if (!Number.isFinite(video.duration) || video.duration <= 0) {
    throw new Error('视频时长无效（可能是流式文件或编码不支持）');
  }
  state[key] = video;
  return video;
}

function disposeVideos() {
  for (const key of ['emoVideo', 'vitVideo']) {
    const v = state[key];
    if (!v) continue;
    try {
      v.pause();
      v.removeAttribute('src');
      v.load();
      v.remove();
    } catch {
      /* ignore */
    }
    state[key] = null;
  }
}

// --------------------------------------------------------------------------- //
// 参数 / 运行控制
// --------------------------------------------------------------------------- //
['emoFps', 'maxSec'].forEach((id) =>
  $(id).addEventListener('change', () => {
    refreshPlan();
    setStepState('emoDecode', state.meta ? 'done' : 'pending', state.meta ? '已解码' : '等待视频');
  })
);

$('btnRun').onclick = () => startAnalysis();
$('btnRunEmotion').onclick = () => startAnalysis('emotion');
$('btnRunVitals').onclick = () => startAnalysis('vitals');
$('btnStop').onclick = () => {
  if (state.abort) {
    state.abort.abort();
    log('正在停止…');
  }
};
$('btnReset').onclick = () => {
  if (state.running) {
    log('请先停止当前分析', 'error');
    return;
  }
  resetAnalysis();
  if (state.meta) setStepState('emoDecode', 'done', '已解码');
  setProgress(0, state.meta ? '已清空结果，可重新分析' : '等待上传视频', false);
  log('已清空分析结果');
};

function setRunning(on) {
  state.running = on;
  for (const id of ['btnRun', 'btnRunEmotion', 'btnRunVitals']) {
    $(id).disabled = on || !state.file;
  }
  $('btnStop').disabled = !on;
  $('btnReset').disabled = on;
  drop.style.pointerEvents = on ? 'none' : '';
  drop.style.opacity = on ? '0.6' : '';
}

async function startAnalysis(only = '') {
  if (!state.file) return;
  if (state.running) return;

  const doEmotion = only === 'emotion' || (!only && $('chkEmotion').checked);
  const doVitals = only === 'vitals' || (!only && $('chkVitals').checked);
  if (!doEmotion && !doVitals) {
    log('请至少勾选「微表情」或「生命体征」', 'error');
    return;
  }

  resetAnalysis(doEmotion && doVitals ? ['emotion', 'vitals'] : doEmotion ? ['emotion'] : ['vitals']);
  state.plan = { emotion: doEmotion, vitals: doVitals };
  state.abort = new AbortController();
  setRunning(true);
  $('log').textContent = '开始分析…';
  setProgress(0, '准备中…', true);

  const startedAt = performance.now();
  try {
    if (doEmotion) {
      try {
        await runEmotion();
      } catch (err) {
        handleModuleError('emotion', err);
      }
    }
    if (doVitals && !state.abort.signal.aborted) {
      try {
        await runVitals();
      } catch (err) {
        handleModuleError('vitals', err);
      }
    }

    if (state.abort.signal.aborted) {
      setProgress(0, '已停止', false);
    } else {
      setProgress(1, `分析完成（用时 ${((performance.now() - startedAt) / 1000).toFixed(1)}s）`, false);
    }
  } finally {
    setRunning(false);
    state.abort = null;
    state.plan = { emotion: false, vitals: false };
  }
}

function handleModuleError(kind, err) {
  const aborted = err?.name === 'AbortError' || state.abort?.signal.aborted;
  const prefix = kind === 'emotion' ? '[微表情]' : '[生命体征]';
  if (aborted) {
    log(`${prefix} 已停止`);
    if (kind === 'emotion') {
      ['emoDetect', 'emoCrop', 'emoInfer', 'emoTimeline'].forEach((id) =>
        setStepState(id, 'skip', '已停止')
      );
    } else {
      ['vitTrack', 'vitRoi', 'vitWindows', 'vitSummary'].forEach((id) =>
        setStepState(id, 'skip', '已停止')
      );
    }
    return;
  }
  log(`${prefix} 失败：${err.message}`, 'error');
  if (kind === 'emotion') {
    ['emoDetect', 'emoCrop', 'emoInfer', 'emoTimeline'].forEach((id) =>
      setStepState(id, 'error', '失败')
    );
  } else {
    ['vitTrack', 'vitRoi', 'vitWindows', 'vitSummary'].forEach((id) =>
      setStepState(id, 'error', '失败')
    );
  }
  setProgress(0, `${prefix} 失败：${err.message}`, false);
}

function resetAnalysis(kinds = ['emotion', 'vitals']) {
  const doEmotion = kinds.includes('emotion');
  const doVitals = kinds.includes('vitals');

  if (doEmotion) {
    state.emo = null;
    state.emoSelected = 0;
    state.emoSampleByFrame.clear();
    state.emoRenderToken++;

    ['emoDetect', 'emoCrop', 'emoInfer', 'emoTimeline'].forEach((id) =>
      setStepState(id, 'pending', state.meta ? '待运行' : '等待视频')
    );
    if (state.meta) setStepState('emoDecode', 'done', '已解码');

    $('emoDetectRange').max = 0;
    $('emoDetectRange').value = 0;
    $('emoDetectRange').disabled = true;
    $('emoDetectPos').textContent = '—';
    $('emoDetectStats').innerHTML = '';
    clearCanvas($('emoDetectCanvas'));
    $('emoCropGrid').innerHTML =
      '<div class="empty">运行后显示逐帧人脸缩略图（点击查看大图与推理结果）。</div>';
    $('emoCropBig').removeAttribute('src');
    setKV($('emoCropInfo'), ['—', '—', '224 × 224', '—']);
    $('emoInferStats').innerHTML = '';
    $('emoSummary').innerHTML = '';
    $('emoSampleTable').querySelector('tbody').innerHTML = '';
    $('emoEventsTable').querySelector('tbody').innerHTML = '';
    clearCanvas($('emoProbChart'));
    clearCanvas($('emoLineChart'));
    clearCanvas($('emoBand'));

    $('btnEmoJson').disabled = true;
    $('btnEmoCsv').disabled = true;
  }

  if (doVitals) {
    state.vit = null;
    state.vitTraces = null;
    state.liveTraces = null;
    state.vitChannel = 'G';
    state.vitRenderToken++;
    $('vitChannelChips')
      .querySelectorAll('.chip')
      .forEach((b) => b.classList.toggle('active', b.dataset.ch === 'G'));

    ['vitTrack', 'vitRoi', 'vitWindows', 'vitSummary'].forEach((id) =>
      setStepState(id, 'pending', '待运行')
    );
    $('vitTrackRange').max = 0;
    $('vitTrackRange').value = 0;
    $('vitTrackRange').disabled = true;
    $('vitTrackPos').textContent = '—';
    $('vitTrackStats').innerHTML = '';
    clearCanvas($('vitTrackCanvas'));
    $('vitRoiStats').innerHTML = '';
    clearCanvas($('vitRoiChart'));
    $('vitWinTable').querySelector('tbody').innerHTML = '';
    $('vitWinProgress').textContent = '运行后按窗口调用 /physio/estimate，逐步追加结果。';
    $('vitCards').innerHTML = '';
    clearCanvas($('vitSummaryChart'));
    $('vitSummaryNote').textContent =
      '血氧与血压为普通 RGB 摄像头下的估计值，非医疗级；需逐设备、逐人标定。';

    $('btnVitJson').disabled = true;
    $('btnVitCsv').disabled = true;
  }
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.removeAttribute('width');
  canvas.removeAttribute('height');
}

// --------------------------------------------------------------------------- //
// 微表情流水线
// --------------------------------------------------------------------------- //
async function runEmotion() {
  const fps = Number($('emoFps').value);
  const maxSeconds = Number($('maxSec').value);
  const forceEp = $('engineSel').value;
  const signal = state.abort.signal;

  setStepState('emoDetect', 'running', '准备中');
  setStepState('emoCrop', 'pending', '待运行');
  setStepState('emoInfer', 'pending', '待运行');
  setStepState('emoTimeline', 'pending', '待运行');

  const video = await ensureAnalysisVideo('emotion');
  log(`[微表情] 开始：${fps}fps，引擎=${forceEp || '自动'}`);
  const firstSampleAt = { t: 0 };

  let samplesSoFar = 0;
  let lastGridAt = 0;
  const t0 = performance.now();

  const result = await analyzeEmotion(
    video,
    { fps, maxSeconds, forceEp, signal },
    {
      onLog: (msg) => log(`[微表情] ${msg}`),
      onProgress: ({ done, total }) => {
        moduleProgress('emotion', done, total, `微表情：逐帧分析 ${done}/${total}`);
        setStepState('emoDetect', 'running', `${done}/${total}`);
      },
      onFrame: (item, { total, frameCanvas }) => {
        drawFrameWithFace($('emoDetectCanvas'), frameCanvas, item);
        if (item.label) {
          samplesSoFar++;
          if (samplesSoFar === 1) firstSampleAt.t = item.t;
          setStepState('emoCrop', 'running', `${samplesSoFar} 张`);
          setStepState('emoInfer', 'running', `${samplesSoFar} 帧`);
          const now = performance.now();
          if (now - lastGridAt > 300 && $('emoCropGrid').querySelectorAll('img').length < 160) {
            lastGridAt = now;
            appendLiveThumb(item);
          }
        }
      },
    }
  );

  state.emo = result;
  state.emoSampleByFrame = new Map(result.samples.map((s, i) => [s.index, i]));
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

  renderEmotionResult(result);
  setStepState('emoDetect', 'done', `${result.summary.faceFrames}/${result.summary.analyzedFrames} 帧`);
  setStepState('emoCrop', 'done', `${result.samples.length} 张`);
  setStepState('emoInfer', 'done', `${result.summary.avgInferMs.toFixed(0)} ms/帧`);
  setStepState('emoTimeline', 'done', `${result.events.length} 个事件`);
  $('btnEmoJson').disabled = false;
  $('btnEmoCsv').disabled = false;
  log(
    `[微表情] 完成：${result.summary.faceFrames}/${result.summary.analyzedFrames} 帧检出，` +
      `主情绪 ${result.summary.dominantZh}，用时 ${elapsed}s`
  );
}

function appendLiveThumb(item) {
  const grid = $('emoCropGrid');
  if (grid.querySelector('.empty')) grid.innerHTML = '';
  const img = document.createElement('img');
  img.src = item.thumb;
  img.title = `t=${item.t.toFixed(2)}s ${item.zh} ${pct(item.confidence, 0)}`;
  grid.appendChild(img);
  grid.scrollTop = grid.scrollHeight;
}

function drawFrameWithFace(canvas, source, item) {
  const w = source.width;
  const h = source.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, 0, 0, w, h);
  if (item.box) {
    const b = item.box;
    ctx.lineWidth = Math.max(2, w / 400);
    ctx.strokeStyle = '#22c55e';
    ctx.strokeRect(b.originX, b.originY, b.width, b.height);
    const label = item.label
      ? `${item.zh} ${pct(item.confidence, 0)}`
      : `人脸 ${item.detScore != null ? pct(item.detScore, 0) : ''}`;
    drawLabel(ctx, label, b.originX, Math.max(14, b.originY - 4), '#22c55e', w);
  } else if (item.detCount === 0) {
    drawLabel(ctx, '未检测到人脸', 10, 18, '#ef4444', w);
  }
}

function drawLabel(ctx, text, x, y, color, canvasWidth) {
  const fs = Math.max(11, Math.round(canvasWidth / 48));
  ctx.font = `600 ${fs}px system-ui, sans-serif`;
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = 'rgba(15,17,21,0.78)';
  ctx.fillRect(x, y - fs - 2, tw + 8, fs + 6);
  ctx.fillStyle = color;
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(text, x + 4, y);
}

function renderEmotionResult(result) {
  const frames = result.frames;
  const samples = result.samples;
  const total = frames.length;

  // 02 人脸检测
  const range = $('emoDetectRange');
  range.max = Math.max(0, total - 1);
  range.value = Math.max(0, total - 1);
  range.disabled = total === 0;
  $('emoDetectPos').textContent = total
    ? `帧 ${total}/${total} · t=${frames[total - 1].t.toFixed(2)}s`
    : '—';
  $('emoDetectStats').innerHTML = statChips([
    ['采样帧', String(total)],
    ['检出人脸', String(result.summary.faceFrames), result.summary.faceFrames ? 'ok' : 'warn'],
    ['检出率', pct(result.summary.detectionRate)],
    ['平均检测人脸数', fixed(frames.reduce((a, f) => a + f.detCount, 0) / Math.max(1, total), 2)],
    ['检测器', 'BlazeFace (MediaPipe)'],
  ]);

  // 03 裁剪
  renderCropGrid(result);
  if (samples.length) selectEmoSample(Math.min(state.emoSelected, samples.length - 1), { syncDetect: false });
  else {
    $('emoCropGrid').innerHTML = '<div class="empty">整段视频未检出人脸，无法裁剪。</div>';
    setStepState('emoCrop', 'error', '无人脸');
  }

  // 04 推理
  $('emoInferStats').innerHTML = statChips([
    ['推理引擎', result.meta.engine],
    ['推理次数', String(samples.length)],
    ['平均耗时', `${fixed(result.summary.avgInferMs, 0)} ms`],
    ['平均置信度', pct(result.summary.meanConfidence)],
  ]);
  renderEmoSampleTable(result);

  // 05 时间线
  renderEmoTimeline(result);

  // 初始展示最后一帧的人脸检测画面
  if (total) renderEmoFrameAt(total - 1);
}

async function renderEmoFrameAt(index) {
  const result = state.emo;
  if (!result || !state.emoVideo) return;
  const item = result.frames[index];
  if (!item) return;
  const token = ++state.emoRenderToken;
  await seekTo(state.emoVideo, item.t).catch(() => {});
  if (token !== state.emoRenderToken) return;

  const video = state.emoVideo;
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = Math.min(1, ANALYSIS_MAX_SIDE / Math.max(1, vw, vh));
  const w = Math.max(2, Math.round(vw * scale));
  const h = Math.max(2, Math.round(vh * scale));
  const canvas = $('emoDetectCanvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  if (item.box) {
    ctx.lineWidth = Math.max(2, w / 400);
    ctx.strokeStyle = '#22c55e';
    ctx.strokeRect(item.box.originX, item.box.originY, item.box.width, item.box.height);
    if (item.label) {
      drawLabel(
        ctx,
        `${item.zh} ${pct(item.confidence, 0)}`,
        item.box.originX,
        Math.max(14, item.box.originY - 4),
        '#22c55e',
        w
      );
    }
  }
  $('emoDetectPos').textContent = `帧 ${index + 1}/${result.frames.length} · t=${item.t.toFixed(2)}s`;
}

let emoSeekTimer = null;
$('emoDetectRange').addEventListener('input', (e) => {
  const index = Number(e.target.value);
  const item = state.emo?.frames[index];
  if (item) {
    $('emoDetectPos').textContent = `帧 ${index + 1}/${state.emo.frames.length} · t=${item.t.toFixed(2)}s`;
  }
  clearTimeout(emoSeekTimer);
  emoSeekTimer = setTimeout(async () => {
    await renderEmoFrameAt(index);
    const si = state.emoSampleByFrame.get(index);
    if (si != null) selectEmoSample(si, { syncDetect: false });
  }, 45);
});

function renderCropGrid(result) {
  const grid = $('emoCropGrid');
  grid.innerHTML = '';
  const samples = result.samples;
  if (!samples.length) {
    grid.innerHTML = '<div class="empty">整段视频未检出人脸。</div>';
    return;
  }
  const maxThumbs = 120;
  let indexes = samples.map((_, i) => i);
  if (indexes.length > maxThumbs) {
    indexes = decimate(indexes, maxThumbs).map((i) => Math.round(i));
    indexes = [...new Set(indexes)];
  }
  for (const i of indexes) {
    const s = samples[i];
    const img = document.createElement('img');
    img.src = s.thumb;
    img.dataset.i = String(i);
    img.title = `t=${s.t.toFixed(2)}s ${s.zh} ${pct(s.confidence, 0)}`;
    img.addEventListener('click', () => selectEmoSample(i));
    grid.appendChild(img);
  }
}

function selectEmoSample(index, { syncDetect = true } = {}) {
  const result = state.emo;
  if (!result) return;
  const s = result.samples[index];
  if (!s) return;
  state.emoSelected = index;

  $('emoCropGrid').querySelectorAll('img').forEach((img) => {
    img.classList.toggle('active', Number(img.dataset.i) === index);
  });
  $('emoCropBig').src = s.thumb;
  setKV($('emoCropInfo'), [
    `${s.t.toFixed(2)}s (帧 ${s.index + 1})`,
    `${s.box.width.toFixed(0)} × ${s.box.height.toFixed(0)} px`,
    '224 × 224',
    s.faceRGB ? s.faceRGB.map((v) => v.toFixed(0)).join(' / ') : '—',
  ]);

  const items = EMOTION_LABELS.map((label, i) => ({
    label: `${EMOTION_ZH[label]}`,
    value: (s.probs?.[i] || 0) * 100,
    color: EMOTION_COLORS[label],
  }));
  drawBarChart($('emoProbChart'), {
    items,
    maxValue: 100,
    labelWidth: 60,
    format: (v) => `${v.toFixed(1)}%`,
  });

  $('emoInferStats').innerHTML = statChips([
    ['识别结果', `${s.zh}`, 'ok'],
    ['置信度', pct(s.confidence)],
    ['推理耗时', `${fixed(s.timing?.runMs, 1)} ms`],
    ['帧序号', `${s.index + 1}`],
    ['情绪属性', NEGATIVE_LABELS.has(s.label) ? '负向' : s.label === 'Neutral' ? '中性' : '正向'],
  ]);

  // 表格中高亮当前行
  $('emoSampleTable').querySelectorAll('tbody tr').forEach((tr) => {
    tr.style.background = Number(tr.dataset.i) === index ? 'rgba(59,130,246,.14)' : '';
  });

  if (syncDetect) {
    const range = $('emoDetectRange');
    range.value = String(s.index);
    $('emoDetectPos').textContent = `帧 ${s.index + 1}/${result.frames.length} · t=${s.t.toFixed(2)}s`;
    renderEmoFrameAt(s.index);
  }
}

function renderEmoSampleTable(result) {
  const tbody = $('emoSampleTable').querySelector('tbody');
  tbody.innerHTML = '';
  const samples = result.samples;
  const maxRows = 400;
  const idxs =
    samples.length > maxRows
      ? [...new Set(decimate(samples.map((_, i) => i), maxRows).map((i) => Math.round(i)))]
      : samples.map((_, i) => i);
  const html = idxs
    .map((i) => {
      const s = samples[i];
      return `<tr data-i="${i}"><td>${s.t.toFixed(2)}s</td><td>${s.zh}</td>
        <td>${pct(s.confidence)}</td><td>${s.label}</td></tr>`;
    })
    .join('');
  tbody.innerHTML = html || '<tr><td colspan="4" class="muted">暂无推理结果</td></tr>';
  tbody.querySelectorAll('tr[data-i]').forEach((tr) => {
    tr.style.cursor = 'pointer';
    tr.addEventListener('click', () => selectEmoSample(Number(tr.dataset.i)));
  });
}

function renderEmoTimeline(result) {
  const frames = result.frames;
  const lastT = frames.length ? frames[frames.length - 1].t : 1;

  drawBand($('emoBand'), {
    segments: result.timeline.map((s) => ({
      t0: s.t0,
      t1: s.t1,
      color: EMOTION_COLORS[s.label] || '#64748b',
    })),
    xRange: [0, lastT],
    xFormat: (v) => formatTime(v),
  });

  const series = EMOTION_LABELS.map((label, li) => ({
    label: EMOTION_ZH[label],
    color: EMOTION_COLORS[label],
    width: label === 'Neutral' ? 1.8 : 1.3,
    points: frames.filter((f) => f.probs).map((f) => ({ x: f.t, y: f.probs[li] })),
  })).filter((s) => s.points.length);

  drawLineChart($('emoLineChart'), {
    series,
    xRange: [0, lastT],
    yRange: [0, 1],
    baseline: 0.5,
    xFormat: (v) => formatTime(v),
    yFormat: (v) => `${Math.round(v * 100)}%`,
  });

  const s = result.summary;
  const negativePct = pct(s.negativeRatio);
  $('emoSummary').innerHTML = statChips([
    ['主导情绪', s.dominantZh, 'ok'],
    ['检出率', pct(s.detectionRate)],
    ['有效样本', String(s.faceFrames)],
    ['平均置信度', pct(s.meanConfidence)],
    ['负向占比', negativePct, s.negativeRatio > 0.5 ? 'warn' : ''],
    ['情绪切换', String(Math.max(0, result.timeline.length - 1))],
    ['微表情事件', String(result.events.length), result.events.length ? 'warn' : ''],
    ['推理引擎', s.engine],
  ]);

  const tbody = $('emoEventsTable').querySelector('tbody');
  if (!result.events.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="muted">未检出满足条件的短时非平静表情</td></tr>';
  } else {
    tbody.innerHTML = result.events
      .map(
        (e) =>
          `<tr><td>${e.t0.toFixed(2)} – ${e.t1.toFixed(2)}s</td><td>${e.zh}</td>
           <td>${e.duration.toFixed(2)}s</td><td>${pct(e.meanConfidence)}</td></tr>`
      )
      .join('');
  }
}

// --------------------------------------------------------------------------- //
// 生命体征流水线
// --------------------------------------------------------------------------- //
async function runVitals() {
  const fps = Number($('vitFps').value);
  const maxSeconds = Number($('maxSec').value);
  const windowSec = Number($('winSec').value) || 20;
  const stepSec = Number($('stepSec').value) || 5;
  const signal = state.abort.signal;

  setStepState('vitTrack', 'running', '准备中');
  setStepState('vitRoi', 'pending', '待运行');
  setStepState('vitWindows', 'pending', '待运行');
  setStepState('vitSummary', 'pending', '待运行');

  const video = await ensureAnalysisVideo('vitals');
  log(`[生命体征] 开始：${fps}fps，窗口 ${windowSec}s / 步长 ${stepSec}s`);
  const t0 = performance.now();

  let liveSamples = [];
  let lastLiveDraw = 0;

  const result = await analyzeVitals(
    video,
    { fps, maxSeconds, windowSec, stepSec, signal },
    {
      onLog: (msg) => log(`[生命体征] ${msg}`),
      onProgress: ({ phase, done, total }) => {
        if (phase === 'sample') {
          moduleProgress('vitals', done / total * 0.85, 1, `生命体征：ROI 采样 ${done}/${total}`);
        } else {
          moduleProgress(
            'vitals',
            0.85 + (done / Math.max(1, total)) * 0.15,
            1,
            `生命体征：生理估计 ${done}/${total} 个窗口`
          );
          setStepState('vitWindows', done >= total ? 'done' : 'running', `${done}/${total}`);
        }
      },
      onSample: (sample, { total, frameCanvas }) => {
        liveSamples.push(sample);
        const now = performance.now();
        if (now - lastLiveDraw > 350) {
          lastLiveDraw = now;
          state.liveTraces = makeTraces(liveSamples);
          state.vitTraces = state.liveTraces;
          drawVitTrackFromSample(sample, frameCanvas);
          drawVitRoi();
          updateVitLiveStats(liveSamples);
          if (liveSamples.length === 1) setStepState('vitRoi', 'running', '采样中');
        }
      },
      onWindow: (win) => {
        appendVitWindowRow(win);
      },
    }
  );

  state.vit = result;
  state.vitTraces = result.traces;
  state.liveTraces = null;
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

  renderVitalsResult(result);
  const s = result.summary;
  setStepState('vitTrack', 'done', `${result.samples.length} 帧`);
  setStepState('vitRoi', 'done', `${pct(s.faceHitRate, 0)} 有效`);
  setStepState(
    'vitWindows',
    s.readyWindows ? 'done' : 'error',
    `${s.readyWindows}/${result.windows.length} 窗口`
  );
  setStepState(
    'vitSummary',
    s.readyWindows ? 'done' : 'error',
    s.hrMedian != null ? `${s.hrMedian} bpm` : '无结果'
  );
  $('btnVitJson').disabled = false;
  $('btnVitCsv').disabled = result.windows.length === 0;
  log(
    `[生命体征] 完成：${s.readyWindows}/${result.windows.length} 个窗口出结果，` +
      `HR 中位 ${s.hrMedian ?? '—'} bpm，用时 ${elapsed}s`
  );
}

function makeTraces(samples) {
  const picked = samples.map((_, i) => i);
  const idx = picked.length > 1500 ? decimate(picked, 1500) : picked;
  return {
    t: idx.map((i) => samples[i].t),
    forehead: idx.map((i) => samples[i].roi.forehead),
    left_cheek: idx.map((i) => samples[i].roi.left_cheek),
    right_cheek: idx.map((i) => samples[i].roi.right_cheek),
  };
}

function drawVitTrackFromSample(sample, frameCanvas) {
  const canvas = $('vitTrackCanvas');
  const w = frameCanvas.width;
  const h = frameCanvas.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(frameCanvas, 0, 0, w, h);
  if (sample.box) {
    const b = sample.box;
    ctx.lineWidth = Math.max(2, w / 400);
    ctx.strokeStyle = '#22c55e';
    ctx.strokeRect(b.originX, b.originY, b.width, b.height);
    const rects = roiRects(b);
    for (const [name, r] of Object.entries(rects)) {
      ctx.strokeStyle = ROI_COLORS[name] || '#fff';
      ctx.strokeRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0);
      drawLabel(ctx, ROI_ZH[name] || name, r.x0, Math.max(14, r.y0 - 3), ROI_COLORS[name], w);
    }
  }
  $('vitTrackPos').textContent = `t=${sample.t.toFixed(2)}s（帧 ${sample.index + 1}）`;
}

function updateVitLiveStats(samples) {
  const valid = samples.filter((s) => s.valid).length;
  $('vitTrackStats').innerHTML = statChips([
    ['采样帧', String(samples.length)],
    ['人脸命中', pct(samples.filter((s) => s.box).length / Math.max(1, samples.length), 0)],
    ['ROI 有效', pct(valid / Math.max(1, samples.length), 0)],
    ['采样率', `${Number($('vitFps').value)} fps`],
  ]);
}

async function renderVitFrameAt(index) {
  const result = state.vit;
  if (!result || !state.vitVideo) return;
  const sample = result.samples[index];
  if (!sample) return;
  const token = ++state.vitRenderToken;
  await seekTo(state.vitVideo, sample.t).catch(() => {});
  if (token !== state.vitRenderToken) return;

  const video = state.vitVideo;
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = Math.min(1, ANALYSIS_MAX_SIDE / Math.max(1, vw, vh));
  const w = Math.max(2, Math.round(vw * scale));
  const h = Math.max(2, Math.round(vh * scale));
  const canvas = $('vitTrackCanvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);

  if (sample.box) {
    const b = sample.box;
    ctx.lineWidth = Math.max(2, w / 400);
    ctx.strokeStyle = '#22c55e';
    ctx.strokeRect(b.originX, b.originY, b.width, b.height);
    const rects = roiRects(b);
    for (const [name, r] of Object.entries(rects)) {
      ctx.strokeStyle = ROI_COLORS[name] || '#fff';
      ctx.strokeRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0);
      drawLabel(ctx, ROI_ZH[name] || name, r.x0, Math.max(14, r.y0 - 3), ROI_COLORS[name], w);
    }
  }
  $('vitTrackPos').textContent = `t=${sample.t.toFixed(2)}s（帧 ${index + 1}/${result.samples.length}）`;
}

let vitSeekTimer = null;
$('vitTrackRange').addEventListener('input', (e) => {
  const index = Number(e.target.value);
  const sample = state.vit?.samples[index];
  if (sample) $('vitTrackPos').textContent = `t=${sample.t.toFixed(2)}s（帧 ${index + 1}）`;
  clearTimeout(vitSeekTimer);
  vitSeekTimer = setTimeout(() => renderVitFrameAt(index), 45);
});

function drawVitRoi() {
  const traces = state.vitTraces || state.liveTraces;
  if (!traces || !traces.t?.length) return;
  const ch = state.vitChannel;
  let series = [];
  let yFormat;

  if (ch === 'ALL') {
    // 额头 ROI 的原始 RGB
    const channels = [
      ['R', 0, '#ef4444'],
      ['G', 1, '#22c55e'],
      ['B', 2, '#3b82f6'],
    ];
    series = channels.map(([name, ci, color]) => ({
      label: `额头 ${name}`,
      color,
      points: traces.forehead.map((v, i) => ({ x: traces.t[i], y: v ? v[ci] : null })),
    }));
    yFormat = (v) => v.toFixed(0);
  } else {
    const ci = ch === 'R' ? 0 : ch === 'B' ? 2 : 1;
    series = Object.keys(ROI_DEFS).map((name) => {
      const values = traces[name];
      let sum = 0;
      let n = 0;
      for (const v of values) {
        if (v) {
          sum += v[ci];
          n++;
        }
      }
      const mean = n ? sum / n : 0;
      return {
        label: ROI_ZH[name],
        color: ROI_COLORS[name],
        points: values.map((v, i) => ({
          x: traces.t[i],
          y: v && mean ? v[ci] / mean - 1 : null,
        })),
      };
    });
    yFormat = (v) => `${(v * 100).toFixed(2)}%`;
  }

  drawLineChart($('vitRoiChart'), {
    series,
    xFormat: (v) => formatTime(v),
    yFormat,
    yLabel: ch === 'ALL' ? '像素值' : '相对变化',
  });
}

$('vitChannelChips').addEventListener('click', (e) => {
  const btn = e.target.closest('.chip');
  if (!btn) return;
  state.vitChannel = btn.dataset.ch;
  $('vitChannelChips').querySelectorAll('.chip').forEach((b) => b.classList.toggle('active', b === btn));
  drawVitRoi();
});

function appendVitWindowRow(win) {
  const tbody = $('vitWinTable').querySelector('tbody');
  const r = win.result;
  let status;
  let statusClass = '';
  if (win.error) {
    status = `失败：${win.error}`;
    statusClass = 'err';
  } else if (!r) {
    status = '无结果';
    statusClass = 'muted';
  } else if (!r.ready) {
    status = `样本不足 (${r.n_samples || 0})`;
    statusClass = 'muted';
  } else {
    status = '就绪';
    statusClass = 'ok';
  }
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${win.t0.toFixed(1)}–${win.t1.toFixed(1)}s</td>
    <td>${r?.ready ? r.heart_rate : '—'}</td>
    <td>${r?.ready ? r.respiratory_rate : '—'}</td>
    <td>${r?.ready ? r.spo2 : '—'}</td>
    <td>${r?.ready ? r.quality : '—'}</td>
    <td class="${statusClass}">${status}</td>`;
  tr.title = status;
  tbody.appendChild(tr);
  tbody.parentElement.parentElement.scrollTop = tbody.parentElement.parentElement.scrollHeight;
}

function renderVitalsResult(result) {
  const s = result.summary;

  // 01 跟踪
  const range = $('vitTrackRange');
  range.max = Math.max(0, result.samples.length - 1);
  range.value = Math.max(0, result.samples.length - 1);
  range.disabled = result.samples.length === 0;
  $('vitTrackStats').innerHTML = statChips([
    ['采样帧', String(result.meta.analyzedFrames)],
    ['人脸检测次数', String(result.meta.detectCount)],
    ['检测命中', `${result.meta.detectHit}/${result.meta.detectCount}`],
    ['人脸跟踪率', pct(s.faceHitRate, 0)],
    ['采样率', `${result.meta.fps} fps`],
  ]);

  // 02 ROI 曲线
  state.vitTraces = result.traces;
  drawVitRoi();
  $('vitRoiStats').innerHTML = statChips([
    ...Object.keys(ROI_DEFS).map((name) => [ROI_ZH[name], pct(s.roiValid[name], 0)]),
    ['有效帧', String(result.samples.filter((x) => x.valid).length)],
    ['总帧数', String(result.samples.length)],
  ]);

  // 03 表格（onWindow 已逐行写入，这里兜底重绘一次）
  const tbody = $('vitWinTable').querySelector('tbody');
  tbody.innerHTML = '';
  for (const win of result.windows) appendVitWindowRow(win);
  $('vitWinProgress').textContent = result.windows.length
    ? `共 ${result.windows.length} 个窗口，${s.readyWindows} 个成功出结果（窗口 ${result.meta.windowSec.toFixed(1)}s / 步长 ${result.meta.stepSec}s）`
    : `未生成窗口（视频有效时长 ${s.duration.toFixed(1)}s）`;

  // 04 汇总
  const hr = s.hrMedian;
  const bp = s.bloodPressure;
  const hrv = s.hrv;
  $('vitCards').innerHTML = [
    card('心率 HR（中位）', hr ?? '—', 'bpm'),
    card('呼吸率 RR', s.rrMedian ?? '—', '/min'),
    card('血氧 SpO₂', s.spo2Median ?? '—', '%'),
    card('血压 BP', bp ? `${bp.sbp}/${bp.dbp}` : '—', 'mmHg'),
    card('HRV RMSSD', hrv ? hrv.rmssd_ms : '—', 'ms'),
    card('信号质量', s.qualityMedian != null ? pct(s.qualityMedian, 0) : '—', ''),
  ].join('');

  const ready = result.windows.filter((w) => w.result?.ready);
  const series = [
    {
      label: '心率 (bpm)',
      color: '#ef4444',
      width: 1.8,
      points: ready.map((w) => ({
        x: (w.t0 + w.t1) / 2,
        y: w.result.heart_rate,
      })),
    },
    {
      label: '信号质量 ×100',
      color: '#22c55e',
      points: ready.map((w) => ({
        x: (w.t0 + w.t1) / 2,
        y: w.result.quality * 100,
      })),
    },
  ];
  const maxHr = Math.max(120, ...ready.map((w) => w.result.heart_rate * 1.15));
  drawLineChart($('vitSummaryChart'), {
    series,
    yRange: [0, maxHr],
    xFormat: (v) => formatTime(v),
    yFormat: (v) => v.toFixed(0),
  });

  const failed = result.windows.find((w) => w.error);
  const notes = [];
  if (!s.readyWindows) {
    notes.push(
      '未能得到有效生理指标。常见原因：人脸过小/运动过大、光照不足、视频过短，或 Python 生理侧车未启动。'
    );
    if (failed) notes.push(`侧车错误：${failed.error}`);
    else notes.push('可尝试提高采样帧率、延长视频，并保证面部稳定、光线均匀。');
  } else {
    notes.push(
      `有效窗口 ${s.readyWindows}/${result.windows.length}；HR 中位 ${s.hrMedian} bpm，` +
        `最新 ${s.hrLatest} bpm，心率变化 ${fixed(s.heartRateChange, 1)} bpm/min。`
    );
  }
  notes.push('血氧与血压为普通 RGB 摄像头下的估计值，非医疗级；需逐设备、逐人标定。');
  $('vitSummaryNote').innerHTML = notes.map((n) => `<div>${n}</div>`).join('');

  if (result.samples.length) {
    renderVitFrameAt(result.samples.length - 1);
  }
}

// --------------------------------------------------------------------------- //
// 生理侧车状态
// --------------------------------------------------------------------------- //
async function checkPhysio(restartIfDown = false) {
  const el = $('physioStatus');
  el.className = 'pill';
  el.textContent = '侧车检查中…';
  try {
    let h = await fetch('/physio/health').then((r) => r.json());
    if (!h.running && restartIfDown) {
      log('[生命体征] 生理侧车未运行，尝试自动重启…');
      await fetch('/physio/restart', { method: 'POST' }).catch(() => {});
      h = await fetch('/physio/health').then((r) => r.json());
    }
    if (h.running) {
      el.className = 'pill ok';
      el.textContent = `生理侧车运行中 :${h.port ?? '?'}`;
      log(`[生命体征] Python 侧车就绪 (pid=${h.pid ?? '?'} port=${h.port ?? '?'})`);
    } else {
      el.className = 'pill err';
      el.textContent = '生理侧车未运行';
      log(
        `[生命体征] Python 侧车未运行：${h.lastError || '未知原因'}。` +
          '可执行 `pip install -r physio_service/requirements.txt` 后重试。',
        'error'
      );
    }
  } catch (err) {
    el.className = 'pill err';
    el.textContent = '侧车状态未知';
    log(`[生命体征] 无法访问 /physio/health：${err.message}`, 'error');
  }
}
$('btnPhysioCheck').onclick = () => checkPhysio(true);
checkPhysio(true);

// --------------------------------------------------------------------------- //
// 导出
// --------------------------------------------------------------------------- //
function stripThumbs(result) {
  return {
    ...result,
    samples: result.samples.map(({ thumb, ...rest }) => rest),
    frames: result.frames.map(({ thumb, ...rest }) => rest),
  };
}

$('btnEmoJson').onclick = () => {
  if (!state.emo) return;
  const payload = {
    exportedAt: new Date().toISOString(),
    file: { ...state.meta, sizeText: formatBytes(state.meta.size) },
    options: {
      emotionFps: Number($('emoFps').value),
      maxSeconds: Number($('maxSec').value),
      engine: $('engineSel').value || 'auto',
    },
    result: stripThumbs(state.emo),
  };
  downloadText(
    `emotion-${Date.now()}.json`,
    JSON.stringify(payload, null, 2)
  );
  log('已导出微表情结果 JSON');
};

$('btnEmoCsv').onclick = () => {
  const result = state.emo;
  if (!result) return;
  const header = ['t_sec', 'face', 'det_count', 'label', 'zh', 'confidence', ...EMOTION_LABELS.map((l) => `p_${l}`)];
  const lines = [header.join(',')];
  for (const f of result.frames) {
    const row = [
      f.t.toFixed(3),
      f.face ? 1 : 0,
      f.detCount,
      f.label || '',
      f.zh || '',
      f.label ? f.confidence.toFixed(4) : '',
      ...(f.probs ? f.probs.map((p) => p.toFixed(4)) : EMOTION_LABELS.map(() => '')),
    ];
    lines.push(row.join(','));
  }
  downloadText(`emotion-frames-${Date.now()}.csv`, lines.join('\n'), 'text/csv');
  log('已导出逐帧 CSV');
};

$('btnVitJson').onclick = () => {
  if (!state.vit) return;
  const payload = {
    exportedAt: new Date().toISOString(),
    file: { ...state.meta, sizeText: formatBytes(state.meta.size) },
    options: {
      vitalsFps: Number($('vitFps').value),
      maxSeconds: Number($('maxSec').value),
      windowSec: Number($('winSec').value),
      stepSec: Number($('stepSec').value),
    },
    result: {
      meta: state.vit.meta,
      summary: state.vit.summary,
      windows: state.vit.windows,
      traces: state.vit.traces,
    },
  };
  downloadText(`vitals-${Date.now()}.json`, JSON.stringify(payload, null, 2));
  log('已导出生命体征结果 JSON');
};

$('btnVitCsv').onclick = () => {
  const result = state.vit;
  if (!result) return;
  const header = ['t0_sec', 't1_sec', 'n_samples', 'ready', 'heart_rate', 'respiratory_rate', 'spo2', 'sbp', 'dbp', 'hrv_sdnn', 'hrv_rmssd', 'quality', 'error'];
  const lines = [header.join(',')];
  for (const w of result.windows) {
    const r = w.result || {};
    lines.push(
      [
        w.t0.toFixed(2),
        w.t1.toFixed(2),
        w.samples,
        r.ready ? 1 : 0,
        r.heart_rate ?? '',
        r.respiratory_rate ?? '',
        r.spo2 ?? '',
        r.blood_pressure?.sbp ?? '',
        r.blood_pressure?.dbp ?? '',
        r.hrv?.sdnn_ms ?? '',
        r.hrv?.rmssd_ms ?? '',
        r.quality ?? '',
        w.error ? `"${w.error.replace(/"/g, '""')}"` : '',
      ].join(',')
    );
  }
  downloadText(`vitals-windows-${Date.now()}.csv`, lines.join('\n'), 'text/csv');
  log('已导出窗口 CSV');
};

// --------------------------------------------------------------------------- //
// 初始化
// --------------------------------------------------------------------------- //
setRunning(false);
$('chkEmotion').addEventListener('change', () => {
  $('btnRun').disabled = state.running || !state.file || (!$('chkEmotion').checked && !$('chkVitals').checked);
});
$('chkVitals').addEventListener('change', () => {
  $('btnRun').disabled = state.running || !state.file || (!$('chkEmotion').checked && !$('chkVitals').checked);
});
setProgress(0, '等待上传视频', false);
log('就绪：请上传一段包含清晰正脸的视频（建议 10 秒以上、光照均匀）。');
