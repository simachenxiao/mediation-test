// 视频文件抽帧工具（离线分析共用）。
//
// 设计目标：不依赖服务端，纯浏览器解码。用法：
//   const { url, video, meta } = await probeVideoFile(file);
//   const grabber = createFrameGrabber(video);
//   const frame = await grabber.grab(1.2);   // { canvas, t, width, height }
//
// 注意：已上传的视频由 <video> 元素解码；seek 是逐帧离线分析里最可靠的取样方式，
// 不受播放帧率/掉帧影响，时间戳与视频真实时间一一对应。

/** 等待元素事件，超时 reject。 */
export function waitEvent(el, name, timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    let timer = null;
    const cleanup = () => {
      el.removeEventListener(name, onEvent);
      el.removeEventListener('error', onError);
      if (timer) clearTimeout(timer);
    };
    const onEvent = (ev) => {
      cleanup();
      resolve(ev);
    };
    const onError = () => {
      cleanup();
      reject(new Error('视频解码失败（文件可能已损坏或浏览器不支持该编码）'));
    };
    el.addEventListener(name, onEvent, { once: true });
    el.addEventListener('error', onError, { once: true });
    if (timeoutMs > 0) {
      timer = setTimeout(() => {
        cleanup();
        reject(new Error(`等待视频事件 ${name} 超时`));
      }, timeoutMs);
    }
  });
}

/** 等待视频至少解出首帧（readyState >= HAVE_CURRENT_DATA）。 */
export async function ensureVideoReady(video) {
  if (video.readyState >= 2) return;
  await waitEvent(video, 'loadeddata', 15000);
}

/**
 * 探测本地视频文件：建立 objectURL、读取元数据。
 * 返回的 video 元素未挂载到 DOM 也能 seek / drawImage（Chromium 支持）。
 */
export async function probeVideoFile(file) {
  if (!file) throw new Error('未选择文件');
  if (file.type && !/^video\//.test(file.type) && !/\.(mp4|webm|mov|mkv|avi|m4v)$/i.test(file.name)) {
    throw new Error(`不是视频文件：${file.name}`);
  }
  const url = URL.createObjectURL(file);
  const video = createVideoElement(url);
  try {
    await waitEvent(video, 'loadedmetadata', 15000);
    // 部分容器（如某些 mov）duration 要等 durationchange 才有值
    for (let i = 0; i < 20 && !Number.isFinite(video.duration); i++) {
      await waitEvent(video, 'durationchange', 500).catch(() => {});
    }
    await ensureVideoReady(video).catch(() => {});
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    const meta = {
      name: file.name,
      size: file.size,
      type: file.type || '',
      duration,
      width: video.videoWidth,
      height: video.videoHeight,
    };
    return { url, video, meta };
  } catch (err) {
    URL.revokeObjectURL(url);
    throw err;
  }
}

/** 创建一个静音、内联、预加载的视频元素（用于离线分析）。 */
export function createVideoElement(url) {
  const v = document.createElement('video');
  v.src = url;
  v.muted = true;
  v.playsInline = true;
  v.preload = 'auto';
  v.crossOrigin = 'anonymous';
  v.setAttribute('playsinline', '');
  return v;
}

/**
 * 规划采样时间点。
 * @param {number} duration 视频时长（秒）
 * @param {number} fps 目标采样帧率
 * @param {{maxSeconds?:number}} [opts]
 */
export function planFrameTimes(duration, fps, { maxSeconds = 0 } = {}) {
  const f = Number(fps) > 0 ? Number(fps) : 10;
  const end = maxSeconds > 0 ? Math.min(duration, maxSeconds) : duration;
  if (!(end > 0)) return [];
  const n = Math.max(1, Math.floor((end - 1e-3) * f) + 1);
  const times = new Array(n);
  for (let i = 0; i < n; i++) {
    // 最后一帧留一点余量，避免 seek 到 duration 边界拿到黑帧
    times[i] = Math.min(i / f, Math.max(0, duration - 0.04));
  }
  return times;
}

/** 精确定位到某一时刻，并等解码器把该帧准备好。 */
export async function seekTo(video, t) {
  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  const target = Math.max(0, Math.min(Number(t) || 0, Math.max(0, duration - 0.001)));
  if (Math.abs(video.currentTime - target) < 0.0008 && video.readyState >= 2) return;
  const p = waitEvent(video, 'seeked', 15000);
  video.currentTime = target;
  await p;
}

/**
 * 帧抓取器：按时间 seek，并把整帧画到一张固定 canvas 上。
 * 默认把长边限制在 maxWidth，兼顾检测精度与速度。
 */
export function createFrameGrabber(video, { maxWidth = 960 } = {}) {
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 360;
  const scale = Math.min(1, maxWidth / Math.max(1, vw, vh));
  const width = Math.max(2, Math.round(vw * scale));
  const height = Math.max(2, Math.round(vh * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  return {
    canvas,
    ctx,
    width,
    height,
    scale,
    async grab(t) {
      await seekTo(video, t);
      ctx.drawImage(video, 0, 0, width, height);
      return { canvas, ctx, t, width, height, scale };
    },
  };
}

/** 线性插值：xs 递增，返回在 x 处的 y；越界取端点。 */
export function interpLinear(xs, ys, x) {
  const n = Math.min(xs.length, ys.length);
  if (!n) return null;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const w = xs[hi] > xs[lo] ? (x - xs[lo]) / (xs[hi] - xs[lo]) : 0;
  return ys[lo] + w * (ys[hi] - ys[lo]);
}

/** 把长序列等间隔抽稀到至多 maxPoints 个点（保留首尾）。 */
export function decimate(list, maxPoints = 1200) {
  if (!Array.isArray(list) || list.length <= maxPoints) return list || [];
  const step = (list.length - 1) / (maxPoints - 1);
  const out = new Array(maxPoints);
  for (let i = 0; i < maxPoints; i++) out[i] = list[Math.round(i * step)];
  return out;
}

export function formatBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

export function formatTime(sec) {
  if (!Number.isFinite(sec) || sec < 0) return '—';
  const s = Math.floor(sec % 60);
  const m = Math.floor((sec / 60) % 60);
  const h = Math.floor(sec / 3600);
  const p = (n) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${p(m)}:${p(s)}` : `${m}:${p(s)}`;
}

/** 统一的取消错误，UI 据此显示“已停止”。 */
export function abortError() {
  const err = new Error('已停止');
  err.name = 'AbortError';
  return err;
}

export function throwIfAborted(signal) {
  if (signal && signal.aborted) throw abortError();
}
