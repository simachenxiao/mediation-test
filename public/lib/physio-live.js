// 实时 A 引擎：摄像头/视频播放画面 → 额头/脸颊 ROI → /physio/estimate
//
// 与离线 vitals.js 的区别：
//   * 不 seek、不排帧，直接在播放中的 <video> 上逐帧采样；
//   * 固定 30Hz 重采样、30 秒滑动窗口，每 2 秒请求一次结果；
//   * 结果用于实时 A/B 对比，不保存完整逐帧结果。

import { sampleRoiMeans } from './roi.js';

const ROI_NAMES = ['forehead', 'left_cheek', 'right_cheek'];

function interpRgb(points, t) {
  const n = points.length;
  if (!n) return null;
  if (t <= points[0][0]) return points[0][1];
  if (t >= points[n - 1][0]) return points[n - 1][1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid][0] <= t) lo = mid;
    else hi = mid;
  }
  const [ta, va] = points[lo];
  const [tb, vb] = points[hi];
  const w = tb > ta ? (t - ta) / (tb - ta) : 0;
  return [
    va[0] + w * (vb[0] - va[0]),
    va[1] + w * (vb[1] - va[1]),
    va[2] + w * (vb[2] - va[2]),
  ];
}

export class LivePhysioA {
  constructor({
    fs = 30,
    windowSec = 30,
    stepSec = 2,
    minSpanSec = 8,
    endpoint = '/physio/estimate',
  } = {}) {
    this.fs = fs;
    this.windowSec = windowSec;
    this.stepSec = stepSec;
    this.minSpanSec = minSpanSec;
    this.endpoint = endpoint;
    this.samples = [];
    this.lastEstimateAt = 0;
    this.lastResult = null;
    this._busy = false;
  }

  reset() {
    this.samples = [];
    this.lastEstimateAt = 0;
    this.lastResult = null;
    this._busy = false;
  }

  span() {
    if (this.samples.length < 2) return 0;
    return this.samples[this.samples.length - 1].t - this.samples[0].t;
  }

  latestTime() {
    return this.samples.length ? this.samples[this.samples.length - 1].t : 0;
  }

  /** 构造最近 windowSec 的 /physio/estimate payload；A+ 复用同一份采样。 */
  buildPayload(tEnd = this.latestTime()) {
    return this._buildPayload(tEnd);
  }

  /** 从播放中的 video 采一帧；box 为视频像素坐标。 */
  addFrame(video, box, t) {
    if (!box || !(box.width > 4) || !(box.height > 4)) return null;
    const roi = sampleRoiMeans(video, box);
    if (!roi.forehead && !roi.left_cheek && !roi.right_cheek) return null;
    this.samples.push({ t, roi });
    const max = Math.ceil(this.windowSec * this.fs) + this.fs * 3;
    if (this.samples.length > max) this.samples.splice(0, this.samples.length - max);
    return roi;
  }

  async maybeEstimate(force = false) {
    if (this._busy) return null;
    if (this.span() < this.minSpanSec) return null;
    const tNow = this.samples[this.samples.length - 1].t;
    if (!force && tNow - this.lastEstimateAt < this.stepSec) return null;
    this.lastEstimateAt = tNow;
    this._busy = true;
    try {
      const payload = this.buildPayload(tNow);
      const res = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`physio HTTP ${res.status}${text ? ': ' + text.slice(0, 200) : ''}`);
      }
      this.lastResult = await res.json();
      return this.lastResult;
    } finally {
      this._busy = false;
    }
  }

  _buildPayload(tEnd) {
    const t0 = Math.max(this.samples[0].t, tEnd - this.windowSec);
    const n = Math.max(2, Math.floor((tEnd - t0) * this.fs) + 1);
    const timestamps = new Array(n);
    for (let i = 0; i < n; i++) timestamps[i] = t0 + i / this.fs;

    const out = { fs: this.fs, timestamps, forehead: [], left_cheek: [], right_cheek: [] };
    for (const name of ROI_NAMES) {
      const valid = [];
      for (const s of this.samples) {
        const v = s.roi[name];
        if (v) valid.push([s.t, v]);
      }
      const target = out[name];
      if (!valid.length) {
        for (let i = 0; i < n; i++) target.push(null);
      } else {
        for (let i = 0; i < n; i++) target.push(interpRgb(valid, timestamps[i]));
      }
    }
    return out;
  }
}
