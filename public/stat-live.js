// 实时会话的滚动统计：均值 / 中位数 / 偏差。
//
// 与 live.js、camera.js 共用。只保存数值序列，不做信号处理。

function median(values) {
  if (!values.length) return NaN;
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function mean(values) {
  if (!values.length) return NaN;
  let sum = 0;
  for (const v of values) sum += v;
  return sum / values.length;
}

/**
 * 把一个序列（[{x, y}]）汇总成统计量。
 * @param {Array<{x:number,y:number}>} points
 */
export function summarizeSeries(points) {
  const ys = (points || []).map((p) => Number(p.y)).filter((v) => Number.isFinite(v) && v > 0);
  if (!ys.length) return { n: 0, mean: NaN, median: NaN, min: NaN, max: NaN, std: NaN };
  const m = mean(ys);
  const variance = ys.reduce((acc, v) => acc + (v - m) ** 2, 0) / ys.length;
  return {
    n: ys.length,
    mean: m,
    median: median(ys),
    min: Math.min(...ys),
    max: Math.max(...ys),
    std: Math.sqrt(variance),
  };
}

/**
 * 两路序列在同一时间轴上的偏差。用最近邻匹配，容差默认 1 秒。
 * @param {Array<{x:number,y:number}>} a
 * @param {Array<{x:number,y:number}>} b
 * @param {number} tolSec
 */
export function compareSeries(a, b, tolSec = 1.0) {
  const left = (a || []).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  const right = (b || []).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (!left.length || !right.length) return { n: 0, meanAbsDiff: NaN, medianAbsDiff: NaN, maxAbsDiff: NaN, bias: NaN };

  const byTime = [...right].sort((p, q) => p.x - q.x);
  const diffs = [];
  let i = 0;
  for (const p of left) {
    while (i < byTime.length - 1 && Math.abs(byTime[i + 1].x - p.x) <= Math.abs(byTime[i].x - p.x)) i++;
    const q = byTime[i];
    if (Math.abs(q.x - p.x) <= tolSec) diffs.push(p.y - q.y);
  }
  if (!diffs.length) return { n: 0, meanAbsDiff: NaN, medianAbsDiff: NaN, maxAbsDiff: NaN, bias: NaN };
  const abs = diffs.map((d) => Math.abs(d));
  return {
    n: diffs.length,
    meanAbsDiff: mean(abs),
    medianAbsDiff: median(abs),
    maxAbsDiff: Math.max(...abs),
    bias: mean(diffs),
  };
}
