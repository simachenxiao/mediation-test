// 极简 canvas 图表工具（无第三方依赖）。
// 只实现离线分析页需要的三种图：折线 / 横向条形 / 时间色带。

const FONT = '11px ui-monospace, "JetBrains Mono", Consolas, monospace';
const AXIS = '#3a4252';
const TEXT = '#9aa4b2';
const GRID = 'rgba(148, 163, 184, 0.16)';

function fitCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = Math.max(120, canvas.clientWidth || canvas.parentElement?.clientWidth || 600);
  const cssH = Math.max(80, canvas.clientHeight || 220);
  const w = Math.round(cssW * dpr);
  const h = Math.round(cssH * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  return { ctx, w: cssW, h: cssH };
}

function niceTicks(min, max, count) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return [min];
  }
  const span = max - min;
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const start = Math.ceil(min / step) * step;
  const out = [];
  for (let v = start; v <= max + step * 1e-6; v += step) out.push(v);
  return out;
}

/**
 * 折线图。
 * spec = {
 *   series: [{ label, color, width?, points: [ {x, y|null} | [x, y|null] ] }],
 *   xRange?: [min,max], yRange?: [min,max] | null（自动）,
 *   xTicks?, yTicks?, xFormat?, yFormat?, legend?: boolean,
 *   yMin?: number, yMax?: number, baseline?: number
 * }
 */
export function drawLineChart(canvas, spec) {
  const { ctx, w, h } = fitCanvas(canvas);
  const series = (spec.series || []).filter(Boolean);
  const points = [];
  for (const s of series) {
    for (const p of s.points || []) {
      const x = Array.isArray(p) ? p[0] : p.x;
      const y = Array.isArray(p) ? p[1] : p.y;
      if (Number.isFinite(x) && Number.isFinite(y)) points.push([x, y]);
    }
  }

  let xMin = spec.xRange?.[0];
  let xMax = spec.xRange?.[1];
  if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) {
    xMin = points.length ? Math.min(...points.map((p) => p[0])) : 0;
    xMax = points.length ? Math.max(...points.map((p) => p[0])) : 1;
  }
  if (xMax - xMin < 1e-9) xMax = xMin + 1;

  let yMin = spec.yRange?.[0];
  let yMax = spec.yRange?.[1];
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    yMin = points.length ? Math.min(...points.map((p) => p[1])) : 0;
    yMax = points.length ? Math.max(...points.map((p) => p[1])) : 1;
    if (yMax - yMin < 1e-9) {
      yMin -= 1;
      yMax += 1;
    }
  }

  const showLegend = spec.legend !== false && series.length > 0;
  const pad = {
    left: spec.yLabel ? 48 : 42,
    right: 12,
    top: showLegend ? 34 : 12,
    bottom: 28,
  };
  const plotW = Math.max(20, w - pad.left - pad.right);
  const plotH = Math.max(20, h - pad.top - pad.bottom);

  const sx = (x) => pad.left + ((x - xMin) / (xMax - xMin)) * plotW;
  const sy = (y) => pad.top + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

  ctx.font = FONT;
  ctx.textBaseline = 'middle';

  // 网格 + y 轴刻度
  const yTicks = niceTicks(yMin, yMax, spec.yTicks || 5);
  for (const v of yTicks) {
    if (v < yMin - 1e-9 || v > yMax + 1e-9) continue;
    const y = sy(v);
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillStyle = TEXT;
    ctx.textAlign = 'right';
    const label = spec.yFormat ? spec.yFormat(v) : String(Math.round(v * 10) / 10);
    ctx.fillText(label, pad.left - 6, y);
  }

  // x 轴刻度
  const xTicks = niceTicks(xMin, xMax, spec.xTicks || 6);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const v of xTicks) {
    if (v < xMin - 1e-9 || v > xMax + 1e-9) continue;
    const x = sx(v);
    ctx.strokeStyle = AXIS;
    ctx.beginPath();
    ctx.moveTo(x, pad.top + plotH);
    ctx.lineTo(x, pad.top + plotH + 4);
    ctx.stroke();
    ctx.fillStyle = TEXT;
    ctx.fillText(spec.xFormat ? spec.xFormat(v) : String(Math.round(v)), x, pad.top + plotH + 7);
  }

  // 坐标轴
  ctx.strokeStyle = AXIS;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  if (Number.isFinite(spec.baseline)) {
    const y = sy(spec.baseline);
    if (y >= pad.top && y <= pad.top + plotH) {
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.5)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // 折线
  ctx.save();
  ctx.beginPath();
  ctx.rect(pad.left, pad.top - 2, plotW, plotH + 4);
  ctx.clip();
  for (const s of series) {
    ctx.strokeStyle = s.color || '#3b82f6';
    ctx.lineWidth = s.width || 1.4;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let started = false;
    for (const p of s.points || []) {
      const x = Array.isArray(p) ? p[0] : p.x;
      const y = Array.isArray(p) ? p[1] : p.y;
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        started = false;
        continue;
      }
      const px = sx(x);
      const py = sy(y);
      if (!started) {
        ctx.moveTo(px, py);
        started = true;
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.stroke();
    // 数据点很少时（如少数几个生理窗口）补上圆点，避免折线看不见
    if (s.points.length <= 40) {
      ctx.fillStyle = s.color || '#3b82f6';
      for (const p of s.points || []) {
        const x = Array.isArray(p) ? p[0] : p.x;
        const y = Array.isArray(p) ? p[1] : p.y;
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        ctx.beginPath();
        ctx.arc(sx(x), sy(y), 2.4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  ctx.restore();

  // 图例（自动换行）
  if (showLegend) {
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    let lx = pad.left;
    let ly = 12;
    for (const s of series) {
      const label = s.label || '';
      const tw = ctx.measureText(label).width;
      const need = 12 + tw + 14;
      if (lx + need > w - 6 && lx > pad.left) {
        lx = pad.left;
        ly += 15;
      }
      ctx.fillStyle = s.color || '#3b82f6';
      ctx.fillRect(lx, ly - 3, 8, 6);
      ctx.fillStyle = TEXT;
      ctx.fillText(label, lx + 11, ly);
      lx += need;
    }
  }

  if (spec.yLabel) {
    ctx.save();
    ctx.translate(10, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = TEXT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(spec.yLabel, 0, 0);
    ctx.restore();
  }
}

/** 横向条形图（情绪概率等）。 */
export function drawBarChart(canvas, spec) {
  const { ctx, w, h } = fitCanvas(canvas);
  const items = spec.items || [];
  const pad = { left: spec.labelWidth || 64, right: 46, top: 8, bottom: 8 };
  const rowH = items.length ? (h - pad.top - pad.bottom) / items.length : 0;
  const barH = Math.max(6, Math.min(18, rowH * 0.56));
  const maxValue = Number.isFinite(spec.maxValue)
    ? spec.maxValue
    : Math.max(1e-6, ...items.map((it) => it.value || 0));
  const plotW = Math.max(20, w - pad.left - pad.right);

  ctx.font = FONT;
  ctx.textBaseline = 'middle';
  items.forEach((it, i) => {
    const cy = pad.top + rowH * (i + 0.5);
    ctx.fillStyle = TEXT;
    ctx.textAlign = 'right';
    ctx.fillText(it.label, pad.left - 8, cy);
    const bw = Math.max(0, Math.min(1, (it.value || 0) / maxValue)) * plotW;
    ctx.fillStyle = 'rgba(148, 163, 184, 0.12)';
    ctx.fillRect(pad.left, cy - barH / 2, plotW, barH);
    ctx.fillStyle = it.color || '#3b82f6';
    ctx.fillRect(pad.left, cy - barH / 2, bw, barH);
    ctx.fillStyle = TEXT;
    ctx.textAlign = 'left';
    const text = spec.format ? spec.format(it.value) : String(Math.round((it.value || 0) * 1000) / 10);
    ctx.fillText(text, pad.left + plotW + 6, cy);
  });

  if (!items.length) {
    ctx.fillStyle = TEXT;
    ctx.textAlign = 'center';
    ctx.fillText('暂无数据', w / 2, h / 2);
  }
}

/** 时间色带（用于展示主导情绪随时间的切换）。 */
export function drawBand(canvas, spec) {
  const { ctx, w, h } = fitCanvas(canvas);
  const segments = spec.segments || [];
  let xMin = spec.xRange?.[0];
  let xMax = spec.xRange?.[1];
  if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) {
    xMin = segments.length ? Math.min(...segments.map((s) => s.t0)) : 0;
    xMax = segments.length ? Math.max(...segments.map((s) => s.t1)) : 1;
  }
  if (xMax - xMin < 1e-9) xMax = xMin + 1;

  const pad = { left: spec.padLeft ?? 42, right: 12, top: 4, bottom: 16 };
  const plotW = Math.max(10, w - pad.left - pad.right);
  const bandH = Math.max(8, h - pad.top - pad.bottom - 2);
  const sx = (x) => pad.left + ((x - xMin) / (xMax - xMin)) * plotW;

  ctx.font = FONT;
  const sorted = [...segments].sort((a, b) => a.t0 - b.t0);
  sorted.forEach((s, i) => {
    const next = sorted[i + 1];
    const x0 = sx(s.t0);
    // 连续片段直接接到下一段起点，避免色带之间出现 1px 缝隙；
    // 若中间有明显空档（丢脸），则只画到本段结束。
    const endT = next && next.t0 - s.t1 < 0.4 ? next.t0 : s.t1;
    const x1 = Math.max(x0 + 1.2, sx(endT));
    ctx.fillStyle = s.color || '#64748b';
    ctx.fillRect(x0, pad.top, Math.min(x1, pad.left + plotW) - x0, bandH);
  });

  ctx.strokeStyle = AXIS;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top + bandH);
  ctx.lineTo(pad.left + plotW, pad.top + bandH);
  ctx.stroke();

  ctx.fillStyle = TEXT;
  ctx.textBaseline = 'top';
  ctx.textAlign = 'center';
  for (const v of niceTicks(xMin, xMax, 6)) {
    if (v < xMin - 1e-9 || v > xMax + 1e-9) continue;
    const x = sx(v);
    ctx.fillText(spec.xFormat ? spec.xFormat(v) : String(Math.round(v)), x, pad.top + bandH + 3);
  }
}
