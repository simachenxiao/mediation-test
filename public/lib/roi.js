// 面部 ROI 像素采样（rPPG 用）——纯函数 + 可复用的临时 canvas。
//
// 与 rppg 的 config.yaml / physio_service/physio/roi.py 对齐：
//   forehead / left_cheek / right_cheek 三个区域按人脸框的固定比例取，
//   并用 YCbCr 肤色阈值剔除头发、背景、口罩等非皮肤像素。

export const ROI_DEFS = {
  forehead: { x0: 0.20, y0: 0.06, x1: 0.80, y1: 0.34 },
  left_cheek: { x0: 0.08, y0: 0.35, x1: 0.42, y1: 0.60 },
  right_cheek: { x0: 0.58, y0: 0.35, x1: 0.92, y1: 0.60 },
};

const SKIN_CB = [77, 127];
const SKIN_CR = [133, 173];
const SKIN_MIN_RATIO = 0.05;

// 人脸框 -> 每个 ROI 的像素矩形（相对于源图），UI 画叠框时也用这个。
export function roiRects(box) {
  const out = {};
  for (const [name, f] of Object.entries(ROI_DEFS)) {
    out[name] = {
      x0: box.originX + f.x0 * box.width,
      y0: box.originY + f.y0 * box.height,
      x1: box.originX + f.x1 * box.width,
      y1: box.originY + f.y1 * box.height,
    };
  }
  return out;
}

let scratch = null;
function getScratch() {
  if (!scratch) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    scratch = { canvas, ctx };
  }
  return scratch;
}

function meanSkin(data, W, H, frac) {
  const x0 = Math.max(0, Math.floor(frac.x0 * W));
  const x1 = Math.min(W - 1, Math.ceil(frac.x1 * W));
  const y0 = Math.max(0, Math.floor(frac.y0 * H));
  const y1 = Math.min(H - 1, Math.ceil(frac.y1 * H));
  let rs = 0;
  let gs = 0;
  let bs = 0;
  let n = 0;
  const total = (x1 - x0 + 1) * (y1 - y0 + 1);
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = (y * W + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const Y = 0.299 * r + 0.587 * g + 0.114 * b;
      const cb = (b - Y) * 0.564 + 128;
      const cr = (r - Y) * 0.713 + 128;
      if (
        cb >= SKIN_CB[0] && cb <= SKIN_CB[1] &&
        cr >= SKIN_CR[0] && cr <= SKIN_CR[1] &&
        Y > 40 && Y < 250
      ) {
        rs += r;
        gs += g;
        bs += b;
        n++;
      }
    }
  }
  if (n < Math.max(4, SKIN_MIN_RATIO * total)) return null;
  return [rs / n, gs / n, bs / n];
}

/**
 * 从整帧画面中裁出人脸，缩放到 128 像素高，再计算三路 ROI 的 RGB 均值。
 * @param {HTMLVideoElement|HTMLCanvasElement} source
 * @param {{originX:number,originY:number,width:number,height:number}} box 源坐标系下的人脸框
 * @returns {{forehead:number[]|null,left_cheek:number[]|null,right_cheek:number[]|null}}
 */
export function sampleRoiMeans(source, box) {
  const empty = { forehead: null, left_cheek: null, right_cheek: null };
  if (!box || !(box.width > 4) || !(box.height > 4)) return empty;
  const sw = source.videoWidth || source.width || 0;
  const sh = source.videoHeight || source.height || 0;
  if (!sw || !sh) return empty;

  const bx = Math.max(0, Math.floor(box.originX));
  const by = Math.max(0, Math.floor(box.originY));
  const bw = Math.min(sw - bx, Math.ceil(box.width));
  const bh = Math.min(sh - by, Math.ceil(box.height));
  if (bw <= 4 || bh <= 4) return empty;

  const { canvas, ctx } = getScratch();
  const H = 128;
  const W = Math.max(24, Math.min(256, Math.round((H * bw) / bh)));
  canvas.width = W;
  canvas.height = H;
  ctx.drawImage(source, bx, by, bw, bh, 0, 0, W, H);
  const { data } = ctx.getImageData(0, 0, W, H);

  const out = {};
  for (const [name, frac] of Object.entries(ROI_DEFS)) {
    out[name] = meanSkin(data, W, H, frac);
  }
  return out;
}
