// 人脸检测（MediaPipe BlazeFace，short/full 可切换）——微表情与生命体征两条流水线共用。
//
// 只负责“在某一帧上找人脸”，不做任何情绪 / 生理计算，
// 因此两个功能模块可以各自独立调用，也可以共用同一个检测器实例。

import { FaceDetector, FilesetResolver } from '/mediapipe/vision_bundle.mjs';

// full-range 版专为更小/更远的人脸设计；short-range 版在 128×128 输入下
// 人脸低于画面高度约 20% 就会漏检。full 模型放在 test/models，不修改 ../code。
export const FACE_MODELS = {
  full: {
    id: 'full',
    label: 'Full Range（小脸）',
    path: '/local-models/blaze_face_full_range.tflite',
    note: '适合人脸占画面较小、距离较远；误检率略高',
  },
  short: {
    id: 'short',
    label: 'Short Range（原版）',
    path: '/models/blaze_face_short_range.tflite',
    note: '原版 128×128 模型；人脸低于画面高度约 20% 容易漏检',
  },
};
export const DEFAULT_FACE_MODEL = 'full';

const WASM_PATH = '/wasm';

const detectorPromises = new Map();
// MediaPipe VIDEO 模式要求时间戳单调递增。离线分析会反复从 t=0 开始，
// 因此这里用一个只增不减的内部时钟，与视频真实时间解耦。
let detectClock = 0;

async function createDetector(modelId) {
  const model = FACE_MODELS[modelId] || FACE_MODELS[DEFAULT_FACE_MODEL];
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
  let lastErr = null;
  for (const delegate of ['GPU', 'CPU']) {
    try {
      const detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: model.path, delegate },
        runningMode: 'VIDEO',
        minDetectionConfidence: 0.5,
      });
      return detector;
    } catch (err) {
      lastErr = err;
      console.warn(`[face] model=${modelId} delegate=${delegate} 创建失败，尝试下一个`, err);
    }
  }
  throw lastErr || new Error('MediaPipe 人脸检测器创建失败');
}

/** 获取（并缓存）指定模型的人脸检测器。默认 full-range。 */
export function getFaceDetector(modelId = DEFAULT_FACE_MODEL) {
  const id = FACE_MODELS[modelId] ? modelId : DEFAULT_FACE_MODEL;
  if (!detectorPromises.has(id)) {
    const promise = createDetector(id).catch((err) => {
      detectorPromises.delete(id);
      throw err;
    });
    detectorPromises.set(id, promise);
  }
  return detectorPromises.get(id);
}

/** 单帧检测；source 可以是 canvas / video / image。返回 detections 数组。 */
export async function detectFaces(detector, source) {
  detectClock += 33; // 33ms 仅用于时间戳连续性
  const res = await detector.detectForVideo(source, detectClock);
  return res?.detections || [];
}

/** 取面积最大的人脸，返回统一格式的框（像素坐标）与置信度。 */
export function largestFace(detections) {
  if (!detections || !detections.length) return null;
  let best = null;
  let bestArea = -1;
  for (const d of detections) {
    const b = d.boundingBox;
    if (!b) continue;
    const area = (b.width || 0) * (b.height || 0);
    if (area > bestArea) {
      bestArea = area;
      best = {
        box: {
          originX: b.originX,
          originY: b.originY,
          width: b.width,
          height: b.height,
        },
        score: typeof d.categories?.[0]?.score === 'number' ? d.categories[0].score : null,
      };
    }
  }
  return best;
}

/** 框裁剪到画面内，并保证宽高为正。 */
export function clampBox(box, width, height) {
  if (!box) return null;
  const x0 = Math.max(0, Math.min(box.originX, width - 2));
  const y0 = Math.max(0, Math.min(box.originY, height - 2));
  const x1 = Math.max(x0 + 1, Math.min(box.originX + box.width, width));
  const y1 = Math.max(y0 + 1, Math.min(box.originY + box.height, height));
  return { originX: x0, originY: y0, width: x1 - x0, height: y1 - y0 };
}
