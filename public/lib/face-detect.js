// 人脸检测（MediaPipe BlazeFace）——微表情与生命体征两条流水线共用。
//
// 只负责“在某一帧上找人脸”，不做任何情绪 / 生理计算，
// 因此两个功能模块可以各自独立调用，也可以共用同一个检测器实例。

import { FaceDetector, FilesetResolver } from '/mediapipe/vision_bundle.mjs';

const MODEL_PATH = '/models/blaze_face_short_range.tflite';
const WASM_PATH = '/wasm';

let detectorPromise = null;
let detectorDelegate = '';
// MediaPipe VIDEO 模式要求时间戳单调递增。离线分析会反复从 t=0 开始，
// 因此这里用一个只增不减的内部时钟，与视频真实时间解耦。
let detectClock = 0;

async function createDetector() {
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
  let lastErr = null;
  for (const delegate of ['GPU', 'CPU']) {
    try {
      const detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate },
        runningMode: 'VIDEO',
        minDetectionConfidence: 0.5,
      });
      detectorDelegate = delegate;
      return detector;
    } catch (err) {
      lastErr = err;
      console.warn(`[face] delegate=${delegate} 创建失败，尝试下一个`, err);
    }
  }
  throw lastErr || new Error('MediaPipe 人脸检测器创建失败');
}

/** 获取（并缓存）人脸检测器。 */
export function getFaceDetector() {
  if (!detectorPromise) {
    detectorPromise = createDetector().catch((err) => {
      detectorPromise = null;
      throw err;
    });
  }
  return detectorPromise;
}

export function faceDetectorInfo() {
  return { delegate: detectorDelegate || '—' };
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
