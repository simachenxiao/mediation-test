"""Facial ROI extraction and skin segmentation.

The rPPG signal is strongest on well-perfused, non-moving skin.  We use the
forehead as the primary ROI and the two cheeks as secondary ROIs (also used to
compute a PTT surrogate for blood pressure).  A cheap YCbCr skin model rejects
hair, background and mask pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in pixel coordinates (inclusive bounds)."""
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0 + 1)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0 + 1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def clipped(self, w: int, h: int) -> "Rect":
        return Rect(
            max(0, self.x0), max(0, self.y0),
            min(w - 1, self.x1), min(h - 1, self.y1),
        )


def face_to_rects(box: tuple[float, float, float, float],
                  roi_cfg: dict) -> dict[str, Rect]:
    """Convert a face box (x0, y0, x1, y1 in pixels) into named ROI rects.

    ``roi_cfg`` maps ROI name -> {x0, y0, x1, y1} as fractions of the face box.
    """
    fx0, fy0, fx1, fy1 = box
    fw = max(1.0, fx1 - fx0)
    fh = max(1.0, fy1 - fy0)
    out: dict[str, Rect] = {}
    for name, frac in roi_cfg.items():
        if not isinstance(frac, dict):
            continue
        r = Rect(
            int(round(fx0 + frac.get("x0", 0.0) * fw)),
            int(round(fy0 + frac.get("y0", 0.0) * fh)),
            int(round(fx0 + frac.get("x1", 1.0) * fw)),
            int(round(fy0 + frac.get("y1", 1.0) * fh)),
        )
        out[name] = r
    return out


def skin_mask(bgr: np.ndarray, cb_range=(77, 127), cr_range=(133, 173)) -> np.ndarray:
    """Return a boolean skin mask for a BGR image using YCbCr thresholds."""
    if bgr.size == 0:
        return np.zeros((0, 0), dtype=bool)
    ycrcb = _bgr_to_ycrcb(bgr)
    cb = ycrcb[:, :, 1]
    cr = ycrcb[:, :, 2]
    mask = (
        (cb >= cb_range[0]) & (cb <= cb_range[1])
        & (cr >= cr_range[0]) & (cr <= cr_range[1])
    )
    # also reject very dark (no signal) and over-saturated pixels
    y = ycrcb[:, :, 0]
    mask &= (y > 40) & (y < 250)
    return mask


def _bgr_to_ycrcb(bgr: np.ndarray) -> np.ndarray:
    """Small integer YCbCr conversion to avoid an OpenCV dependency here."""
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 128.0
    cr = (r - y) * 0.713 + 128.0
    return np.stack([y, cb, cr], axis=-1)


def roi_mean_rgb(bgr: np.ndarray, rect: Rect, cb_range=(77, 127),
                 cr_range=(133, 173), min_skin_ratio: float = 0.05):
    """Return the mean R, G, B of skin pixels inside ``rect``.

    Returns ``None`` when the ROI contains too few skin pixels.
    """
    h, w = bgr.shape[:2]
    r = rect.clipped(w, h)
    if r.area <= 0:
        return None
    crop = bgr[r.y0:r.y1 + 1, r.x0:r.x1 + 1]
    mask = skin_mask(crop, cb_range, cr_range)
    if mask.size == 0 or mask.sum() < max(4, min_skin_ratio * mask.size):
        return None
    px = crop[mask].astype(np.float64)
    mean_bgr = px.mean(axis=0)  # B, G, R
    return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)  # R, G, B


def extract_roi_means(bgr: np.ndarray, box: tuple[float, float, float, float],
                      roi_cfg: dict) -> dict[str, np.ndarray | None]:
    """Extract (R, G, B) means for every configured ROI of a face box."""
    cb = tuple(roi_cfg.get("skin_cb", [77, 127]))
    cr = tuple(roi_cfg.get("skin_cr", [133, 173]))
    rects = face_to_rects(box, roi_cfg)
    out: dict[str, np.ndarray | None] = {}
    for name, rect in rects.items():
        out[name] = roi_mean_rgb(bgr, rect, cb, cr)
    return out
