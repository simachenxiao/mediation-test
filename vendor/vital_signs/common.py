"""Shared helpers for the webcam vital-sign demo."""

from __future__ import annotations

from typing import Any

import numpy as np


def clean_number(value: Any, digits: int = 3) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return round(number, digits)


def clean_dict(data: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned[key] = clean_dict(value)
        else:
            cleaned[key] = clean_number(value)
    return cleaned


def sample_bvp(bvp: Any, timestamps: Any, limit: int = 360) -> list[dict[str, float]]:
    if bvp is None or timestamps is None or len(bvp) == 0 or len(timestamps) == 0:
        return []
    y = np.asarray(bvp, dtype=float)
    t = np.asarray(timestamps, dtype=float)
    if len(y) != len(t):
        n = min(len(y), len(t))
        y = y[:n]
        t = t[:n]
    if len(y) > limit:
        idx = np.linspace(0, len(y) - 1, limit).astype(int)
        y = y[idx]
        t = t[idx]

    points = []
    for x_value, y_value in zip(t, y):
        if np.isfinite(x_value) and np.isfinite(y_value):
            points.append({"t": round(float(x_value), 3), "v": round(float(y_value), 4)})
    return points
