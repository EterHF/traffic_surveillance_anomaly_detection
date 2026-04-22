from __future__ import annotations

import numpy as np


def clip01(value: float) -> float:
    return float(min(max(float(value), 0.0), 1.0))


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    start = max(int(a[0]), int(b[0]))
    end = min(int(a[1]), int(b[1]))
    inter = max(0, end - start + 1)
    union = max(1, (int(a[1]) - int(a[0]) + 1) + (int(b[1]) - int(b[0]) + 1) - inter)
    return float(inter / union)


def top_mean(values: np.ndarray, ratio: float = 0.25) -> float:
    if values.size == 0:
        return 0.0
    keep = max(1, int(np.ceil(values.size * float(ratio))))
    return float(np.mean(np.sort(values)[-keep:]))


def peak_idx_for_span(scores: np.ndarray, start_idx: int, end_idx: int) -> int:
    local = scores[start_idx : end_idx + 1]
    if local.size == 0:
        return int(start_idx)
    return int(start_idx + int(np.argmax(local)))


def node_signal_summary(values: np.ndarray, start_idx: int, end_idx: int) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    lo = max(0, int(start_idx))
    hi = min(values.size - 1, int(end_idx))
    if hi < lo:
        return 0.0, 0.0
    seg = values[lo : hi + 1]
    if seg.size == 0:
        return 0.0, 0.0
    return float(np.max(seg)), top_mean(seg, ratio=0.25)
