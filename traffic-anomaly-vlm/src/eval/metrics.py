from __future__ import annotations

from typing import Any

import numpy as np

from src.eval.utils import interval_iou


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def roc_auc_score_binary(y_true: list[int] | np.ndarray, y_score: list[float] | np.ndarray) -> float | None:
    y_t = np.asarray(y_true, dtype=np.int32)
    y_s = np.asarray(y_score, dtype=np.float64)
    if y_t.size == 0 or y_t.size != y_s.size:
        return None

    pos = int(np.sum(y_t == 1))
    neg = int(np.sum(y_t == 0))
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(-y_s, kind="mergesort")
    y_sorted = y_t[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    tpr = np.concatenate(([0.0], tp / float(pos)))
    fpr = np.concatenate(([0.0], fp / float(neg)))
    return float(np.trapz(tpr, fpr))


def average_precision_score_binary(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
) -> float | None:
    y_t = np.asarray(y_true, dtype=np.int32)
    y_s = np.asarray(y_score, dtype=np.float64)
    if y_t.size == 0 or y_t.size != y_s.size:
        return None

    pos = int(np.sum(y_t == 1))
    if pos == 0:
        return None

    order = np.argsort(-y_s, kind="mergesort")
    y_sorted = y_t[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(pos)

    ap = 0.0
    prev_recall = 0.0
    for i in range(y_sorted.size):
        if y_sorted[i] != 1:
            continue
        ap += precision[i] * (recall[i] - prev_recall)
        prev_recall = recall[i]
    return float(ap)


def _binary_roc_curve(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    y_t = np.asarray(y_true, dtype=np.int32)
    y_s = np.asarray(y_score, dtype=np.float64)
    if y_t.size == 0 or y_t.size != y_s.size:
        return None

    pos = int(np.sum(y_t == 1))
    neg = int(np.sum(y_t == 0))
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(-y_s, kind="mergesort")
    y_sorted = y_t[order]
    s_sorted = y_s[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    distinct = np.where(np.diff(s_sorted))[0]
    thresh_idx = np.r_[distinct, y_sorted.size - 1]

    tp_t = tp[thresh_idx]
    fp_t = fp[thresh_idx]
    thresholds = s_sorted[thresh_idx]

    tpr = np.r_[0.0, tp_t / float(pos)]
    fpr = np.r_[0.0, fp_t / float(neg)]
    thresholds = np.r_[np.inf, thresholds]
    return fpr, tpr, thresholds


def equal_error_rate_binary(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
) -> dict[str, float] | None:
    curve = _binary_roc_curve(y_true, y_score)
    if curve is None:
        return None

    fpr, tpr, thresholds = curve
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    edr = float(tpr[idx])
    thr = float(thresholds[idx])
    return {
        "eer": eer,
        "edr": edr,
        "threshold_at_eer": thr,
    }


def binary_accuracy_at_threshold(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> float | None:
    y_t = np.asarray(y_true, dtype=np.int32)
    y_s = np.asarray(y_score, dtype=np.float64)
    if y_t.size == 0 or y_t.size != y_s.size:
        return None

    y_pred = (y_s >= float(threshold)).astype(np.int32)
    return float(np.mean(y_pred == y_t))


def compute_gt_span_metrics(
    pred_spans: list[tuple[int, int]],
    gt_intervals: list[list[int]],
) -> dict[str, Any]:
    """Measure how well predicted frame spans cover GT intervals."""

    if not gt_intervals:
        return {
            "num_gt_intervals": 0,
            "hit_ratio": 0.0,
            "full80_ratio": 0.0,
            "mean_best_cover": 0.0,
            "mean_best_iou": 0.0,
            "items": [],
        }

    items: list[dict[str, Any]] = []
    hit = 0
    full80 = 0
    iou_sum = 0.0
    cover_sum = 0.0
    for gt_start, gt_end in gt_intervals:
        gt_start = int(gt_start)
        gt_end = int(gt_end)
        gt_len = max(1, gt_end - gt_start + 1)
        best_cover = 0.0
        best_iou = 0.0
        best_span: tuple[int, int] | None = None
        for pred_start, pred_end in pred_spans:
            pred_start = int(pred_start)
            pred_end = int(pred_end)
            inter_start = max(gt_start, pred_start)
            inter_end = min(gt_end, pred_end)
            inter = max(0, inter_end - inter_start + 1)
            cover = float(inter / gt_len)
            iou = interval_iou((gt_start, gt_end), (pred_start, pred_end))
            if cover > best_cover or (abs(cover - best_cover) < 1e-6 and iou > best_iou):
                best_cover = cover
                best_iou = iou
                best_span = (pred_start, pred_end)

        hit += int(best_cover > 0.0)
        full80 += int(best_cover >= 0.8)
        iou_sum += best_iou
        cover_sum += best_cover
        items.append(
            {
                "start_frame": gt_start,
                "end_frame": gt_end,
                "best_cover": float(best_cover),
                "best_iou": float(best_iou),
                "best_pred_span": list(best_span) if best_span is not None else None,
            }
        )

    total_gt = max(1, len(gt_intervals))
    return {
        "num_gt_intervals": int(len(gt_intervals)),
        "hit_ratio": float(hit / total_gt),
        "full80_ratio": float(full80 / total_gt),
        "mean_best_cover": float(cover_sum / total_gt),
        "mean_best_iou": float(iou_sum / total_gt),
        "items": items,
    }
