from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.utils import interval_iou, robust_unit_scale
from tmp.run_raw_score_sweep import (
    BoundaryScheme,
    ScoreScheme,
    _dense_scores,
    _detect_spans,
    _load_or_extract_dataset,
    _window_curves,
)


OBJECT_KEYS = [
    "dir_turn_rates_sum",
    "pred_center_sum_err",
    "pred_iou_sum_err",
    "speed_turn_rates_sum",
    "turn_active_ratio",
    "collision_warning",
    "pred_center_max_err",
    "pred_iou_max_err",
    "disappear_far_sum",
]

SCENE_KEYS = [
    "lowres_eventness",
    "track_layout_eventness",
    "clip_eventness",
    "raft_eventness",
]


def _mask_gt(frame_ids: list[int], gt_intervals: list[list[int]]) -> np.ndarray:
    mask = np.zeros((len(frame_ids),), dtype=bool)
    for i, frame_id in enumerate(frame_ids):
        for start, end in gt_intervals:
            if int(start) <= int(frame_id) <= int(end):
                mask[i] = True
                break
    return mask


def _top_mean(values: np.ndarray, ratio: float = 0.2) -> float:
    if values.size == 0:
        return 0.0
    keep = max(1, int(np.ceil(values.size * float(ratio))))
    return float(np.mean(np.sort(values)[-keep:]))


def _gt_lift(frame_ids: list[int], gt_intervals: list[list[int]], dense_scores: np.ndarray) -> float:
    scaled = robust_unit_scale(dense_scores)
    mask = _mask_gt(frame_ids, gt_intervals)
    if mask.any() and (~mask).any():
        return float(_top_mean(scaled[mask], 0.2) - float(np.mean(scaled[~mask])))
    if mask.any():
        return float(_top_mean(scaled[mask], 0.2))
    return 0.0


def _metrics(frame_ids: list[int], gt_intervals: list[list[int]], pred_spans: list[tuple[int, int]], dense_scores: np.ndarray) -> dict[str, float]:
    pred_frames = [(int(frame_ids[s]), int(frame_ids[e])) for s, e in pred_spans if 0 <= s <= e < len(frame_ids)]
    hit = 0
    full80 = 0
    iou_sum = 0.0
    cover_sum = 0.0
    pred_gt_ratio_sum = 0.0

    for gt_start, gt_end in gt_intervals:
        gt_len = max(1, int(gt_end) - int(gt_start) + 1)
        best_iou = 0.0
        best_cover = 0.0
        best_ratio = 999.0
        for pred_start, pred_end in pred_frames:
            inter_start = max(int(gt_start), int(pred_start))
            inter_end = min(int(gt_end), int(pred_end))
            inter = max(0, inter_end - inter_start + 1)
            cover = float(inter / gt_len)
            iou = interval_iou((int(gt_start), int(gt_end)), (int(pred_start), int(pred_end)))
            pred_ratio = float(max(1, int(pred_end) - int(pred_start) + 1) / gt_len)
            if cover > best_cover or (abs(cover - best_cover) < 1e-6 and iou > best_iou):
                best_cover = cover
                best_iou = iou
                best_ratio = pred_ratio
        hit += int(best_cover > 0.0)
        full80 += int(best_cover >= 0.8)
        cover_sum += best_cover
        iou_sum += best_iou
        pred_gt_ratio_sum += best_ratio

    total_gt = max(1, len(gt_intervals))
    total_pred_frames = sum(max(0, end - start + 1) for start, end in pred_frames)
    return {
        "mean_best_cover": float(cover_sum / total_gt),
        "mean_best_iou": float(iou_sum / total_gt),
        "hit_ratio": float(hit / total_gt),
        "full80_ratio": float(full80 / total_gt),
        "avg_span_ratio": float(total_pred_frames / max(1, len(frame_ids))),
        "mean_best_pred_to_gt_ratio": float(pred_gt_ratio_sum / total_gt),
        "num_pred_spans": float(len(pred_frames)),
        "gt_lift": float(_gt_lift(frame_ids, gt_intervals, dense_scores)),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = list(rows[0].keys()) if rows else []
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _rank(metrics: dict[str, float]) -> float:
    width_penalty = min(1.0, max(0.0, metrics["avg_span_ratio"] - 0.80) / 0.20)
    excess_penalty = min(1.0, max(0.0, metrics["mean_best_pred_to_gt_ratio"] - 1.8) / 3.0)
    count_penalty = min(1.0, max(0.0, metrics["num_pred_spans"] - 1.5) / 2.0)
    return float(
        0.35 * metrics["mean_best_cover"]
        + 0.22 * metrics["mean_best_iou"]
        + 0.16 * metrics["full80_ratio"]
        + 0.12 * metrics["gt_lift"]
        + 0.10 * metrics["hit_ratio"]
        - 0.10 * width_penalty
        - 0.07 * excess_penalty
        - 0.04 * count_penalty
    )


def _boundary_library() -> list[BoundaryScheme]:
    return [
        BoundaryScheme("cov_wide", "by_peeks", 1.35, 0.25, 4, (14, 24), 6, 12, False),
        BoundaryScheme("cov_mid", "by_peeks", 1.15, 0.25, 4, (12, 20), 6, 12, False),
        BoundaryScheme("tight_mid", "by_peeks", 1.35, 0.25, 4, (10, 16), 6, 12, False),
        BoundaryScheme("tight_small", "by_peeks", 1.35, 0.25, 4, (8, 12), 6, 8, False),
    ]


def _scheme_library() -> list[ScoreScheme]:
    object_base = {
        "dir_turn_rates_sum": 0.5,
        "pred_center_sum_err": 0.5,
        "pred_iou_sum_err": 0.5,
        "speed_turn_rates_sum": 0.5,
        "turn_active_ratio": 0.5,
        "collision_warning": 0.0,
        "pred_center_max_err": 0.0,
        "pred_iou_max_err": 0.0,
        "disappear_far_sum": 0.10,
    }
    return [
        ScoreScheme("object_only", object_base, {}, {"object_score": 1.0, "scene_score": 0.0}, True, 5),
        ScoreScheme("object_lowres", object_base, {"lowres_eventness": 1.0}, {"object_score": 0.8, "scene_score": 0.2}, True, 5),
        ScoreScheme("object_layout", object_base, {"track_layout_eventness": 1.0}, {"object_score": 0.8, "scene_score": 0.2}, True, 5),
        ScoreScheme("object_clip", object_base, {"clip_eventness": 1.0}, {"object_score": 0.8, "scene_score": 0.2}, True, 5),
        ScoreScheme("object_raft", object_base, {"raft_eventness": 1.0}, {"object_score": 0.8, "scene_score": 0.2}, True, 5),
        ScoreScheme("object_lowres_layout", object_base, {"lowres_eventness": 0.5, "track_layout_eventness": 0.5}, {"object_score": 0.75, "scene_score": 0.25}, True, 5),
        ScoreScheme("object_lowres_layout_clip", object_base, {"lowres_eventness": 0.35, "track_layout_eventness": 0.35, "clip_eventness": 0.30}, {"object_score": 0.75, "scene_score": 0.25}, True, 5),
        ScoreScheme("object_lowres_layout_raft", object_base, {"lowres_eventness": 0.35, "track_layout_eventness": 0.35, "raft_eventness": 0.30}, {"object_score": 0.75, "scene_score": 0.25}, True, 5),
        ScoreScheme("object_all_scene", object_base, {"lowres_eventness": 0.35, "track_layout_eventness": 0.25, "clip_eventness": 0.20, "raft_eventness": 0.20}, {"object_score": 0.75, "scene_score": 0.25}, True, 5),
    ]


def _available_feature_mask(dataset: dict[str, dict[str, Any]]) -> dict[str, bool]:
    available = {key: False for key in SCENE_KEYS}
    for video in dataset.values():
        for row in video["window_rows"]:
            feat = row["feature_dict"]
            for key in SCENE_KEYS:
                if abs(float(feat.get(key, 0.0))) > 1e-8:
                    available[key] = True
    return available


def _filter_schemes_by_availability(schemes: list[ScoreScheme], available: dict[str, bool]) -> list[ScoreScheme]:
    filtered: list[ScoreScheme] = []
    for scheme in schemes:
        needed = [key for key, weight in scheme.scene_weights.items() if float(weight) > 0.0]
        if any(key in available and not available[key] for key in needed):
            continue
        filtered.append(scheme)
    return filtered


def _evaluate_scheme(
    dataset: dict[str, dict[str, Any]],
    scheme: ScoreScheme,
    boundaries: list[BoundaryScheme],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for boundary in boundaries:
        metrics_per_video: list[dict[str, float]] = []
        for video_id, video in dataset.items():
            rows = [row["feature_dict"] for row in video["window_rows"]]
            curves = _window_curves(rows, scheme)
            dense_trigger = _dense_scores(curves["trigger_score"], len(video["frame_ids"]), video["window_size"])
            pred_spans = _detect_spans(dense_trigger, boundary)
            metrics_per_video.append(_metrics(video["frame_ids"], video["gt_intervals"], pred_spans, dense_trigger))
        aggregate = _aggregate(metrics_per_video)
        aggregate["rank"] = _rank(aggregate)
        candidate = {
            "scheme": scheme.name,
            "boundary": boundary.name,
            **aggregate,
        }
        if best is None or candidate["rank"] > best["rank"]:
            best = candidate
    assert best is not None
    return best


def _evaluate_feature(
    dataset: dict[str, dict[str, Any]],
    feature_key: str,
    boundaries: list[BoundaryScheme],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for boundary in boundaries:
        metrics_per_video: list[dict[str, float]] = []
        for video in dataset.values():
            rows = [row["feature_dict"] for row in video["window_rows"]]
            window_scores = robust_unit_scale(np.asarray([float(row.get(feature_key, 0.0)) for row in rows], dtype=np.float32))
            dense_scores = _dense_scores(window_scores, len(video["frame_ids"]), video["window_size"])
            pred_spans = _detect_spans(dense_scores, boundary)
            metrics_per_video.append(_metrics(video["frame_ids"], video["gt_intervals"], pred_spans, dense_scores))
        aggregate = _aggregate(metrics_per_video)
        aggregate["rank"] = _rank(aggregate)
        candidate = {
            "feature": feature_key,
            "boundary": boundary.name,
            **aggregate,
        }
        if best is None or candidate["rank"] > best["rank"]:
            best = candidate
    assert best is not None
    return best


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observe per-component effect before larger raw-score search")
    parser.add_argument("--config", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/configs/default.yaml")
    parser.add_argument("--frames-root", default="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal")
    parser.add_argument("--manifest", default="/nvme2/VAD_yemao/data/TU-DAT-25/abnormal.txt")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--precompute-clip", action="store_true")
    parser.add_argument("--precompute-raft", action="store_true")
    parser.add_argument("--video-ids", nargs="*", default=None)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    dataset = _load_or_extract_dataset(args, Path(args.cache_dir))
    if args.video_ids:
        keep = set(args.video_ids)
        dataset = {k: v for k, v in dataset.items() if k in keep}

    available = _available_feature_mask(dataset)
    boundaries = _boundary_library()
    schemes = _filter_schemes_by_availability(_scheme_library(), available)

    feature_results = []
    for feature_key in OBJECT_KEYS + SCENE_KEYS:
        if feature_key in available and not available[feature_key]:
            continue
        feature_results.append(_evaluate_feature(dataset, feature_key, boundaries))
    feature_results.sort(key=lambda item: item["rank"], reverse=True)

    scheme_results = [_evaluate_scheme(dataset, scheme, boundaries) for scheme in schemes]
    scheme_results.sort(key=lambda item: item["rank"], reverse=True)

    payload = {
        "num_videos": int(len(dataset)),
        "available_scene_features": available,
        "feature_results": feature_results,
        "scheme_results": scheme_results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(build_args())
