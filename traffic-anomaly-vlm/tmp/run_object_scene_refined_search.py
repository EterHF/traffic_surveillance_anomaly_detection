from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
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
    _plot_video,
    _window_curves,
)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    valid = {key: float(value) for key, value in weights.items() if float(value) > 0.0}
    total = sum(valid.values())
    if total <= 0.0:
        return {}
    return {key: float(value / total) for key, value in valid.items()}


def _object_profiles() -> dict[str, dict[str, float]]:
    return {
        "src_like": _normalize_weights(
            {
                "dir_turn_rates_sum": 1.0,
                "pred_center_sum_err": 1.0,
                "pred_iou_sum_err": 1.0,
                "speed_turn_rates_sum": 1.0,
                "turn_active_ratio": 1.0,
                "disappear_far_sum": 0.20,
            }
        ),
        "turn_focus": _normalize_weights(
            {
                "dir_turn_rates_sum": 1.25,
                "speed_turn_rates_sum": 1.15,
                "turn_active_ratio": 1.10,
                "pred_center_sum_err": 0.85,
                "pred_iou_sum_err": 0.85,
                "disappear_far_sum": 0.10,
            }
        ),
        "residual_focus": _normalize_weights(
            {
                "dir_turn_rates_sum": 0.90,
                "speed_turn_rates_sum": 0.90,
                "turn_active_ratio": 0.85,
                "pred_center_sum_err": 1.20,
                "pred_iou_sum_err": 1.15,
                "disappear_far_sum": 0.10,
            }
        ),
        "compact_mixed": _normalize_weights(
            {
                "dir_turn_rates_sum": 1.10,
                "speed_turn_rates_sum": 1.00,
                "turn_active_ratio": 1.10,
                "pred_center_sum_err": 0.85,
                "pred_iou_sum_err": 0.85,
                "collision_warning": 0.20,
                "disappear_far_sum": 0.25,
            }
        ),
    }


def _scene_profiles() -> dict[str, dict[str, float]]:
    return {
        "none": {},
        "layout_only": _normalize_weights({"track_layout_eventness": 1.0}),
        "lowres_only": _normalize_weights({"lowres_eventness": 1.0}),
        "layout_dominant": _normalize_weights({"track_layout_eventness": 0.65, "lowres_eventness": 0.35}),
        "balanced": _normalize_weights({"track_layout_eventness": 0.50, "lowres_eventness": 0.50}),
        "lowres_dominant": _normalize_weights({"track_layout_eventness": 0.35, "lowres_eventness": 0.65}),
    }


def _scheme_library() -> list[ScoreScheme]:
    schemes: list[ScoreScheme] = []
    for object_name, object_weights in _object_profiles().items():
        for scene_name, scene_weights in _scene_profiles().items():
            smooth_windows = [3, 5]
            if scene_name == "none":
                branch_scene_shares = [0.0]
            else:
                branch_scene_shares = [0.15, 0.25]
            for scene_share in branch_scene_shares:
                for smooth_window in smooth_windows:
                    object_share = 1.0 - scene_share
                    schemes.append(
                        ScoreScheme(
                            name=f"{object_name}__{scene_name}__scene{scene_share:.2f}__smooth{smooth_window}",
                            object_weights=object_weights,
                            scene_weights=scene_weights,
                            branch_weights={
                                "object_score": object_share,
                                "scene_score": scene_share,
                            },
                            normalize_per_feature=True,
                            final_smooth_window=smooth_window,
                        )
                    )
    return schemes


def _boundary_library() -> list[BoundaryScheme]:
    schemes: list[BoundaryScheme] = []
    for high_z in [1.15, 1.35]:
        for peak_expand in [(10, 16), (12, 20), (14, 24)]:
            for merge_gap in [8, 12]:
                for min_span_len in [6]:
                    schemes.append(
                        BoundaryScheme(
                            name=(
                                f"peeks_hz{high_z}"
                                f"_exp{peak_expand[0]}_{peak_expand[1]}"
                                f"_mg{merge_gap}_min{min_span_len}"
                            ),
                            method="by_peeks",
                            high_z=float(high_z),
                            low_z=0.25,
                            peak_gap=4,
                            peak_expand=tuple(peak_expand),
                            min_span_len=int(min_span_len),
                            merge_gap=int(merge_gap),
                            use_savgol_filter=False,
                        )
                    )
    return schemes


def _top_mean(values: np.ndarray, ratio: float = 0.2) -> float:
    if values.size == 0:
        return 0.0
    keep = max(1, int(np.ceil(values.size * float(ratio))))
    return float(np.mean(np.sort(values)[-keep:]))


def _mask_gt(frame_ids: list[int], gt_intervals: list[list[int]]) -> np.ndarray:
    mask = np.zeros((len(frame_ids),), dtype=bool)
    for idx, frame_id in enumerate(frame_ids):
        for start, end in gt_intervals:
            if int(start) <= int(frame_id) <= int(end):
                mask[idx] = True
                break
    return mask


def _video_metrics(
    frame_ids: list[int],
    gt_intervals: list[list[int]],
    pred_spans_idx: list[tuple[int, int]],
    dense_trigger: np.ndarray,
) -> dict[str, float]:
    pred_frames = [(int(frame_ids[s]), int(frame_ids[e])) for s, e in pred_spans_idx if 0 <= s <= e < len(frame_ids)]
    total_pred_frames = sum(max(0, end - start + 1) for start, end in pred_frames)
    total_frames = max(1, len(frame_ids))

    scaled = robust_unit_scale(dense_trigger)
    gt_mask = _mask_gt(frame_ids, gt_intervals)
    if gt_mask.any() and (~gt_mask).any():
        gt_lift = float(_top_mean(scaled[gt_mask], 0.2) - float(np.mean(scaled[~gt_mask])))
    elif gt_mask.any():
        gt_lift = float(_top_mean(scaled[gt_mask], 0.2))
    else:
        gt_lift = 0.0

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
        iou_sum += best_iou
        cover_sum += best_cover
        pred_gt_ratio_sum += best_ratio

    total_gt = max(1, len(gt_intervals))
    avg_span_ratio = float(total_pred_frames / total_frames)
    num_pred_spans = float(len(pred_frames))
    wide_single_flag = float(num_pred_spans <= 1.05 and avg_span_ratio >= 0.90)
    return {
        "hit_ratio": float(hit / total_gt),
        "full80_ratio": float(full80 / total_gt),
        "mean_best_cover": float(cover_sum / total_gt),
        "mean_best_iou": float(iou_sum / total_gt),
        "mean_best_pred_to_gt_ratio": float(pred_gt_ratio_sum / total_gt),
        "avg_span_ratio": avg_span_ratio,
        "num_pred_spans": num_pred_spans,
        "gt_lift": gt_lift,
        "wide_single_flag": wide_single_flag,
    }


def _aggregate(video_metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = list(video_metrics[0].keys()) if video_metrics else []
    return {key: float(np.mean([row[key] for row in video_metrics])) for key in keys}


def _rank(agg: dict[str, float]) -> float:
    width_penalty = min(1.0, max(0.0, agg["avg_span_ratio"] - 0.84) / 0.18)
    excess_penalty = min(1.0, max(0.0, agg["mean_best_pred_to_gt_ratio"] - 1.8) / 2.5)
    fragment_penalty = min(1.0, max(0.0, agg["num_pred_spans"] - 2.0) / 2.0)
    return float(
        0.40 * agg["mean_best_cover"]
        + 0.18 * agg["full80_ratio"]
        + 0.16 * agg["mean_best_iou"]
        + 0.10 * agg["hit_ratio"]
        + 0.08 * agg["gt_lift"]
        - 0.08 * width_penalty
        - 0.05 * excess_penalty
        - 0.03 * fragment_penalty
        - 0.06 * agg["wide_single_flag"]
    )


def _evaluate_combo(
    dataset: dict[str, dict[str, Any]],
    scheme: ScoreScheme,
    boundary: BoundaryScheme,
) -> dict[str, Any]:
    metrics_per_video: dict[str, dict[str, Any]] = {}
    aggregate_rows: list[dict[str, float]] = []
    for video_id, video_data in dataset.items():
        rows = [row["feature_dict"] for row in video_data["window_rows"]]
        curves = _window_curves(rows, scheme)
        dense_object = _dense_scores(curves["object_score"], len(video_data["frame_ids"]), video_data["window_size"])
        dense_scene = _dense_scores(curves["scene_score"], len(video_data["frame_ids"]), video_data["window_size"])
        dense_trigger = _dense_scores(curves["trigger_score"], len(video_data["frame_ids"]), video_data["window_size"])
        pred_spans = _detect_spans(dense_trigger, boundary)
        metrics = _video_metrics(video_data["frame_ids"], video_data["gt_intervals"], pred_spans, dense_trigger)
        metrics["pred_spans"] = [[int(video_data["frame_ids"][s]), int(video_data["frame_ids"][e])] for s, e in pred_spans]
        metrics["curves"] = {
            "dense_object": dense_object,
            "dense_scene": dense_scene,
            "dense_trigger": dense_trigger,
        }
        metrics_per_video[video_id] = metrics
        aggregate_rows.append({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})

    aggregate = _aggregate(aggregate_rows)
    aggregate["rank"] = _rank(aggregate)
    aggregate["scheme"] = scheme.name
    aggregate["boundary"] = boundary.name
    return {"aggregate": aggregate, "video_metrics": metrics_per_video}


def _write_outputs(
    out_dir: Path,
    leaderboard: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    dataset: dict[str, dict[str, Any]],
    scheme_map: dict[str, ScoreScheme],
    boundary_map: dict[str, BoundaryScheme],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "leaderboard.json").open("w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)
    with (out_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(leaderboard[0].keys()))
        writer.writeheader()
        writer.writerows(leaderboard)

    for tag, result in selected.items():
        aggregate = result["aggregate"]
        tag_dir = out_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        with (tag_dir / "best_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "scheme": asdict(scheme_map[aggregate["scheme"]]),
                    "boundary": asdict(boundary_map[aggregate["boundary"]]),
                    "aggregate": aggregate,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        for video_id, metrics in result["video_metrics"].items():
            video_dir = tag_dir / "videos" / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            index_spans = []
            for start_frame, end_frame in metrics["pred_spans"]:
                try:
                    start_idx = dataset[video_id]["frame_ids"].index(start_frame)
                    end_idx = dataset[video_id]["frame_ids"].index(end_frame)
                except ValueError:
                    continue
                index_spans.append((start_idx, end_idx))
            curves = metrics["curves"]
            _plot_video(
                frame_ids=dataset[video_id]["frame_ids"],
                curves=curves,
                pred_spans=index_spans,
                gt_intervals=dataset[video_id]["gt_intervals"],
                title=f"{video_id} | {aggregate['scheme']} | {aggregate['boundary']}",
                save_path=video_dir / "scores.png",
            )
            with (video_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump({k: v for k, v in metrics.items() if k != "curves"}, f, ensure_ascii=False, indent=2)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused search over object + lowres + layout raw-score presets")
    parser.add_argument("--config", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/configs/default.yaml")
    parser.add_argument("--frames-root", default="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal")
    parser.add_argument("--manifest", default="/nvme2/VAD_yemao/data/TU-DAT-25/abnormal.txt")
    parser.add_argument("--cache-dir", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/tmp/raw_score_search/cache")
    parser.add_argument("--out-dir", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/tmp/raw_score_search/object_scene_refined")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    dataset = _load_or_extract_dataset(args, Path(args.cache_dir))
    schemes = _scheme_library()
    boundaries = _boundary_library()
    scheme_map = {scheme.name: scheme for scheme in schemes}
    boundary_map = {boundary.name: boundary for boundary in boundaries}

    detailed_results: list[dict[str, Any]] = []
    leaderboard: list[dict[str, Any]] = []
    for scheme in schemes:
        for boundary in boundaries:
            result = _evaluate_combo(dataset, scheme, boundary)
            detailed_results.append(result)
            leaderboard.append(result["aggregate"])

    leaderboard.sort(
        key=lambda item: (
            float(item["rank"]),
            float(item["mean_best_cover"]),
            float(item["mean_best_iou"]),
            -float(item["avg_span_ratio"]),
        ),
        reverse=True,
    )

    best_overall = leaderboard[0]
    compact_rows = [
        row
        for row in leaderboard
        if row["mean_best_cover"] >= 0.90
        and row["full80_ratio"] >= 0.80
        and row["avg_span_ratio"] <= 0.90
        and row["mean_best_pred_to_gt_ratio"] <= 1.90
        and row["wide_single_flag"] <= 0.50
    ]
    best_compact = compact_rows[0] if compact_rows else leaderboard[0]

    selected_rows = {
        "best_overall": best_overall,
        "best_compact": best_compact,
    }
    selected_results: dict[str, dict[str, Any]] = {}
    for tag, row in selected_rows.items():
        selected_results[tag] = next(
            result
            for result in detailed_results
            if result["aggregate"]["scheme"] == row["scheme"]
            and result["aggregate"]["boundary"] == row["boundary"]
        )

    _write_outputs(Path(args.out_dir), leaderboard, selected_results, dataset, scheme_map, boundary_map)


if __name__ == "__main__":
    main(build_args())
