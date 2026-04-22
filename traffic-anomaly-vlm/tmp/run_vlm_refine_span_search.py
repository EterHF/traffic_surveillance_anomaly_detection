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


def _scheme_library() -> list[ScoreScheme]:
    base_object = {
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
    base_scene = {
        "count_std": 0.0,
        "count_delta": 0.0,
        "density_std": 0.0,
        "density_delta": 0.0,
        "lowres_eventness": 0.20,
        "track_layout_eventness": 0.20,
    }
    return [
        ScoreScheme(
            name="source_like_no_smooth",
            object_weights=base_object,
            scene_weights=base_scene,
            branch_weights={"object_score": 0.5, "scene_score": 0.5},
            normalize_per_feature=False,
            final_smooth_window=0,
        ),
        ScoreScheme(
            name="source_like_smooth3",
            object_weights=base_object,
            scene_weights=base_scene,
            branch_weights={"object_score": 0.5, "scene_score": 0.5},
            normalize_per_feature=False,
            final_smooth_window=3,
        ),
        ScoreScheme(
            name="source_like_smooth5",
            object_weights=base_object,
            scene_weights=base_scene,
            branch_weights={"object_score": 0.5, "scene_score": 0.5},
            normalize_per_feature=False,
            final_smooth_window=5,
        ),
        ScoreScheme(
            name="source_like_smooth7",
            object_weights=base_object,
            scene_weights=base_scene,
            branch_weights={"object_score": 0.5, "scene_score": 0.5},
            normalize_per_feature=False,
            final_smooth_window=7,
        ),
        ScoreScheme(
            name="object_only_smooth5",
            object_weights=base_object,
            scene_weights={},
            branch_weights={"object_score": 1.0, "scene_score": 0.0},
            normalize_per_feature=False,
            final_smooth_window=5,
        ),
        ScoreScheme(
            name="object_scene_light_norm",
            object_weights={
                "dir_turn_rates_sum": 0.16,
                "pred_center_sum_err": 0.18,
                "pred_iou_sum_err": 0.18,
                "speed_turn_rates_sum": 0.14,
                "turn_active_ratio": 0.10,
                "collision_warning": 0.14,
                "pred_center_max_err": 0.05,
                "pred_iou_max_err": 0.03,
                "disappear_far_sum": 0.02,
            },
            scene_weights={
                "lowres_eventness": 0.55,
                "track_layout_eventness": 0.35,
                "density_delta": 0.10,
            },
            branch_weights={"object_score": 0.75, "scene_score": 0.25},
            normalize_per_feature=True,
            final_smooth_window=7,
        ),
    ]


def _boundary_library() -> list[BoundaryScheme]:
    schemes: list[BoundaryScheme] = []
    for high_z in [0.55, 0.75, 0.95, 1.15, 1.35]:
        for peak_expand in [(4, 8), (6, 10), (8, 12), (10, 16), (12, 20), (14, 24)]:
            for merge_gap in [4, 8, 12]:
                for min_span_len in [6, 8, 10]:
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
    if not gt_intervals:
        return mask
    for i, frame_id in enumerate(frame_ids):
        for start, end in gt_intervals:
            if int(start) <= int(frame_id) <= int(end):
                mask[i] = True
                break
    return mask


def _video_metrics(
    frame_ids: list[int],
    pred_spans_idx: list[tuple[int, int]],
    gt_intervals: list[list[int]],
    dense_trigger: np.ndarray,
) -> dict[str, Any]:
    pred_frames = [(int(frame_ids[s]), int(frame_ids[e])) for s, e in pred_spans_idx if 0 <= s <= e < len(frame_ids)]
    total_video_frames = max(1, len(frame_ids))
    total_pred_frames = sum(max(0, end - start + 1) for start, end in pred_frames)
    avg_span_ratio = float(total_pred_frames / total_video_frames)

    scaled = robust_unit_scale(dense_trigger)
    gt_mask = _mask_gt(frame_ids, gt_intervals)
    if gt_mask.any() and (~gt_mask).any():
        gt_lift = float(_top_mean(scaled[gt_mask], 0.2) - float(np.mean(scaled[~gt_mask])))
    elif gt_mask.any():
        gt_lift = float(_top_mean(scaled[gt_mask], 0.2))
    else:
        gt_lift = 0.0

    items: list[dict[str, Any]] = []
    hit = 0
    full80 = 0
    iou_sum = 0.0
    cover_sum = 0.0
    pred_gt_ratio_sum = 0.0

    for gt_start, gt_end in gt_intervals:
        gt_len = max(1, int(gt_end) - int(gt_start) + 1)
        best_cover = 0.0
        best_iou = 0.0
        best_ratio = 999.0
        best_pred_span: tuple[int, int] | None = None
        for pred_start, pred_end in pred_frames:
            inter_start = max(int(gt_start), int(pred_start))
            inter_end = min(int(gt_end), int(pred_end))
            inter = max(0, inter_end - inter_start + 1)
            cover = float(inter / gt_len)
            iou = interval_iou((int(gt_start), int(gt_end)), (int(pred_start), int(pred_end)))
            pred_len = max(1, int(pred_end) - int(pred_start) + 1)
            pred_ratio = float(pred_len / gt_len)
            if cover > best_cover or (abs(cover - best_cover) < 1e-6 and iou > best_iou):
                best_cover = cover
                best_iou = iou
                best_ratio = pred_ratio
                best_pred_span = (int(pred_start), int(pred_end))
        hit += int(best_cover > 0.0)
        full80 += int(best_cover >= 0.8)
        iou_sum += best_iou
        cover_sum += best_cover
        pred_gt_ratio_sum += best_ratio if best_pred_span is not None else 999.0
        items.append(
            {
                "start_frame": int(gt_start),
                "end_frame": int(gt_end),
                "best_iou": float(best_iou),
                "best_cover": float(best_cover),
                "best_pred_to_gt_ratio": float(best_ratio),
                "best_pred_span": list(best_pred_span) if best_pred_span is not None else None,
            }
        )

    total_gt = max(1, len(gt_intervals))
    return {
        "num_gt_intervals": int(len(gt_intervals)),
        "num_pred_spans": int(len(pred_frames)),
        "pred_spans": [[int(s), int(e)] for s, e in pred_frames],
        "hit_ratio": float(hit / total_gt),
        "full80_ratio": float(full80 / total_gt),
        "mean_best_iou": float(iou_sum / total_gt),
        "mean_best_cover": float(cover_sum / total_gt),
        "mean_best_pred_to_gt_ratio": float(pred_gt_ratio_sum / total_gt),
        "avg_span_ratio": float(avg_span_ratio),
        "gt_lift": float(gt_lift),
        "items": items,
    }


def _aggregate_video_metrics(video_metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    covers = [float(v["mean_best_cover"]) for v in video_metrics.values()]
    ious = [float(v["mean_best_iou"]) for v in video_metrics.values()]
    hits = [float(v["hit_ratio"]) for v in video_metrics.values()]
    full80 = [float(v["full80_ratio"]) for v in video_metrics.values()]
    pred_counts = [float(v["num_pred_spans"]) for v in video_metrics.values()]
    span_ratios = [float(v["avg_span_ratio"]) for v in video_metrics.values()]
    pred_gt_ratios = [float(v["mean_best_pred_to_gt_ratio"]) for v in video_metrics.values()]
    gt_lifts = [float(v["gt_lift"]) for v in video_metrics.values()]

    mean_cover = float(np.mean(covers)) if covers else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_hit = float(np.mean(hits)) if hits else 0.0
    mean_full80 = float(np.mean(full80)) if full80 else 0.0
    avg_num_pred_spans = float(np.mean(pred_counts)) if pred_counts else 0.0
    avg_span_ratio = float(np.mean(span_ratios)) if span_ratios else 0.0
    mean_pred_gt_ratio = float(np.mean(pred_gt_ratios)) if pred_gt_ratios else 0.0
    mean_gt_lift = float(np.mean(gt_lifts)) if gt_lifts else 0.0

    count_penalty = min(1.0, max(0.0, avg_num_pred_spans - 1.2) / 2.0)
    width_penalty = min(1.0, max(0.0, avg_span_ratio - 0.65) / 0.25)
    excess_penalty = min(1.0, max(0.0, mean_pred_gt_ratio - 1.5) / 4.0)

    vlm_refine_rank = float(
        0.32 * mean_cover
        + 0.20 * mean_iou
        + 0.16 * mean_full80
        + 0.12 * mean_gt_lift
        + 0.10 * mean_hit
        - 0.12 * excess_penalty
        - 0.10 * width_penalty
        - 0.04 * count_penalty
    )

    return {
        "mean_hit_ratio": mean_hit,
        "mean_best_cover": mean_cover,
        "mean_best_iou": mean_iou,
        "full80_ratio": mean_full80,
        "avg_num_pred_spans": avg_num_pred_spans,
        "avg_span_ratio": avg_span_ratio,
        "mean_best_pred_to_gt_ratio": mean_pred_gt_ratio,
        "mean_gt_lift": mean_gt_lift,
        "vlm_refine_rank": vlm_refine_rank,
    }


def _evaluate_combo(
    dataset: dict[str, dict[str, Any]],
    scheme: ScoreScheme,
    boundary: BoundaryScheme,
) -> dict[str, Any]:
    video_metrics: dict[str, dict[str, Any]] = {}

    for video_id, video_data in dataset.items():
        rows = [row["feature_dict"] for row in video_data["window_rows"]]
        curves = _window_curves(rows, scheme)
        dense_object = _dense_scores(curves["object_score"], len(video_data["frame_ids"]), video_data["window_size"])
        dense_scene = _dense_scores(curves["scene_score"], len(video_data["frame_ids"]), video_data["window_size"])
        dense_trigger = _dense_scores(curves["trigger_score"], len(video_data["frame_ids"]), video_data["window_size"])
        pred_spans_idx = _detect_spans(dense_trigger, boundary)
        metrics = _video_metrics(
            frame_ids=video_data["frame_ids"],
            pred_spans_idx=pred_spans_idx,
            gt_intervals=video_data["gt_intervals"],
            dense_trigger=dense_trigger,
        )
        metrics["curves"] = {
            "dense_object": dense_object,
            "dense_scene": dense_scene,
            "dense_trigger": dense_trigger,
        }
        video_metrics[video_id] = metrics

    aggregate = _aggregate_video_metrics(video_metrics)
    aggregate["scheme"] = scheme.name
    aggregate["boundary"] = boundary.name
    return {
        "aggregate": aggregate,
        "video_metrics": video_metrics,
    }


def _write_outputs(
    leaderboard: list[dict[str, Any]],
    best: dict[str, Any],
    dataset: dict[str, dict[str, Any]],
    out_dir: Path,
    scheme_map: dict[str, ScoreScheme],
    boundary_map: dict[str, BoundaryScheme],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "leaderboard.json").open("w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)
    if leaderboard:
        with (out_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(leaderboard[0].keys()))
            writer.writeheader()
            writer.writerows(leaderboard)

    best_dir = out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    aggregate = best["aggregate"]
    with (best_dir / "best_config.json").open("w", encoding="utf-8") as f:
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

    for video_id, metrics in best["video_metrics"].items():
        video_dir = best_dir / "videos" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        curves = metrics.pop("curves")
        index_spans: list[tuple[int, int]] = []
        for span_start, span_end in metrics["pred_spans"]:
            try:
                start_idx = dataset[video_id]["frame_ids"].index(span_start)
                end_idx = dataset[video_id]["frame_ids"].index(span_end)
            except ValueError:
                continue
            index_spans.append((start_idx, end_idx))
        _plot_video(
            frame_ids=dataset[video_id]["frame_ids"],
            curves=curves,
            pred_spans=index_spans,
            gt_intervals=dataset[video_id]["gt_intervals"],
            title=f"{video_id} | {aggregate['scheme']} | {aggregate['boundary']}",
            save_path=video_dir / "scores.png",
        )
        with (video_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search raw spans that are most suitable for later VLM refinement")
    parser.add_argument("--config", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/configs/default.yaml")
    parser.add_argument("--frames-root", default="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal")
    parser.add_argument("--manifest", default="/nvme2/VAD_yemao/data/TU-DAT-25/abnormal.txt")
    parser.add_argument("--cache-dir", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/tmp/raw_score_search/cache")
    parser.add_argument("--out-dir", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/tmp/raw_score_search/vlm_refine_results")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    dataset = _load_or_extract_dataset(args, Path(args.cache_dir))
    schemes = _scheme_library()
    boundaries = _boundary_library()
    scheme_map = {item.name: item for item in schemes}
    boundary_map = {item.name: item for item in boundaries}

    detailed: list[dict[str, Any]] = []
    leaderboard: list[dict[str, Any]] = []
    for scheme in schemes:
        for boundary in boundaries:
            result = _evaluate_combo(dataset, scheme, boundary)
            detailed.append(result)
            leaderboard.append(result["aggregate"])

    leaderboard.sort(
        key=lambda item: (
            float(item["vlm_refine_rank"]),
            float(item["mean_best_cover"]),
            float(item["mean_best_iou"]),
            -float(item["avg_span_ratio"]),
        ),
        reverse=True,
    )
    best_row = leaderboard[0]
    best = next(
        item
        for item in detailed
        if item["aggregate"]["scheme"] == best_row["scheme"]
        and item["aggregate"]["boundary"] == best_row["boundary"]
    )
    _write_outputs(leaderboard, best, dataset, Path(args.out_dir), scheme_map, boundary_map)


if __name__ == "__main__":
    main(build_args())
