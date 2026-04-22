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

from src.eval.utils import interval_iou, parse_manifest_intervals, robust_unit_scale
from src.proposals.boundary_detector import BoundaryDetector, BoundaryDetectorConfig
from tmp.run_raw_score_sweep import ScoreScheme, _dense_scores, _window_curves


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    valid = {key: float(value) for key, value in weights.items() if float(value) > 0.0}
    total = sum(valid.values())
    if total <= 0.0:
        return {}
    return {key: float(value / total) for key, value in valid.items()}


def _score_schemes() -> list[ScoreScheme]:
    turn_focus = _normalize_weights(
        {
            "dir_turn_rates_sum": 1.25,
            "speed_turn_rates_sum": 1.15,
            "turn_active_ratio": 1.10,
            "pred_center_sum_err": 0.85,
            "pred_iou_sum_err": 0.85,
            "disappear_far_sum": 0.10,
        }
    )
    source_like = {
        "dir_turn_rates_sum": 0.5,
        "pred_center_sum_err": 0.5,
        "pred_iou_sum_err": 0.5,
        "speed_turn_rates_sum": 0.5,
        "turn_active_ratio": 0.5,
        "collision_warning": 0.0,
        "pred_center_max_err": 0.0,
        "pred_iou_max_err": 0.0,
        "disappear_far_sum": 0.1,
    }
    return [
        ScoreScheme(
            name="source_like_smooth3",
            object_weights=source_like,
            scene_weights={
                "count_std": 0.0,
                "count_delta": 0.0,
                "density_std": 0.0,
                "density_delta": 0.0,
                "lowres_eventness": 0.20,
                "track_layout_eventness": 0.20,
            },
            branch_weights={"object_score": 0.5, "scene_score": 0.5},
            normalize_per_feature=False,
            final_smooth_window=3,
        ),
        ScoreScheme(
            name="turn_focus__none__scene0.00__smooth3",
            object_weights=turn_focus,
            scene_weights={},
            branch_weights={"object_score": 1.0, "scene_score": 0.0},
            normalize_per_feature=True,
            final_smooth_window=3,
        ),
        ScoreScheme(
            name="turn_focus__layout_only__scene0.15__smooth5",
            object_weights=turn_focus,
            scene_weights={"track_layout_eventness": 1.0},
            branch_weights={"object_score": 0.85, "scene_score": 0.15},
            normalize_per_feature=True,
            final_smooth_window=5,
        ),
    ]


def _boundary_cfgs() -> list[BoundaryDetectorConfig]:
    configs: list[BoundaryDetectorConfig] = []
    for high_z in [1.0, 1.15, 1.35]:
        for peak_gap in [2, 4]:
            for peak_expand in [(2, 4), (4, 8), (6, 10), (8, 12), (10, 16)]:
                for merge_gap in [0, 2, 4, 8, 12]:
                    for min_span_len in [4, 6, 8]:
                        configs.append(
                            BoundaryDetectorConfig(
                                method="by_peeks",
                                use_savgol_filter=False,
                                high_z=float(high_z),
                                low_z=0.25,
                                peak_gap=int(peak_gap),
                                peak_expand=tuple(peak_expand),
                                min_span_len=int(min_span_len),
                                merge_gap=int(merge_gap),
                            )
                        )

    for high_z in [0.8, 1.0, 1.2]:
        for low_z in [0.0, 0.1, 0.25, 0.4]:
            for peak_expand in [(2, 4), (4, 8), (6, 10), (8, 12)]:
                for merge_gap in [0, 2, 4, 8]:
                    for min_span_len in [4, 6, 8]:
                        configs.append(
                            BoundaryDetectorConfig(
                                method="by_thres",
                                use_savgol_filter=False,
                                high_z=float(high_z),
                                low_z=float(low_z),
                                peak_gap=4,
                                peak_expand=tuple(peak_expand),
                                min_span_len=int(min_span_len),
                                merge_gap=int(merge_gap),
                            )
                        )
    return configs


def _load_dataset_from_cache(cache_dir: Path, manifest_path: str) -> dict[str, dict[str, Any]]:
    dataset: dict[str, dict[str, Any]] = {}
    for video_dir in sorted(path for path in cache_dir.iterdir() if path.is_dir()):
        feature_path = video_dir / "feature_rows.json"
        if not feature_path.exists():
            continue
        with feature_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        dataset[video_dir.name] = {
            "frame_ids": [int(value) for value in payload["sampled_frame_ids"]],
            "window_size": int(payload["window_size"]),
            "window_rows": list(payload["window_rows"]),
            "gt_intervals": parse_manifest_intervals(manifest_path, video_dir.name),
        }
    return dataset


def _boundary_name(cfg: BoundaryDetectorConfig) -> str:
    expand = f"{int(cfg.peak_expand[0])}_{int(cfg.peak_expand[1])}"
    if cfg.method == "by_peeks":
        return (
            f"peeks_hz{cfg.high_z:g}"
            f"_pg{int(cfg.peak_gap)}"
            f"_exp{expand}"
            f"_mg{int(cfg.merge_gap)}"
            f"_min{int(cfg.min_span_len)}"
        )
    return (
        f"thres_hz{cfg.high_z:g}"
        f"_lz{cfg.low_z:g}"
        f"_exp{expand}"
        f"_mg{int(cfg.merge_gap)}"
        f"_min{int(cfg.min_span_len)}"
    )


def _top_mean(values: np.ndarray, ratio: float = 0.2) -> float:
    if values.size == 0:
        return 0.0
    keep = max(1, int(np.ceil(values.size * float(ratio))))
    return float(np.mean(np.sort(values)[-keep:]))


def _mask_gt(frame_ids: list[int], gt_intervals: list[list[int]]) -> np.ndarray:
    mask = np.zeros((len(frame_ids),), dtype=bool)
    for index, frame_id in enumerate(frame_ids):
        for start, end in gt_intervals:
            if int(start) <= int(frame_id) <= int(end):
                mask[index] = True
                break
    return mask


def _analyze_spans(scores: np.ndarray, cfg: BoundaryDetectorConfig) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    detector = BoundaryDetector(cfg)
    values = scores.astype(np.float32).tolist()
    if cfg.method == "by_peeks":
        raw = detector._peeks_to_spans(
            detector._detect_peeks(values),
            gap=detector.peak_gap,
            expand=detector.peak_expand,
            n=len(values),
        )
    else:
        raw = detector._detect_spans_by_adaptive_threshold(values)
    final = detector._refine_spans(raw, n=len(values))
    return raw, final


def _gt_metrics(
    frame_ids: list[int],
    pred_spans_idx: list[tuple[int, int]],
    gt_intervals: list[list[int]],
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
        best_cover = 0.0
        best_iou = 0.0
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
    return {
        "hit_ratio": float(hit / total_gt),
        "full80_ratio": float(full80 / total_gt),
        "mean_best_cover": float(cover_sum / total_gt),
        "mean_best_iou": float(iou_sum / total_gt),
        "mean_best_pred_to_gt_ratio": float(pred_gt_ratio_sum / total_gt),
        "avg_span_ratio": avg_span_ratio,
        "num_pred_spans": float(len(pred_frames)),
        "gt_lift": gt_lift,
        "wide_single_flag": float(len(pred_frames) <= 1 and avg_span_ratio >= 0.90),
    }


def _merge_to_single_flag(raw_spans: list[tuple[int, int]], final_spans: list[tuple[int, int]], avg_span_ratio: float) -> float:
    return float(len(raw_spans) >= 2 and len(final_spans) <= 1 and avg_span_ratio >= 0.80)


def _span_count_bonus(avg_num_pred_spans: float) -> float:
    if 1.5 <= avg_num_pred_spans <= 3.5:
        return 1.0
    if avg_num_pred_spans < 1.5:
        return max(0.0, avg_num_pred_spans - 0.5)
    return max(0.0, 1.0 - (avg_num_pred_spans - 3.5) / 2.0)


def _coverage_rank(agg: dict[str, float]) -> float:
    span_penalty = min(1.0, max(0.0, agg["avg_span_ratio"] - 0.92) / 0.10)
    return float(
        0.42 * agg["mean_best_cover"]
        + 0.18 * agg["full80_ratio"]
        + 0.18 * agg["mean_best_iou"]
        + 0.10 * agg["hit_ratio"]
        + 0.08 * agg["gt_lift"]
        - 0.04 * span_penalty
    )


def _vlm_ready_rank(agg: dict[str, float]) -> float:
    width_penalty = min(1.0, max(0.0, agg["avg_span_ratio"] - 0.82) / 0.20)
    excess_penalty = min(1.0, max(0.0, agg["mean_best_pred_to_gt_ratio"] - 1.8) / 2.0)
    fragment_penalty = min(1.0, max(0.0, agg["num_pred_spans"] - 3.5) / 2.5)
    merge_to_single = float(agg.get("merge_to_single_rate", agg.get("merge_to_single_flag", 0.0)))
    return float(
        0.30 * agg["mean_best_cover"]
        + 0.14 * agg["full80_ratio"]
        + 0.16 * agg["mean_best_iou"]
        + 0.08 * agg["hit_ratio"]
        + 0.08 * agg["gt_lift"]
        + 0.08 * _span_count_bonus(agg["num_pred_spans"])
        - 0.06 * width_penalty
        - 0.04 * excess_penalty
        - 0.06 * agg["wide_single_flag"]
        - 0.06 * merge_to_single
        - 0.04 * fragment_penalty
    )


def _aggregate(video_metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = list(video_metrics[0].keys()) if video_metrics else []
    return {key: float(np.mean([row[key] for row in video_metrics])) for key in keys}


def _evaluate_combo(
    dataset: dict[str, dict[str, Any]],
    scheme: ScoreScheme,
    boundary_cfg: BoundaryDetectorConfig,
) -> dict[str, Any]:
    metrics_per_video: dict[str, dict[str, float]] = {}
    aggregate_rows: list[dict[str, float]] = []

    for video_id, video_data in dataset.items():
        rows = [row["feature_dict"] for row in video_data["window_rows"]]
        curves = _window_curves(rows, scheme)
        dense_trigger = _dense_scores(
            curves["trigger_score"],
            len(video_data["frame_ids"]),
            video_data["window_size"],
        )

        raw_spans, final_spans = _analyze_spans(dense_trigger, boundary_cfg)
        metrics = _gt_metrics(video_data["frame_ids"], final_spans, video_data["gt_intervals"], dense_trigger)
        raw_span_count = float(len(raw_spans))
        final_span_count = float(len(final_spans))
        merge_compression = float(max(0.0, raw_span_count - final_span_count) / max(1.0, raw_span_count))

        metrics.update(
            {
                "raw_span_count": raw_span_count,
                "final_span_count": final_span_count,
                "merge_compression": merge_compression,
                "merge_to_single_flag": _merge_to_single_flag(raw_spans, final_spans, metrics["avg_span_ratio"]),
            }
        )
        metrics_per_video[video_id] = metrics
        aggregate_rows.append(metrics)

    aggregate = _aggregate(aggregate_rows)
    aggregate["merge_to_single_rate"] = float(aggregate.get("merge_to_single_flag", 0.0))
    aggregate["scheme"] = scheme.name
    aggregate["boundary"] = _boundary_name(boundary_cfg)
    aggregate["coverage_rank"] = _coverage_rank(aggregate)
    aggregate["vlm_ready_rank"] = _vlm_ready_rank(aggregate)
    aggregate["method"] = str(boundary_cfg.method)
    aggregate["high_z"] = float(boundary_cfg.high_z)
    aggregate["low_z"] = float(boundary_cfg.low_z)
    aggregate["peak_gap"] = int(boundary_cfg.peak_gap)
    aggregate["peak_expand_left"] = int(boundary_cfg.peak_expand[0])
    aggregate["peak_expand_right"] = int(boundary_cfg.peak_expand[1])
    aggregate["peak_expand_sum"] = int(boundary_cfg.peak_expand[0] + boundary_cfg.peak_expand[1])
    aggregate["merge_gap"] = int(boundary_cfg.merge_gap)
    aggregate["min_span_len"] = int(boundary_cfg.min_span_len)
    return {"aggregate": aggregate, "video_metrics": metrics_per_video}


def _group_mean(records: list[dict[str, Any]], key: str, metric: str, top_n: int = 100) -> dict[str, float]:
    subset = records[: min(top_n, len(records))]
    grouped: dict[str, list[float]] = {}
    for row in subset:
        grouped.setdefault(str(row[key]), []).append(float(row[metric]))
    return {name: float(np.mean(values)) for name, values in sorted(grouped.items(), key=lambda item: item[0])}


def _group_frequency(records: list[dict[str, Any]], key: str, top_n: int = 100) -> dict[str, int]:
    subset = records[: min(top_n, len(records))]
    counts: dict[str, int] = {}
    for row in subset:
        counts[str(row[key])] = counts.get(str(row[key]), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _write_leaderboard(records: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "leaderboard.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    if not records:
        return
    fieldnames = list(records[0].keys())
    with (out_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_best(path: Path, result: dict[str, Any], scheme_map: dict[str, ScoreScheme], boundary_map: dict[str, BoundaryDetectorConfig]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    aggregate = result["aggregate"]
    with (path / "best_config.json").open("w", encoding="utf-8") as f:
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


def _write_report(
    out_dir: Path,
    coverage_best: dict[str, Any],
    vlm_best: dict[str, Any],
    thres_best: dict[str, Any] | None,
    all_records: list[dict[str, Any]],
) -> None:
    method_summary = {
        method: {
            "mean_cover": float(np.mean([float(row["mean_best_cover"]) for row in all_records if row["method"] == method])),
            "mean_iou": float(np.mean([float(row["mean_best_iou"]) for row in all_records if row["method"] == method])),
            "mean_wide_single": float(np.mean([float(row["wide_single_flag"]) for row in all_records if row["method"] == method])),
            "mean_merge_to_single": float(np.mean([float(row["merge_to_single_rate"]) for row in all_records if row["method"] == method])),
        }
        for method in sorted({str(row["method"]) for row in all_records})
    }
    top_vlm = sorted(all_records, key=lambda row: float(row["vlm_ready_rank"]), reverse=True)

    report = [
        "# Boundary Behavior Search",
        "",
        "## Coverage Best",
        f"- scheme: `{coverage_best['aggregate']['scheme']}`",
        f"- boundary: `{coverage_best['aggregate']['boundary']}`",
        f"- mean_best_cover: `{coverage_best['aggregate']['mean_best_cover']:.4f}`",
        f"- mean_best_iou: `{coverage_best['aggregate']['mean_best_iou']:.4f}`",
        f"- avg_num_pred_spans: `{coverage_best['aggregate']['num_pred_spans']:.4f}`",
        f"- avg_span_ratio: `{coverage_best['aggregate']['avg_span_ratio']:.4f}`",
        f"- wide_single_flag: `{coverage_best['aggregate']['wide_single_flag']:.4f}`",
        "",
        "## VLM-Ready Best",
        f"- scheme: `{vlm_best['aggregate']['scheme']}`",
        f"- boundary: `{vlm_best['aggregate']['boundary']}`",
        f"- mean_best_cover: `{vlm_best['aggregate']['mean_best_cover']:.4f}`",
        f"- mean_best_iou: `{vlm_best['aggregate']['mean_best_iou']:.4f}`",
        f"- avg_num_pred_spans: `{vlm_best['aggregate']['num_pred_spans']:.4f}`",
        f"- avg_span_ratio: `{vlm_best['aggregate']['avg_span_ratio']:.4f}`",
        f"- wide_single_flag: `{vlm_best['aggregate']['wide_single_flag']:.4f}`",
        f"- merge_to_single_rate: `{vlm_best['aggregate']['merge_to_single_rate']:.4f}`",
        "",
    ]
    if thres_best is not None:
        report.extend(
            [
                "## Best `by_thres`",
                f"- scheme: `{thres_best['aggregate']['scheme']}`",
                f"- boundary: `{thres_best['aggregate']['boundary']}`",
                f"- mean_best_cover: `{thres_best['aggregate']['mean_best_cover']:.4f}`",
                f"- mean_best_iou: `{thres_best['aggregate']['mean_best_iou']:.4f}`",
                f"- avg_num_pred_spans: `{thres_best['aggregate']['num_pred_spans']:.4f}`",
                f"- avg_span_ratio: `{thres_best['aggregate']['avg_span_ratio']:.4f}`",
                f"- wide_single_flag: `{thres_best['aggregate']['wide_single_flag']:.4f}`",
                "",
            ]
        )

    report.extend(
        [
            "## Method Summary",
            f"- `{json.dumps(method_summary, ensure_ascii=False, indent=2)}`",
            "",
            "## Top-100 VLM-Ready Frequency",
            f"- method: `{json.dumps(_group_frequency(top_vlm, 'method', 100), ensure_ascii=False)}`",
            f"- merge_gap: `{json.dumps(_group_frequency(top_vlm, 'merge_gap', 100), ensure_ascii=False)}`",
            f"- peak_expand_sum: `{json.dumps(_group_frequency(top_vlm, 'peak_expand_sum', 100), ensure_ascii=False)}`",
            "",
            "## Top-100 VLM-Ready Means",
            f"- merge_gap -> wide_single_flag: `{json.dumps(_group_mean(top_vlm, 'merge_gap', 'wide_single_flag', 100), ensure_ascii=False)}`",
            f"- merge_gap -> merge_to_single_rate: `{json.dumps(_group_mean(top_vlm, 'merge_gap', 'merge_to_single_rate', 100), ensure_ascii=False)}`",
            f"- peak_expand_sum -> avg_span_ratio: `{json.dumps(_group_mean(top_vlm, 'peak_expand_sum', 'avg_span_ratio', 100), ensure_ascii=False)}`",
            f"- peak_expand_sum -> mean_best_cover: `{json.dumps(_group_mean(top_vlm, 'peak_expand_sum', 'mean_best_cover', 100), ensure_ascii=False)}`",
            "",
            "## Takeaways",
            "- `merge_gap` and `peak_expand` are the main drivers of over-wide single spans; they matter more than `min_span_len` in the current grid.",
            "- `by_thres` is worth re-checking after the source bug fix, but it still needs to beat `by_peeks` on both coverage and compactness before becoming a default coarse proposal.",
            "- For later VLM usage, the better preset is not the one with the highest raw coverage alone, but the one that preserves multiple compact candidate spans without dropping the anomaly body.",
        ]
    )
    with (out_dir / "boundary_behavior_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search boundary behavior for VLM-ready coarse spans")
    parser.add_argument("--config", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/configs/default.yaml")
    parser.add_argument("--frames-root", default="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal")
    parser.add_argument("--manifest", default="/nvme2/VAD_yemao/data/TU-DAT-25/abnormal.txt")
    parser.add_argument("--cache-dir", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/tmp/raw_score_search/cache")
    parser.add_argument("--out-dir", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/tmp/raw_score_search/boundary_behavior")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    dataset = _load_dataset_from_cache(Path(args.cache_dir), args.manifest)
    schemes = _score_schemes()
    boundaries = _boundary_cfgs()
    scheme_map = {scheme.name: scheme for scheme in schemes}
    boundary_map = {_boundary_name(cfg): cfg for cfg in boundaries}

    detailed_results: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    total = len(schemes) * len(boundaries)
    index = 0
    for scheme in schemes:
        for boundary in boundaries:
            index += 1
            if index % 100 == 0 or index == 1 or index == total:
                print(f"[boundary-search] {index}/{total} {scheme.name} | {_boundary_name(boundary)}")
            result = _evaluate_combo(dataset, scheme, boundary)
            detailed_results.append(result)
            aggregates.append(result["aggregate"])

    aggregates.sort(
        key=lambda row: (
            float(row["vlm_ready_rank"]),
            float(row["mean_best_cover"]),
            float(row["mean_best_iou"]),
        ),
        reverse=True,
    )
    out_dir = Path(args.out_dir)
    _write_leaderboard(aggregates, out_dir)

    coverage_aggregate = max(aggregates, key=lambda row: float(row["coverage_rank"]))
    vlm_aggregate = aggregates[0]
    thres_aggregate = next((row for row in aggregates if row["method"] == "by_thres"), None)

    def _find(aggregate: dict[str, Any]) -> dict[str, Any]:
        return next(
            result
            for result in detailed_results
            if result["aggregate"]["scheme"] == aggregate["scheme"]
            and result["aggregate"]["boundary"] == aggregate["boundary"]
        )

    coverage_best = _find(coverage_aggregate)
    vlm_best = _find(vlm_aggregate)
    thres_best = _find(thres_aggregate) if thres_aggregate is not None else None

    _write_best(out_dir / "best_coverage", coverage_best, scheme_map, boundary_map)
    _write_best(out_dir / "best_vlm_ready", vlm_best, scheme_map, boundary_map)
    if thres_best is not None:
        _write_best(out_dir / "best_by_thres", thres_best, scheme_map, boundary_map)
    _write_report(out_dir, coverage_best, vlm_best, thres_best, aggregates)


if __name__ == "__main__":
    main(build_args())
