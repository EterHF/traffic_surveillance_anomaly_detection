from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.utils import interval_iou, parse_manifest_intervals, robust_unit_scale
from src.proposals.boundary_detector import BoundaryDetector, BoundaryDetectorConfig
from tmp.orchestrator_raw_score import RawScoreOrchestrator


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
    "count_std",
    "count_delta",
    "density_std",
    "density_delta",
    "lowres_eventness",
    "track_layout_eventness",
]


@dataclass
class ScoreScheme:
    name: str
    object_weights: dict[str, float]
    scene_weights: dict[str, float]
    branch_weights: dict[str, float]
    normalize_per_feature: bool = True
    final_smooth_window: int = 0


@dataclass
class BoundaryScheme:
    name: str
    method: str
    high_z: float
    low_z: float
    peak_gap: int
    peak_expand: tuple[int, int]
    min_span_len: int
    merge_gap: int
    use_savgol_filter: bool = False


def _safe_series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=np.float32)


def _weighted_sum(curves: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    valid = {k: float(v) for k, v in weights.items() if float(v) > 0.0}
    if not valid:
        length = len(next(iter(curves.values()))) if curves else 0
        return np.zeros((length,), dtype=np.float32)
    total = sum(valid.values())
    out = np.zeros_like(next(iter(curves.values())), dtype=np.float32)
    for key, weight in valid.items():
        out += curves.get(key, 0.0) * float(weight)
    return (out / max(1e-6, total)).astype(np.float32)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.astype(np.float32, copy=True)
    pad = max(0, int(window) // 2)
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones((int(window),), dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _window_curves(rows: list[dict[str, Any]], scheme: ScoreScheme) -> dict[str, np.ndarray]:
    object_curves: dict[str, np.ndarray] = {}
    scene_curves: dict[str, np.ndarray] = {}

    for key in OBJECT_KEYS:
        values = _safe_series(rows, key)
        object_curves[key] = robust_unit_scale(values) if scheme.normalize_per_feature else values

    for key in SCENE_KEYS:
        values = _safe_series(rows, key)
        scene_curves[key] = robust_unit_scale(values) if scheme.normalize_per_feature else values

    object_score = _weighted_sum(object_curves, scheme.object_weights)
    scene_score = _weighted_sum(scene_curves, scheme.scene_weights)
    trigger_score = _weighted_sum(
        {
            "object_score": object_score,
            "scene_score": scene_score,
        },
        scheme.branch_weights,
    )
    trigger_score = robust_unit_scale(trigger_score)
    if int(scheme.final_smooth_window) > 1:
        trigger_score = robust_unit_scale(_moving_average(trigger_score, int(scheme.final_smooth_window)))

    return {
        "object_score": object_score.astype(np.float32),
        "scene_score": scene_score.astype(np.float32),
        "trigger_score": trigger_score.astype(np.float32),
    }


def _dense_scores(window_scores: np.ndarray, num_frames: int, window_size: int) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    if window_scores.size == 0:
        return np.zeros((num_frames,), dtype=np.float32)
    prefix = np.full((max(0, int(window_size) - 1),), float(window_scores[0]), dtype=np.float32)
    dense = np.concatenate([prefix, window_scores.astype(np.float32)], axis=0)
    if dense.size < num_frames:
        dense = np.pad(dense, (0, num_frames - dense.size), mode="edge")
    return dense[:num_frames].astype(np.float32)


def _detect_spans(scores: np.ndarray, boundary: BoundaryScheme) -> list[tuple[int, int]]:
    if str(boundary.method) == "by_thres":
        values = scores.astype(np.float32)
        if values.size == 0:
            return []
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        sigma = max(1e-6, 1.4826 * mad)
        high_thr = median + float(boundary.high_z) * sigma
        low_thr = median + float(boundary.low_z) * sigma
        low_thr = min(low_thr, high_thr)

        spans: list[tuple[int, int]] = []
        in_span = False
        start = 0
        for idx, value in enumerate(values.tolist()):
            if not in_span and value >= high_thr:
                start = idx
                in_span = True
            elif in_span and value < low_thr:
                spans.append((start, idx - 1))
                in_span = False
        if in_span:
            spans.append((start, len(values) - 1))

        expanded: list[tuple[int, int]] = []
        for start, end in spans:
            start = max(0, int(start) - int(boundary.peak_expand[0]) // 2)
            end = min(len(values) - 1, int(end) + int(boundary.peak_expand[1]) // 2)
            expanded.append((start, end))

        if not expanded:
            return []
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= int(boundary.merge_gap):
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return [span for span in merged if (span[1] - span[0] + 1) >= int(boundary.min_span_len)]

    detector = BoundaryDetector(
        cfg=BoundaryDetectorConfig(
            method=str(boundary.method),
            use_savgol_filter=bool(boundary.use_savgol_filter),
            high_z=float(boundary.high_z),
            low_z=float(boundary.low_z),
            peak_gap=int(boundary.peak_gap),
            peak_expand=tuple(boundary.peak_expand),
            min_span_len=int(boundary.min_span_len),
            merge_gap=int(boundary.merge_gap),
        )
    )
    raw = detector.detect(scores.astype(np.float32).tolist())
    return raw if isinstance(raw, list) else []


def _gt_metrics(
    frame_ids: list[int],
    pred_spans: list[tuple[int, int]],
    gt_intervals: list[list[int]],
) -> dict[str, Any]:
    pred_frames = [(int(frame_ids[s]), int(frame_ids[e])) for s, e in pred_spans if 0 <= s <= e < len(frame_ids)]
    items: list[dict[str, Any]] = []
    hit = 0
    full80 = 0
    iou_sum = 0.0
    cover_sum = 0.0

    for gt_start, gt_end in gt_intervals:
        best_iou = 0.0
        best_cover = 0.0
        for pred_start, pred_end in pred_frames:
            inter_start = max(int(gt_start), int(pred_start))
            inter_end = min(int(gt_end), int(pred_end))
            inter = max(0, inter_end - inter_start + 1)
            gt_len = max(1, int(gt_end) - int(gt_start) + 1)
            cover = float(inter / gt_len)
            best_cover = max(best_cover, cover)
            best_iou = max(best_iou, interval_iou((int(gt_start), int(gt_end)), (int(pred_start), int(pred_end))))
        hit += int(best_cover > 0.0)
        full80 += int(best_cover >= 0.8)
        iou_sum += best_iou
        cover_sum += best_cover
        items.append(
            {
                "start_frame": int(gt_start),
                "end_frame": int(gt_end),
                "best_iou": float(best_iou),
                "best_cover": float(best_cover),
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
        "items": items,
    }


def _plot_video(
    frame_ids: list[int],
    curves: dict[str, np.ndarray],
    pred_spans: list[tuple[int, int]],
    gt_intervals: list[list[int]],
    title: str,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    dense_object = curves["dense_object"]
    dense_scene = curves["dense_scene"]
    dense_trigger = curves["dense_trigger"]

    for ax, curve, name in zip(
        axes,
        [dense_object, dense_scene, dense_trigger],
        ["object_score", "scene_score", "trigger_score"],
    ):
        ax.plot(frame_ids, curve, linewidth=1.3, label=name)
        for s, e in gt_intervals:
            ax.axvspan(s, e, color="red", alpha=0.10)
        for s, e in pred_spans:
            ax.axvspan(frame_ids[s], frame_ids[e], color="#ffb74d", alpha=0.18)
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("frame_id")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _scheme_library() -> list[ScoreScheme]:
    return [
        ScoreScheme(
            name="baseline_current_raw",
            object_weights={
                "dir_turn_rates_sum": 0.5,
                "pred_center_sum_err": 0.5,
                "pred_iou_sum_err": 0.5,
                "speed_turn_rates_sum": 0.5,
                "turn_active_ratio": 0.5,
                "collision_warning": 0.0,
                "pred_center_max_err": 0.0,
                "pred_iou_max_err": 0.0,
                "disappear_far_sum": 0.10,
            },
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
            final_smooth_window=5,
        ),
        ScoreScheme(
            name="object_residual_turn",
            object_weights={
                "dir_turn_rates_sum": 0.20,
                "pred_center_sum_err": 0.25,
                "pred_iou_sum_err": 0.20,
                "speed_turn_rates_sum": 0.20,
                "turn_active_ratio": 0.15,
            },
            scene_weights={},
            branch_weights={"object_score": 1.0, "scene_score": 0.0},
            normalize_per_feature=True,
            final_smooth_window=7,
        ),
        ScoreScheme(
            name="object_balanced_with_collision",
            object_weights={
                "dir_turn_rates_sum": 0.15,
                "pred_center_sum_err": 0.18,
                "pred_iou_sum_err": 0.18,
                "speed_turn_rates_sum": 0.15,
                "turn_active_ratio": 0.10,
                "collision_warning": 0.14,
                "pred_center_max_err": 0.05,
                "pred_iou_max_err": 0.03,
                "disappear_far_sum": 0.02,
            },
            scene_weights={},
            branch_weights={"object_score": 1.0, "scene_score": 0.0},
            normalize_per_feature=True,
            final_smooth_window=7,
        ),
        ScoreScheme(
            name="object_scene_light",
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
        ScoreScheme(
            name="object_scene_balanced",
            object_weights={
                "dir_turn_rates_sum": 0.14,
                "pred_center_sum_err": 0.16,
                "pred_iou_sum_err": 0.16,
                "speed_turn_rates_sum": 0.12,
                "turn_active_ratio": 0.08,
                "collision_warning": 0.18,
                "pred_center_max_err": 0.08,
                "pred_iou_max_err": 0.05,
                "disappear_far_sum": 0.03,
            },
            scene_weights={
                "lowres_eventness": 0.45,
                "track_layout_eventness": 0.35,
                "density_delta": 0.10,
                "count_delta": 0.10,
            },
            branch_weights={"object_score": 0.65, "scene_score": 0.35},
            normalize_per_feature=True,
            final_smooth_window=9,
        ),
    ]


def _boundary_library() -> list[BoundaryScheme]:
    schemes: list[BoundaryScheme] = []
    for high_z in [0.55, 0.75, 0.95]:
        for peak_expand in [(6, 12), (10, 18), (14, 24)]:
            schemes.append(
                BoundaryScheme(
                    name=f"peeks_hz{high_z}_exp{peak_expand[0]}_{peak_expand[1]}",
                    method="by_peeks",
                    high_z=float(high_z),
                    low_z=0.25,
                    peak_gap=4,
                    peak_expand=tuple(peak_expand),
                    min_span_len=8,
                    merge_gap=8,
                    use_savgol_filter=False,
                )
            )
    for high_z in [0.6, 0.8, 1.0]:
        for low_z in [0.10, 0.25, 0.40]:
            for peak_expand in [(4, 8), (8, 12)]:
                schemes.append(
                    BoundaryScheme(
                        name=f"thres_hz{high_z}_lz{low_z}_exp{peak_expand[0]}_{peak_expand[1]}",
                        method="by_thres",
                        high_z=float(high_z),
                        low_z=float(low_z),
                        peak_gap=4,
                        peak_expand=tuple(peak_expand),
                        min_span_len=8,
                        merge_gap=6,
                        use_savgol_filter=False,
                    )
                )
    return schemes


def _score_combo(
    dataset: dict[str, dict[str, Any]],
    scheme: ScoreScheme,
    boundary: BoundaryScheme,
) -> dict[str, Any]:
    video_metrics: dict[str, Any] = {}
    hit_values: list[float] = []
    cover_values: list[float] = []
    iou_values: list[float] = []
    full80_values: list[float] = []
    pred_counts: list[int] = []

    for video_id, video_data in dataset.items():
        rows = [row["feature_dict"] for row in video_data["window_rows"]]
        curves = _window_curves(rows, scheme)
        dense_object = _dense_scores(curves["object_score"], len(video_data["frame_ids"]), video_data["window_size"])
        dense_scene = _dense_scores(curves["scene_score"], len(video_data["frame_ids"]), video_data["window_size"])
        dense_trigger = _dense_scores(curves["trigger_score"], len(video_data["frame_ids"]), video_data["window_size"])
        pred_spans = _detect_spans(dense_trigger, boundary)
        metrics = _gt_metrics(video_data["frame_ids"], pred_spans, video_data["gt_intervals"])
        metrics["curves"] = {
            "dense_object": dense_object,
            "dense_scene": dense_scene,
            "dense_trigger": dense_trigger,
        }
        video_metrics[video_id] = metrics
        hit_values.append(float(metrics["hit_ratio"]))
        cover_values.append(float(metrics["mean_best_cover"]))
        iou_values.append(float(metrics["mean_best_iou"]))
        full80_values.append(float(metrics["full80_ratio"]))
        pred_counts.append(int(metrics["num_pred_spans"]))

    avg_pred_spans = float(np.mean(pred_counts)) if pred_counts else 0.0
    aggregate = {
        "scheme": scheme.name,
        "boundary": boundary.name,
        "mean_hit_ratio": float(np.mean(hit_values)) if hit_values else 0.0,
        "mean_best_cover": float(np.mean(cover_values)) if cover_values else 0.0,
        "mean_best_iou": float(np.mean(iou_values)) if iou_values else 0.0,
        "full80_ratio": float(np.mean(full80_values)) if full80_values else 0.0,
        "avg_num_pred_spans": avg_pred_spans,
    }
    aggregate["rank_score"] = float(
        0.50 * aggregate["mean_best_cover"]
        + 0.25 * aggregate["mean_best_iou"]
        + 0.20 * aggregate["mean_hit_ratio"]
        + 0.10 * aggregate["full80_ratio"]
        - 0.03 * min(1.0, avg_pred_spans / 6.0)
    )
    return {
        "aggregate": aggregate,
        "video_metrics": video_metrics,
    }


def _load_or_extract_dataset(args: argparse.Namespace, cache_root: Path) -> dict[str, dict[str, Any]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    video_ids = sorted(path.name for path in Path(args.frames_root).iterdir() if path.is_dir())
    requested_video_ids = getattr(args, "video_ids", None)
    if requested_video_ids:
        keep = set(str(video_id) for video_id in requested_video_ids)
        video_ids = [video_id for video_id in video_ids if video_id in keep]
    orchestrator = RawScoreOrchestrator(
        args.config,
        precompute_clip=bool(getattr(args, "precompute_clip", False)),
        precompute_raft=bool(getattr(args, "precompute_raft", False)),
    )
    dataset: dict[str, dict[str, Any]] = {}

    for video_id in video_ids:
        video_cache = cache_root / video_id
        feature_path = video_cache / "feature_rows.json"
        if not feature_path.exists() or args.force_extract:
            orchestrator.extract_video(
                frames_dir=str(Path(args.frames_root) / video_id),
                output_dir=str(video_cache),
                video_id=video_id,
            )
        with feature_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        dataset[video_id] = {
            "frame_ids": [int(v) for v in payload["sampled_frame_ids"]],
            "window_size": int(payload["window_size"]),
            "window_rows": list(payload["window_rows"]),
            "gt_intervals": parse_manifest_intervals(args.manifest, video_id),
        }
    return dataset


def _write_leaderboard(records: list[dict[str, Any]], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "leaderboard.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    if not records:
        return
    fieldnames = list(records[0].keys())
    with (save_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_best_outputs(
    best: dict[str, Any],
    dataset: dict[str, dict[str, Any]],
    save_dir: Path,
    scheme_map: dict[str, ScoreScheme],
    boundary_map: dict[str, BoundaryScheme],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    aggregate = best["aggregate"]
    scheme = scheme_map[aggregate["scheme"]]
    boundary = boundary_map[aggregate["boundary"]]
    with (save_dir / "best_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scheme": asdict(scheme),
                "boundary": asdict(boundary),
                "aggregate": aggregate,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    for video_id, metrics in best["video_metrics"].items():
        video_dir = save_dir / "videos" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        curves = metrics.pop("curves")
        _plot_video(
            frame_ids=dataset[video_id]["frame_ids"],
            curves=curves,
            pred_spans=[
                (
                    dataset[video_id]["frame_ids"].index(span[0]) if span[0] in dataset[video_id]["frame_ids"] else 0,
                    dataset[video_id]["frame_ids"].index(span[1]) if span[1] in dataset[video_id]["frame_ids"] else len(dataset[video_id]["frame_ids"]) - 1,
                )
                for span in metrics["pred_spans"]
            ],
            gt_intervals=dataset[video_id]["gt_intervals"],
            title=f"{video_id} | {scheme.name} | {boundary.name}",
            save_path=video_dir / "scores.png",
        )
        with (video_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search raw initial anomaly score schemes on TU-DAT abnormal set")
    parser.add_argument("--config", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/configs/default.yaml")
    parser.add_argument("--frames-root", default="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal")
    parser.add_argument("--manifest", default="/nvme2/VAD_yemao/data/TU-DAT-25/abnormal.txt")
    parser.add_argument("--cache-dir", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/tmp/raw_score_search/cache")
    parser.add_argument("--out-dir", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/tmp/raw_score_search/results")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    cache_root = Path(args.cache_dir)
    out_root = Path(args.out_dir)

    dataset = _load_or_extract_dataset(args, cache_root)
    schemes = _scheme_library()
    boundaries = _boundary_library()
    scheme_map = {scheme.name: scheme for scheme in schemes}
    boundary_map = {boundary.name: boundary for boundary in boundaries}

    all_results: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []
    for scheme in schemes:
        for boundary in boundaries:
            result = _score_combo(dataset, scheme, boundary)
            all_results.append(result["aggregate"])
            detailed_results.append(result)

    all_results.sort(
        key=lambda item: (
            float(item["rank_score"]),
            float(item["mean_best_cover"]),
            float(item["mean_best_iou"]),
            -float(item["avg_num_pred_spans"]),
        ),
        reverse=True,
    )
    _write_leaderboard(all_results, out_root)

    best_aggregate = all_results[0]
    best = next(
        result
        for result in detailed_results
        if result["aggregate"]["scheme"] == best_aggregate["scheme"]
        and result["aggregate"]["boundary"] == best_aggregate["boundary"]
    )
    _write_best_outputs(best, dataset, out_root / "best", scheme_map, boundary_map)


if __name__ == "__main__":
    main(build_args())
