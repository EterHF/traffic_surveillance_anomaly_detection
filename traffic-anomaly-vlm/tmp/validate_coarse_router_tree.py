from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.utils import write_json
from src.eval.validation_support import list_frame_paths, parse_frame_id, parse_manifest_intervals, robust_unit_scale, summarize_gt_overlap
from src.features.feature_builder import FeatureBuilder
from src.perception.detector_tracker import DetectorTracker
from src.perception.track_parser import parse_ultralytics_results
from src.perception.track_refiner import refine_track_ids
from src.proposals.adaptive_fine_tree import AdaptiveFineTreeConfig
from src.proposals.coarse_proposal import CoarseProposalConfig
from src.proposals.node_selector import NodeSelectorConfig
from src.proposals.routed_tree import build_routed_event_tree
from src.settings import load_settings
from src.vlm.coarse_router import CoarseRouterConfig


def build_coarse_cfg(cfg) -> CoarseProposalConfig:
    coarse_cfg = getattr(getattr(cfg, "components", None), "coarse_proposal", None)
    high_z = getattr(coarse_cfg, "tree_high_z", [0.9]) if coarse_cfg is not None else [0.9]
    if not isinstance(high_z, (list, tuple)):
        high_z = [0.9]
    return CoarseProposalConfig(
        interval_high_z=float(getattr(coarse_cfg, "interval_high_z", 0.85)) if coarse_cfg is not None else 0.85,
        interval_peak_gap=int(getattr(coarse_cfg, "interval_peak_gap", 4)) if coarse_cfg is not None else 4,
        interval_peak_expand=tuple(getattr(coarse_cfg, "interval_peak_expand", [10, 18])) if coarse_cfg is not None else (10, 18),
        interval_min_span_len=int(getattr(coarse_cfg, "interval_min_span_len", 12)) if coarse_cfg is not None else 12,
        interval_merge_gap=int(getattr(coarse_cfg, "interval_merge_gap", 10)) if coarse_cfg is not None else 10,
        tree_max_depth=int(getattr(coarse_cfg, "tree_max_depth", 1)) if coarse_cfg is not None else 1,
        tree_min_span_len=int(getattr(coarse_cfg, "tree_min_span_len", 8)) if coarse_cfg is not None else 8,
        tree_split_min_len=int(getattr(coarse_cfg, "tree_split_min_len", 20)) if coarse_cfg is not None else 20,
        tree_peak_gap=int(getattr(coarse_cfg, "tree_peak_gap", 4)) if coarse_cfg is not None else 4,
        tree_peak_expand=tuple(getattr(coarse_cfg, "tree_peak_expand", [4, 8])) if coarse_cfg is not None else (4, 8),
        tree_merge_gap=int(getattr(coarse_cfg, "tree_merge_gap", 4)) if coarse_cfg is not None else 4,
        tree_high_z=tuple(float(v) for v in high_z),
        merge_iou=float(getattr(coarse_cfg, "merge_iou", 0.40)) if coarse_cfg is not None else 0.40,
        merge_center_gap=int(getattr(coarse_cfg, "merge_center_gap", 20)) if coarse_cfg is not None else 20,
        max_proposals=int(getattr(coarse_cfg, "max_proposals", 8)) if coarse_cfg is not None else 8,
    )


def build_router_cfg(cfg, args: argparse.Namespace) -> CoarseRouterConfig:
    router_cfg = getattr(getattr(cfg, "components", None), "coarse_router", None)
    return CoarseRouterConfig(
        enabled=bool(args.enable_router_vlm or getattr(router_cfg, "enabled", False)),
        mode="vlm" if bool(args.enable_router_vlm) else str(getattr(router_cfg, "mode", "heuristic")) if router_cfg is not None else "heuristic",
        max_nodes=int(getattr(router_cfg, "max_nodes", 4)) if router_cfg is not None else 4,
        max_new_tokens=int(getattr(router_cfg, "max_new_tokens", 256)) if router_cfg is not None else 256,
        max_image_size=int(getattr(router_cfg, "max_image_size", 640)) if router_cfg is not None else 640,
        min_confidence=float(getattr(router_cfg, "min_confidence", 0.35)) if router_cfg is not None else 0.35,
        model_path=str(getattr(cfg.vlm, "model_path", "")),
        device=str(getattr(cfg.vlm, "device", "cuda")),
        dtype=str(getattr(cfg.vlm, "dtype", "float16")),
    )


def build_fine_cfg(cfg) -> AdaptiveFineTreeConfig:
    fine_cfg = getattr(getattr(cfg, "components", None), "fine_tree", None)
    return AdaptiveFineTreeConfig(
        min_output_span=int(getattr(fine_cfg, "min_output_span", 8)) if fine_cfg is not None else 8,
        focus_region_pad=float(getattr(fine_cfg, "focus_region_pad", 0.10)) if fine_cfg is not None else 0.10,
        localized_high_z=tuple(float(v) for v in getattr(fine_cfg, "localized_high_z", [0.8, 0.45])) if fine_cfg is not None else (0.8, 0.45),
        localized_split_min_len=int(getattr(fine_cfg, "localized_split_min_len", 12)) if fine_cfg is not None else 12,
        localized_peak_gap=int(getattr(fine_cfg, "localized_peak_gap", 3)) if fine_cfg is not None else 3,
        localized_peak_expand=tuple(getattr(fine_cfg, "localized_peak_expand", [2, 5])) if fine_cfg is not None else (2, 5),
        localized_merge_gap=int(getattr(fine_cfg, "localized_merge_gap", 3)) if fine_cfg is not None else 3,
        multi_stage_high_z=tuple(float(v) for v in getattr(fine_cfg, "multi_stage_high_z", [0.75, 0.42])) if fine_cfg is not None else (0.75, 0.42),
        multi_stage_split_min_len=int(getattr(fine_cfg, "multi_stage_split_min_len", 10)) if fine_cfg is not None else 10,
        multi_stage_peak_gap=int(getattr(fine_cfg, "multi_stage_peak_gap", 3)) if fine_cfg is not None else 3,
        multi_stage_peak_expand=tuple(getattr(fine_cfg, "multi_stage_peak_expand", [2, 4])) if fine_cfg is not None else (2, 4),
        multi_stage_merge_gap=int(getattr(fine_cfg, "multi_stage_merge_gap", 2)) if fine_cfg is not None else 2,
        uncertain_high_z=tuple(float(v) for v in getattr(fine_cfg, "uncertain_high_z", [0.9, 0.55])) if fine_cfg is not None else (0.9, 0.55),
        uncertain_split_min_len=int(getattr(fine_cfg, "uncertain_split_min_len", 16)) if fine_cfg is not None else 16,
        uncertain_peak_gap=int(getattr(fine_cfg, "uncertain_peak_gap", 4)) if fine_cfg is not None else 4,
        uncertain_peak_expand=tuple(getattr(fine_cfg, "uncertain_peak_expand", [4, 8])) if fine_cfg is not None else (4, 8),
        uncertain_merge_gap=int(getattr(fine_cfg, "uncertain_merge_gap", 4)) if fine_cfg is not None else 4,
    )


def build_selector_cfg(cfg) -> NodeSelectorConfig:
    selector_cfg = getattr(getattr(cfg, "components", None), "node_selector", None)
    return NodeSelectorConfig(
        top_k=int(getattr(selector_cfg, "top_k", 6)) if selector_cfg is not None else 6,
        min_node_len=int(getattr(selector_cfg, "min_node_len", 8)) if selector_cfg is not None else 8,
        overlap_iou=float(getattr(selector_cfg, "overlap_iou", 0.85)) if selector_cfg is not None else 0.85,
        gate_floor=float(getattr(selector_cfg, "gate_floor", 0.55)) if selector_cfg is not None else 0.55,
        reference_span=int(getattr(selector_cfg, "reference_span", 96)) if selector_cfg is not None else 96,
    )


def _plot_scores(
    frame_ids: np.ndarray,
    gt_intervals: list[list[int]],
    track_scores: np.ndarray,
    object_scores: np.ndarray,
    seed_scores: np.ndarray,
    coarse_proposals: list,
    selected_nodes: list,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    def _shade(ax):
        for s, e in gt_intervals:
            ax.axvspan(s, e, color="red", alpha=0.08)

    ax = axes[0]
    ax.plot(frame_ids, track_scores, label="track_risk", linewidth=1.4)
    ax.plot(frame_ids, object_scores, label="object_prior", linewidth=1.2)
    _shade(ax)
    ax.set_ylabel("Track/Object")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(frame_ids, seed_scores, color="#fb8c00", linewidth=1.4, label="seed_score")
    for proposal in coarse_proposals:
        ax.axvspan(proposal.start_frame, proposal.end_frame, color="#90caf9", alpha=0.18)
        ax.scatter([proposal.peak_frame], [proposal.seed_peak], color="black", s=16, zorder=5)
        ax.text(proposal.start_frame, min(1.02, proposal.seed_peak + 0.05), proposal.proposal_id, fontsize=7)
    _shade(ax)
    ax.set_ylabel("Coarse")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(frame_ids, seed_scores, color="#9e9e9e", linewidth=1.0, label="seed_score")
    for node in selected_nodes:
        ax.axvspan(node.start_frame, node.end_frame, color="#66bb6a", alpha=0.18)
        ax.scatter([node.peak_frame], [node.eventness_peak], color="black", s=18, zorder=5)
        ax.text(node.start_frame, min(1.02, node.eventness_peak + 0.05), node.node_id, fontsize=7)
    _shade(ax)
    ax.set_ylabel("Selected")
    ax.set_xlabel("Frame ID")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _plot_span_timeline(total_frames: int, gt_intervals: list[list[int]], rows: list[tuple[str, int, int, str]], save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig_h = max(3.0, 0.5 * max(1, len(rows)) + 1.8)
    fig, ax = plt.subplots(1, 1, figsize=(15, fig_h))

    for s, e in gt_intervals:
        ax.axvspan(s, e, color="red", alpha=0.08)

    for idx, (label, start_frame, end_frame, color) in enumerate(rows):
        ax.hlines(idx, start_frame, end_frame, colors=color, linewidth=7, alpha=0.9)
        ax.text(max(0, start_frame), idx + 0.12, label, fontsize=8)

    ax.set_xlim(0, max(1, int(total_frames)))
    ax.set_ylim(-0.8, max(0.8, len(rows) - 0.2))
    ax.set_yticks([])
    ax.set_xlabel("Frame ID")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _save_node_montage(images: list[np.ndarray], frame_ids: list[int], nodes: list, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not nodes:
        blank = np.full((160, 840, 3), 235, dtype=np.uint8)
        cv2.putText(blank, "No selected nodes", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.imwrite(str(save_path), blank)
        return

    frame_index = {int(fid): idx for idx, fid in enumerate(frame_ids)}
    rows: list[np.ndarray] = []
    for node in nodes:
        picks = [node.start_frame, node.peak_frame, node.end_frame]
        tiles: list[np.ndarray] = []
        for frame_id in picks:
            idx = frame_index.get(int(frame_id))
            if idx is None:
                continue
            tile = cv2.resize(images[idx], (280, 160))
            cv2.putText(tile, f"frame={frame_id}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) < 3:
            tiles.append(np.full((160, 280, 3), 235, dtype=np.uint8))
        row = np.hstack(tiles)
        banner = np.full((32, row.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(
            banner,
            f"{node.node_id} | role={node.route_role or 'na'} | pattern={node.route_temporal_pattern or 'na'} | span=[{node.start_frame}, {node.end_frame}]",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.vstack([banner, row]))

    cv2.imwrite(str(save_path), np.vstack(rows))


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = load_settings(args.config)
    frames_dir = Path(args.frames_root) / args.video_id
    frame_paths = list_frame_paths(str(frames_dir))
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    frame_paths = frame_paths[:: max(1, int(args.frame_stride))]
    images: list[np.ndarray] = []
    frame_ids: list[int] = []
    for path in frame_paths:
        image = cv2.imread(path)
        if image is None:
            continue
        images.append(image)
        frame_ids.append(parse_frame_id(path))
    if not images:
        raise ValueError("No readable images after loading")

    gt_intervals = parse_manifest_intervals(args.manifest, args.video_id)
    frame_ids_np = np.array(frame_ids, dtype=np.int32)
    logging.info("Loaded %d frames for %s", len(images), args.video_id)

    ref_cfg = getattr(getattr(cfg, "perception", None), "track_refiner", None)
    refine_max_gap = int(getattr(ref_cfg, "max_gap", 8)) if ref_cfg is not None else 8
    refine_max_center_dist = float(getattr(ref_cfg, "max_center_dist", 0.06)) if ref_cfg is not None else 0.06
    refine_max_size_ratio = float(getattr(ref_cfg, "max_size_ratio", 2.5)) if ref_cfg is not None else 2.5

    detector = DetectorTracker(
        model_path=cfg.perception.yolo_model,
        tracker=cfg.perception.tracker,
        conf=cfg.perception.conf,
        iou=cfg.perception.iou,
        classes=list(cfg.perception.classes),
    )

    raw_tracks_per_frame: list[list] = []
    for image, frame_id in zip(images, frame_ids_np):
        results = detector.track_frame(image, persist=True)
        raw_tracks_per_frame.append(parse_ultralytics_results(results, int(frame_id)))

    refined_tracks_per_frame, id_mapping = refine_track_ids(
        raw_tracks_per_frame,
        max_frame_gap=refine_max_gap,
        max_center_dist=refine_max_center_dist,
        max_size_ratio=refine_max_size_ratio,
    )

    feature_builder = FeatureBuilder(
        window_size=max(2, int(args.window_size)),
        window_step=1,
        track_fit_degree=max(1, int(args.track_fit_degree)),
        track_history_len=int(args.track_history_len) if int(args.track_history_len) > 0 else None,
    )
    track_scores: list[float] = []
    object_scores: list[float] = []
    scene_scores: list[float] = []
    feature_rows: list[dict[str, float]] = []
    for idx in range(len(images)):
        start = max(0, idx - int(args.window_size) + 1)
        frame_win = frame_ids[start : idx + 1]
        track_win = refined_tracks_per_frame[start : idx + 1]
        window_feature = feature_builder.build(frame_win, track_win)
        track_scores.append(float(window_feature.feature_dict.get("track_risk_score", window_feature.trigger_score)))
        object_scores.append(float(window_feature.feature_dict.get("object_prior_score", 0.0)))
        scene_scores.append(float(window_feature.feature_dict.get("scene_context_score", 0.0)))
        feature_rows.append({k: float(v) for k, v in window_feature.feature_dict.items()})

    track_arr = np.asarray(track_scores, dtype=np.float32)
    object_arr = np.asarray(object_scores, dtype=np.float32)
    scene_arr = np.asarray(scene_scores, dtype=np.float32)
    seed_scores = robust_unit_scale(track_arr + 0.5 * object_arr)

    routed = build_routed_event_tree(
        frame_ids=frame_ids,
        images=images,
        tracks_per_frame=refined_tracks_per_frame,
        track_scores=track_arr,
        object_scores=object_arr,
        seed_scores=seed_scores,
        output_dir=str(out_dir / "coarse_evidence"),
        coarse_cfg=build_coarse_cfg(cfg),
        router_cfg=build_router_cfg(cfg, args),
        fine_cfg=build_fine_cfg(cfg),
        selector_cfg=build_selector_cfg(cfg),
    )

    _plot_scores(
        frame_ids=frame_ids_np,
        gt_intervals=gt_intervals,
        track_scores=track_arr,
        object_scores=object_arr,
        seed_scores=seed_scores,
        coarse_proposals=routed.coarse_proposals,
        selected_nodes=routed.selected_nodes,
        save_path=out_dir / "scores_overview.png",
    )
    coarse_rows = [(p.proposal_id, p.start_frame, p.end_frame, "#1e88e5") for p in routed.coarse_proposals]
    _plot_span_timeline(
        total_frames=int(frame_ids_np[-1]) if frame_ids_np.size else 0,
        gt_intervals=gt_intervals,
        rows=coarse_rows,
        save_path=out_dir / "coarse_proposals_timeline.png",
        title="Coarse Proposals",
    )
    fine_rows = [(n.node_id, n.start_frame, n.end_frame, "#43a047") for n in routed.fine_nodes]
    _plot_span_timeline(
        total_frames=int(frame_ids_np[-1]) if frame_ids_np.size else 0,
        gt_intervals=gt_intervals,
        rows=fine_rows,
        save_path=out_dir / "fine_nodes_timeline.png",
        title="Fine Nodes",
    )
    _save_node_montage(images, frame_ids, routed.selected_nodes, out_dir / "selected_nodes_montage.jpg")

    route_json = {proposal_id: decision.to_dict() for proposal_id, decision in routed.route_decisions.items()}
    write_json(str(out_dir / "router_decisions.json"), route_json)
    write_json(str(out_dir / "feature_rows.json"), feature_rows)

    summary = {
        "video_id": args.video_id,
        "num_frames": int(len(frame_ids)),
        "gt_intervals": gt_intervals,
        "track_refine": {
            "max_gap": int(refine_max_gap),
            "max_center_dist": float(refine_max_center_dist),
            "max_size_ratio": float(refine_max_size_ratio),
            "refined_id_count": int(sum(1 for k, v in id_mapping.items() if int(k) != int(v))),
        },
        "coarse_proposals": [proposal.to_dict() for proposal in routed.coarse_proposals],
        "route_decisions": route_json,
        "fine_nodes": [node.to_dict() for node in routed.fine_nodes],
        "selected_nodes": [node.to_dict() for node in routed.selected_nodes],
        "gt_overlap": summarize_gt_overlap(routed.selected_nodes, gt_intervals),
        "paths": {
            "scores_overview": str(out_dir / "scores_overview.png"),
            "coarse_proposals_timeline": str(out_dir / "coarse_proposals_timeline.png"),
            "fine_nodes_timeline": str(out_dir / "fine_nodes_timeline.png"),
            "selected_nodes_montage": str(out_dir / "selected_nodes_montage.jpg"),
            "router_decisions": str(out_dir / "router_decisions.json"),
        },
        "score_stats": {
            "track_mean": float(np.mean(track_arr)) if track_arr.size else 0.0,
            "object_mean": float(np.mean(object_arr)) if object_arr.size else 0.0,
            "scene_mean": float(np.mean(scene_arr)) if scene_arr.size else 0.0,
            "seed_mean": float(np.mean(seed_scores)) if seed_scores.size else 0.0,
        },
    }
    write_json(str(out_dir / "summary.json"), summary)
    logging.info("Done. Outputs saved to %s", out_dir)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate coarse proposal + coarse router + adaptive fine tree")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--frames-root", default="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal")
    parser.add_argument("--video-id", default="v1")
    parser.add_argument("--manifest", default="/nvme2/VAD_yemao/data/TU-DAT-25/abnormal.txt")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--track-fit-degree", type=int, default=2)
    parser.add_argument("--track-history-len", type=int, default=0)
    parser.add_argument("--enable-router-vlm", action="store_true")
    parser.add_argument("--out-dir", default="outputs/debug/coarse_router_tree")
    return parser.parse_args()


if __name__ == "__main__":
    run(build_args())
