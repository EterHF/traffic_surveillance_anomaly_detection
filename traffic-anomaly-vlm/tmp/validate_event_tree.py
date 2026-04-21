from __future__ import annotations

import argparse
import csv
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
from src.features.eventness_features import EventnessFeatureConfig, EventnessFeatureExtractor
from src.features.feature_builder import FeatureBuilder
from src.perception.detector_tracker import DetectorTracker
from src.perception.track_parser import parse_ultralytics_results
from src.perception.track_refiner import refine_track_ids
from src.proposals.event_tree import EventTreeConfig, EventNode, build_event_tree, flatten_event_tree
from src.proposals.node_selector import NodeSelectorConfig, select_salient_nodes
from src.settings import load_settings


def list_frame_paths(frame_dir: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in Path(frame_dir).iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return [str(p) for p in files]


def parse_frame_id(path: str) -> int:
    stem = Path(path).stem
    if stem.isdigit():
        return int(stem)
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def parse_manifest_intervals(manifest_path: str, video_id: str) -> list[list[int]]:
    p = Path(manifest_path)
    if not p.exists() or not video_id:
        return []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if not parts or parts[0] != video_id:
                continue
            nums = [int(x) for x in parts[2:] if x.lstrip("-").isdigit()]
            intervals: list[list[int]] = []
            for i in range(0, len(nums) - 1, 2):
                a, b = nums[i], nums[i + 1]
                if b < a:
                    a, b = b, a
                intervals.append([a, b])
            return intervals
    return []


def load_track_scores_proxy(csv_path: str, frame_ids: list[int]) -> tuple[list[float], list[float], list[float]]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"timeseries csv not found: {csv_path}")

    rows = list(csv.DictReader(p.open("r", encoding="utf-8")))
    by_frame = {int(row["frame_id"]): row for row in rows if str(row.get("frame_id", "")).isdigit()}
    track_risk: list[float] = []
    object_prior: list[float] = []
    scene_context: list[float] = []
    for fid in frame_ids:
        row = by_frame.get(int(fid), {})
        track_risk.append(float(row.get("score_track", 0.0) or 0.0))
        object_prior.append(float(row.get("score_scene", 0.0) or 0.0))
        scene_context.append(float(row.get("score_scene", 0.0) or 0.0))
    return track_risk, object_prior, scene_context


def build_eventness_cfg(cfg, args: argparse.Namespace) -> EventnessFeatureConfig:
    feat_cfg = getattr(getattr(cfg, "features", None), "eventness", None)
    clip_cfg = getattr(getattr(cfg, "features", None), "clip", None)
    raft_cfg = getattr(getattr(cfg, "features", None), "raft", None)
    return EventnessFeatureConfig(
        use_clip=bool(args.use_clip or getattr(feat_cfg, "use_clip", False)),
        use_raft=bool(args.use_raft or getattr(feat_cfg, "use_raft", False)),
        use_lowres=not bool(args.disable_lowres) if args.disable_lowres else bool(getattr(feat_cfg, "use_lowres", True)),
        use_track_layout=not bool(args.disable_track_layout) if args.disable_track_layout else bool(getattr(feat_cfg, "use_track_layout", True)),
        lowres_size=int(getattr(feat_cfg, "lowres_size", 32)),
        occupancy_grid=int(getattr(feat_cfg, "occupancy_grid", 4)),
        smooth_window=int(getattr(feat_cfg, "smooth_window", 9)),
        smooth_polyorder=int(getattr(feat_cfg, "smooth_polyorder", 2)),
        clip_weight=float(getattr(feat_cfg, "clip_weight", 0.45)),
        raft_weight=float(getattr(feat_cfg, "raft_weight", 0.15)),
        lowres_weight=float(getattr(feat_cfg, "lowres_weight", 0.20)),
        track_weight=float(getattr(feat_cfg, "track_weight", 0.20)),
        signal_weight=float(getattr(feat_cfg, "signal_weight", 0.30)),
        clip_model_name=str(getattr(clip_cfg, "model_name", "openai/clip-vit-base-patch32")),
        clip_model_revision=getattr(clip_cfg, "model_revision", None),
        clip_device=str(getattr(clip_cfg, "device", "cuda")),
        clip_batch_size=int(getattr(clip_cfg, "batch_size", 16)),
        clip_resize_long_side=int(getattr(clip_cfg, "resize_long_side", 640)),
        clip_resize_interpolation=str(getattr(clip_cfg, "resize_interpolation", "area")),
        raft_root=str(getattr(raft_cfg, "root", "")),
        raft_weights=str(getattr(raft_cfg, "weights", "")),
        raft_device=str(getattr(raft_cfg, "device", "cuda")),
        raft_iters=int(getattr(raft_cfg, "iters", 12)),
        raft_resize_long_side=int(getattr(raft_cfg, "resize_long_side", 640)),
        raft_resize_interpolation=str(getattr(raft_cfg, "resize_interpolation", "area")),
        raft_flow_proj_dim=int(getattr(raft_cfg, "flow_proj_dim", 128)),
        raft_flow_proj_seed=int(getattr(raft_cfg, "flow_proj_seed", 42)),
    )


def build_event_tree_cfg(cfg, args: argparse.Namespace) -> EventTreeConfig:
    tree_cfg = getattr(getattr(cfg, "components", None), "event_tree", None)
    high_z = getattr(tree_cfg, "per_level_high_z", [1.2, 0.8]) if tree_cfg is not None else [1.2, 0.8]
    if not isinstance(high_z, (list, tuple)):
        high_z = [1.2, 0.8]
    if getattr(args, "tree_high_z", None):
        high_z = [float(v) for v in args.tree_high_z]
    peak_expand = tuple(getattr(tree_cfg, "peak_expand", [10, 18])) if tree_cfg is not None else (10, 18)
    if getattr(args, "tree_peak_expand", None):
        peak_expand = tuple(int(v) for v in args.tree_peak_expand)
    return EventTreeConfig(
        max_depth=int(getattr(tree_cfg, "max_depth", 2)) if tree_cfg is not None else 2,
        min_span_len=int(args.tree_min_span_len or getattr(tree_cfg, "min_span_len", 12)) if tree_cfg is not None else int(args.tree_min_span_len or 12),
        split_min_len=int(args.tree_split_min_len or getattr(tree_cfg, "split_min_len", 36)) if tree_cfg is not None else int(args.tree_split_min_len or 36),
        peak_gap=int(args.tree_peak_gap or getattr(tree_cfg, "peak_gap", 6)) if tree_cfg is not None else int(args.tree_peak_gap or 6),
        peak_expand=peak_expand,
        merge_gap=int(args.tree_merge_gap or getattr(tree_cfg, "merge_gap", 14)) if tree_cfg is not None else int(args.tree_merge_gap or 14),
        per_level_high_z=tuple(float(v) for v in high_z),
        use_savgol_filter=bool(getattr(tree_cfg, "use_savgol_filter", False)) if tree_cfg is not None else False,
    )


def build_selector_cfg(cfg, args: argparse.Namespace) -> NodeSelectorConfig:
    selector_cfg = getattr(getattr(cfg, "components", None), "node_selector", None)
    return NodeSelectorConfig(
        top_k=int(args.top_k or getattr(selector_cfg, "top_k", 4)) if selector_cfg is not None else int(args.top_k or 4),
        min_node_len=int(args.selector_min_node_len or getattr(selector_cfg, "min_node_len", 10)) if selector_cfg is not None else int(args.selector_min_node_len or 10),
        overlap_iou=float(args.selector_overlap_iou if args.selector_overlap_iou >= 0.0 else getattr(selector_cfg, "overlap_iou", 0.65)) if selector_cfg is not None else float(args.selector_overlap_iou if args.selector_overlap_iou >= 0.0 else 0.65),
        gate_floor=float(args.selector_gate_floor if args.selector_gate_floor >= 0.0 else getattr(selector_cfg, "gate_floor", 0.35)) if selector_cfg is not None else float(args.selector_gate_floor if args.selector_gate_floor >= 0.0 else 0.35),
        reference_span=int(args.selector_reference_span or getattr(selector_cfg, "reference_span", 32)) if selector_cfg is not None else int(args.selector_reference_span or 32),
    )


def _interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0, e - s + 1)
    union = max(1, (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter)
    return float(inter / union)


def robust_unit_scale(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    if values.size == 0:
        return values
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = max(1e-6, 1.4826 * mad)
    z = np.maximum((values - med) / sigma, 0.0)
    hi = max(float(np.percentile(z, 95)), 1e-6)
    return np.clip(z / hi, 0.0, 1.0)


def summarize_gt_overlap(selected_nodes: list[EventNode], gt_intervals: list[list[int]]) -> dict:
    selected = [(int(n.start_frame), int(n.end_frame)) for n in selected_nodes]
    gt_summary: list[dict] = []
    hit = 0
    for s, e in gt_intervals:
        best_iou = 0.0
        for ps, pe in selected:
            best_iou = max(best_iou, _interval_iou((s, e), (ps, pe)))
        covered = best_iou > 0.0
        hit += int(covered)
        gt_summary.append(
            {
                "start_frame": int(s),
                "end_frame": int(e),
                "covered": bool(covered),
                "best_iou": float(best_iou),
            }
        )
    total = len(gt_intervals)
    return {
        "num_gt_intervals": int(total),
        "num_covered_gt_intervals": int(hit),
        "covered_ratio": float(hit / max(1, total)),
        "items": gt_summary,
    }


def plot_score_overview(
    frame_ids: np.ndarray,
    gt_intervals: list[list[int]],
    track_risk: np.ndarray,
    object_prior: np.ndarray,
    scene_context: np.ndarray,
    eventness_result,
    tree_seed_scores: np.ndarray,
    selected_nodes: list[EventNode],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    def _shade(ax):
        for s, e in gt_intervals:
            ax.axvspan(s, e, color="red", alpha=0.10)
        for node in selected_nodes:
            ax.axvspan(node.start_frame, node.end_frame, color="green", alpha=0.08)

    ax = axes[0]
    ax.plot(frame_ids, track_risk, label="track_risk", linewidth=1.4)
    ax.plot(frame_ids, object_prior, label="object_prior", linewidth=1.2)
    ax.plot(frame_ids, scene_context, label="scene_context", linewidth=1.0)
    _shade(ax)
    ax.set_ylabel("Track Branch")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for name, values in sorted(eventness_result.cue_scores.items()):
        ax.plot(frame_ids, values, label=name, linewidth=1.1)
    _shade(ax)
    ax.set_ylabel("Eventness Cues")
    if eventness_result.cue_scores:
        ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(frame_ids, eventness_result.frame_scores, color="#6a1b9a", linewidth=1.8, label="eventness_fused")
    ax.plot(frame_ids, tree_seed_scores, color="#fb8c00", linewidth=1.2, label="tree_seed")
    _shade(ax)
    ax.set_ylabel("Eventness")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[3]
    ax.plot(frame_ids, tree_seed_scores, color="#bdbdbd", linewidth=1.1, label="tree_seed")
    for node in selected_nodes:
        ax.axvspan(node.start_frame, node.end_frame, color="green", alpha=0.12)
        ax.scatter([node.peak_frame], [node.eventness_peak], color="black", s=22, zorder=5)
        ax.text(node.start_frame, min(1.02, node.eventness_peak + 0.05), node.node_id, fontsize=8)
    for s, e in gt_intervals:
        ax.axvspan(s, e, color="red", alpha=0.08)
    ax.set_ylabel("Selected")
    ax.set_xlabel("Frame ID")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_event_tree_timeline(
    total_frames: int,
    gt_intervals: list[list[int]],
    all_nodes: list[EventNode],
    selected_nodes: list[EventNode],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    max_level = max((n.level for n in all_nodes), default=1)
    plt.figure(figsize=(15, 2.8 + max_level * 1.2))
    selected_ids = {n.node_id for n in selected_nodes}

    for s, e in gt_intervals:
        plt.axvspan(s, e, color="red", alpha=0.08)

    for node in all_nodes:
        y = max_level - node.level + 1
        color = "#2e7d32" if node.node_id in selected_ids else "#1e88e5"
        width = 5 if node.node_id in selected_ids else 3
        alpha = 0.95 if node.node_id in selected_ids else 0.55
        plt.hlines(y, node.start_frame, node.end_frame, colors=color, linewidth=width, alpha=alpha)
        plt.scatter([node.peak_frame], [y], color=color, s=22, zorder=5)
        plt.text(node.start_frame, y + 0.12, node.node_id, fontsize=8)

    plt.hlines(0.5, 0, total_frames, colors="#9e9e9e", linewidth=2)
    plt.xlim(0, max(1, total_frames))
    plt.yticks(list(range(1, max_level + 1)), [f"L{lvl}" for lvl in range(max_level, 0, -1)])
    plt.xlabel("Frame ID")
    plt.title("Hierarchical Event Tree")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def _draw_tracks_on_frame(frame: np.ndarray, frame_tracks: list) -> np.ndarray:
    vis = frame.copy()
    for t in frame_tracks:
        x1, y1, x2, y2 = map(int, t.bbox_xyxy)
        tid = int(t.track_id)
        color = ((tid * 37) % 255, (tid * 67) % 255, (tid * 97) % 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"id={t.track_id} {t.cls_name}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def save_selected_node_montage(
    images: list[np.ndarray],
    frame_ids: list[int],
    tracks_per_frame: list[list],
    selected_nodes: list[EventNode],
    save_path: Path,
    thumb_size: tuple[int, int] = (280, 160),
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not selected_nodes:
        canvas = np.full((thumb_size[1], thumb_size[0] * 3, 3), 235, dtype=np.uint8)
        cv2.putText(canvas, "No selected event nodes", (20, thumb_size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(str(save_path), canvas)
        return

    frame_index = {int(fid): idx for idx, fid in enumerate(frame_ids)}
    rows: list[np.ndarray] = []
    for node in selected_nodes:
        picks = [node.start_frame, node.peak_frame, node.end_frame]
        tiles: list[np.ndarray] = []
        for fid in picks:
            idx = frame_index.get(int(fid))
            if idx is None:
                continue
            vis = _draw_tracks_on_frame(images[idx], tracks_per_frame[idx])
            vis = cv2.resize(vis, thumb_size)
            cv2.putText(
                vis,
                f"frame={fid}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(vis)
        while len(tiles) < 3:
            tiles.append(np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8))
        row = np.hstack(tiles)
        banner = np.full((30, row.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(
            banner,
            f"{node.node_id} | fused={node.fused_score:.3f} | span=[{node.start_frame}, {node.end_frame}]",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.vstack([banner, row]))

    montage = np.vstack(rows)
    cv2.imwrite(str(save_path), montage)


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
    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        images.append(img)
        frame_ids.append(parse_frame_id(p))
    if not images:
        raise ValueError("No readable images after loading")

    gt_intervals = parse_manifest_intervals(args.manifest, args.video_id)
    frame_ids_np = np.array(frame_ids, dtype=np.int32)
    logging.info("Loaded %d frames for %s", len(images), args.video_id)

    ref_cfg = getattr(getattr(cfg, "perception", None), "track_refiner", None)
    refine_max_gap = int(getattr(ref_cfg, "max_gap", 8)) if ref_cfg is not None else 8
    refine_max_center_dist = float(getattr(ref_cfg, "max_center_dist", 0.06)) if ref_cfg is not None else 0.06
    refine_max_size_ratio = float(getattr(ref_cfg, "max_size_ratio", 2.5)) if ref_cfg is not None else 2.5

    refined_tracks_per_frame: list[list] = []
    id_mapping: dict[int, int] = {}
    track_risk_scores: list[float] = []
    object_prior_scores: list[float] = []
    scene_context_scores: list[float] = []
    feature_rows: list[dict[str, float]] = []

    use_proxy = False
    proxy_csv = str(args.timeseries_csv or "").strip()
    if not proxy_csv:
        auto_proxy = Path(cfg.output.output_dir) / "test" / args.video_id / "timeseries.csv"
        if auto_proxy.exists():
            proxy_csv = str(auto_proxy)

    try:
        detector = DetectorTracker(
            model_path=cfg.perception.yolo_model,
            tracker=cfg.perception.tracker,
            conf=cfg.perception.conf,
            iou=cfg.perception.iou,
            classes=list(cfg.perception.classes),
        )

        all_tracks_per_frame: list[list] = []
        for i, img in enumerate(images):
            results = detector.track_frame(img, persist=True)
            tracks = parse_ultralytics_results(results, int(frame_ids_np[i]))
            all_tracks_per_frame.append(tracks)

        refined_tracks_per_frame, id_mapping = refine_track_ids(
            all_tracks_per_frame,
            max_frame_gap=refine_max_gap,
            max_center_dist=refine_max_center_dist,
            max_size_ratio=refine_max_size_ratio,
        )
        logging.info("Track refinement changed %d ids", sum(1 for k, v in id_mapping.items() if int(k) != int(v)))

        feature_builder = FeatureBuilder(
            window_size=max(2, int(args.window_size)),
            window_step=1,
            track_fit_degree=max(1, int(args.track_fit_degree)),
            track_history_len=int(args.track_history_len) if int(args.track_history_len) > 0 else None,
        )
        for i in range(len(images)):
            start = max(0, i - int(args.window_size) + 1)
            frame_win = frame_ids[start : i + 1]
            track_win = refined_tracks_per_frame[start : i + 1]
            wf = feature_builder.build(frame_win, track_win)
            track_risk_scores.append(float(wf.trigger_score))
            object_prior_scores.append(float(wf.feature_dict.get("object_prior_score", 0.0)))
            scene_context_scores.append(float(wf.feature_dict.get("scene_context_score", 0.0)))
            feature_rows.append({k: float(v) for k, v in wf.feature_dict.items()})
    except ImportError as e:
        if not proxy_csv:
            raise RuntimeError(
                "DetectorTracker is unavailable and no timeseries proxy was provided. "
                "Install ultralytics or pass --timeseries-csv."
            ) from e
        use_proxy = True
        logging.warning("Falling back to proxy scores from %s because detector init failed: %s", proxy_csv, e)
        refined_tracks_per_frame = [[] for _ in images]
        track_risk_scores, object_prior_scores, scene_context_scores = load_track_scores_proxy(proxy_csv, frame_ids)
        feature_rows = [
            {
                "track_risk_score": float(track_risk_scores[i]),
                "object_prior_score": float(object_prior_scores[i]),
                "scene_context_score": float(scene_context_scores[i]),
            }
            for i in range(len(frame_ids))
        ]

    eventness_cfg = build_eventness_cfg(cfg, args)
    if use_proxy:
        eventness_cfg.use_track_layout = False
    logging.info(
        "Eventness branch: lowres=%s track_layout=%s clip=%s raft=%s",
        eventness_cfg.use_lowres,
        eventness_cfg.use_track_layout,
        eventness_cfg.use_clip,
        eventness_cfg.use_raft,
    )
    auxiliary_signals = {
        "track_branch_eventness": (
            np.asarray(track_risk_scores, dtype=np.float32)
            + 0.5 * np.asarray(object_prior_scores, dtype=np.float32)
        )
    }
    eventness_result = EventnessFeatureExtractor(eventness_cfg).compute(
        images,
        refined_tracks_per_frame,
        auxiliary_signals=auxiliary_signals,
    )
    track_seed_scores = robust_unit_scale(
        np.asarray(track_risk_scores, dtype=np.float32) + 0.5 * np.asarray(object_prior_scores, dtype=np.float32)
    )
    tree_seed_scores = np.maximum(eventness_result.frame_scores, track_seed_scores).astype(np.float32)

    tree_cfg = build_event_tree_cfg(cfg, args)
    root = build_event_tree(frame_ids, tree_seed_scores, tree_cfg)
    all_nodes = flatten_event_tree(root, include_root=False)
    selector_cfg = build_selector_cfg(cfg, args)
    selected_nodes = select_salient_nodes(
        root,
        eventness_scores=tree_seed_scores,
        track_risk_scores=np.array(track_risk_scores, dtype=np.float32),
        object_prior_scores=np.array(object_prior_scores, dtype=np.float32),
        cfg=selector_cfg,
    )

    plot_score_overview(
        frame_ids=frame_ids_np,
        gt_intervals=gt_intervals,
        track_risk=np.array(track_risk_scores, dtype=np.float32),
        object_prior=np.array(object_prior_scores, dtype=np.float32),
        scene_context=np.array(scene_context_scores, dtype=np.float32),
        eventness_result=eventness_result,
        tree_seed_scores=tree_seed_scores,
        selected_nodes=selected_nodes,
        save_path=out_dir / "scores_overview.png",
    )
    plot_event_tree_timeline(
        total_frames=int(frame_ids_np[-1]) if frame_ids_np.size else 0,
        gt_intervals=gt_intervals,
        all_nodes=all_nodes,
        selected_nodes=selected_nodes,
        save_path=out_dir / "event_tree_timeline.png",
    )
    save_selected_node_montage(
        images=images,
        frame_ids=frame_ids,
        tracks_per_frame=refined_tracks_per_frame,
        selected_nodes=selected_nodes,
        save_path=out_dir / "selected_nodes_montage.jpg",
    )

    summary = {
        "video_id": args.video_id,
        "num_frames": int(len(frame_ids)),
        "gt_intervals": gt_intervals,
        "eventness_weights": {k: float(v) for k, v in eventness_result.weights.items()},
        "tree_seed": {
            "source": "max(eventness_fused, robust(track_risk + 0.5*object_prior))",
        },
        "eventness_cfg": {
            "use_clip": bool(eventness_cfg.use_clip),
            "use_raft": bool(eventness_cfg.use_raft),
            "use_lowres": bool(eventness_cfg.use_lowres),
            "use_track_layout": bool(eventness_cfg.use_track_layout),
            "lowres_size": int(eventness_cfg.lowres_size),
            "occupancy_grid": int(eventness_cfg.occupancy_grid),
        },
        "tree_cfg": {
            "max_depth": int(tree_cfg.max_depth),
            "min_span_len": int(tree_cfg.min_span_len),
            "split_min_len": int(tree_cfg.split_min_len),
            "peak_gap": int(tree_cfg.peak_gap),
            "peak_expand": [int(tree_cfg.peak_expand[0]), int(tree_cfg.peak_expand[1])],
            "merge_gap": int(tree_cfg.merge_gap),
            "per_level_high_z": [float(v) for v in tree_cfg.per_level_high_z],
        },
        "selector_cfg": {
            "top_k": int(selector_cfg.top_k),
            "min_node_len": int(selector_cfg.min_node_len),
            "overlap_iou": float(selector_cfg.overlap_iou),
            "gate_floor": float(selector_cfg.gate_floor),
            "reference_span": int(selector_cfg.reference_span),
        },
        "track_refine": {
            "max_gap": int(refine_max_gap),
            "max_center_dist": float(refine_max_center_dist),
            "max_size_ratio": float(refine_max_size_ratio),
            "refined_id_count": int(sum(1 for k, v in id_mapping.items() if int(k) != int(v))),
            "used_proxy_scores": bool(use_proxy),
            "proxy_csv": str(proxy_csv) if proxy_csv else "",
        },
        "all_nodes": [n.to_dict() for n in all_nodes],
        "selected_nodes": [n.to_dict() for n in selected_nodes],
        "gt_overlap": summarize_gt_overlap(selected_nodes, gt_intervals),
        "paths": {
            "scores_overview": str(out_dir / "scores_overview.png"),
            "event_tree_timeline": str(out_dir / "event_tree_timeline.png"),
            "selected_nodes_montage": str(out_dir / "selected_nodes_montage.jpg"),
        },
    }
    write_json(str(out_dir / "summary.json"), summary)

    with (out_dir / "feature_rows.json").open("w", encoding="utf-8") as f:
        json.dump(feature_rows, f, ensure_ascii=False, indent=2)

    logging.info("Done. Validation outputs saved to %s", out_dir)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate event-aware pseudo-events and event tree selection")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--frames-root",
        default="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal",
        help="Path to root directory containing video frame directories",
    )
    parser.add_argument("--video-id", default="v1")
    parser.add_argument("--manifest", default="/nvme2/VAD_yemao/data/TU-DAT-25/abnormal.txt")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--track-fit-degree", type=int, default=2)
    parser.add_argument("--track-history-len", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--tree-min-span-len", type=int, default=0)
    parser.add_argument("--tree-split-min-len", type=int, default=0)
    parser.add_argument("--tree-peak-gap", type=int, default=0)
    parser.add_argument("--tree-peak-expand", nargs=2, type=int, default=None)
    parser.add_argument("--tree-merge-gap", type=int, default=0)
    parser.add_argument("--tree-high-z", nargs="*", type=float, default=None)
    parser.add_argument("--selector-min-node-len", type=int, default=0)
    parser.add_argument("--selector-overlap-iou", type=float, default=-1.0)
    parser.add_argument("--selector-gate-floor", type=float, default=-1.0)
    parser.add_argument("--selector-reference-span", type=int, default=0)
    parser.add_argument("--use-clip", action="store_true")
    parser.add_argument("--use-raft", action="store_true")
    parser.add_argument("--disable-lowres", action="store_true")
    parser.add_argument("--disable-track-layout", action="store_true")
    parser.add_argument("--timeseries-csv", default="")
    parser.add_argument("--out-dir", default="outputs/debug/event_tree")
    return parser.parse_args()


if __name__ == "__main__":
    run(build_args())
