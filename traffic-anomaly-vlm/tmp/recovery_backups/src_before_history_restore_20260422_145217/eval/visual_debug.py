from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.schemas import FinalResult, WindowFeature


def save_window_score_curve(windows: list[WindowFeature], save_path: str) -> str:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    # Use original video frame ids so all visualizations share the same time axis.
    xs = [(w.start_frame + w.end_frame) / 2.0 for w in windows]
    ys = [w.trigger_score for w in windows]

    plt.figure(figsize=(10, 4))
    if xs:
        plt.plot(xs, ys, color="#1976d2", linewidth=2)
    else:
        plt.text(0.5, 0.5, "No windows generated", ha="center", va="center")
    plt.title("Window Trigger Score Curve")
    plt.xlabel("Frame ID")
    plt.ylabel("Trigger Score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def save_event_timeline(
    total_frames: int,
    proposals: list,
    save_path: str,
) -> str:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(total_frames, 1)

    plt.figure(figsize=(12, 2.8))
    plt.hlines(1, 0, total_frames, colors="#999999", linewidth=2)

    for p in proposals:
        y = 1
        plt.hlines(y, p.start_frame, p.end_frame, colors="#e53935", linewidth=8)
        plt.scatter([p.peak_frame], [y], color="#1e88e5", s=35, zorder=3)
        plt.text(p.start_frame, y + 0.1, p.event_id, fontsize=8)

    plt.xlim(0, total_frames)
    plt.yticks([])
    plt.xlabel("Frame ID")
    plt.title("Event Timeline")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def save_keyframe_montage(
    results: list[FinalResult],
    save_path: str,
    max_events: int = 6,
    per_event_frames: int = 4,
    thumb_w: int = 256,
    thumb_h: int = 144,
) -> str:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    selected = results[:max_events]
    rows: list[np.ndarray] = []

    for r in selected:
        imgs: list[np.ndarray] = []
        for fp in r.evidence.keyframe_paths[:per_event_frames]:
            im = cv2.imread(fp)
            if im is None:
                continue
            im = cv2.resize(im, (thumb_w, thumb_h))
            imgs.append(im)

        if not imgs:
            continue

        while len(imgs) < per_event_frames:
            imgs.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))

        row = np.hstack(imgs)
        banner = np.full((28, row.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(
            banner,
            f"{r.event_id} | {r.vlm_result.event_type} | conf={r.vlm_result.confidence:.2f}",
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.vstack([banner, row]))

    if not rows:
        canvas = np.full((thumb_h, thumb_w * per_event_frames, 3), 240, dtype=np.uint8)
        cv2.putText(canvas, "No keyframes available", (20, thumb_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(save_path, canvas)
        return save_path

    montage = np.vstack(rows)
    cv2.imwrite(save_path, montage)
    return save_path


def save_detection_preview_montage(
    sampled_frame_ids: list[int],
    sampled_frames: dict[int, np.ndarray | str],
    sampled_tracks: list[list],
    save_path: str,
    max_frames: int = 12,
    cols: int = 4,
    thumb_w: int = 320,
    thumb_h: int = 180,
) -> str:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Select frames across the entire sequence instead of only the beginning.
    if len(sampled_frame_ids) <= max_frames:
        chosen_ids = sampled_frame_ids
    else:
        sel_idx = np.linspace(0, len(sampled_frame_ids) - 1, max_frames, dtype=int).tolist()
        # Keep order stable while removing potential duplicates.
        sel_idx = list(dict.fromkeys(sel_idx))
        chosen_ids = [sampled_frame_ids[i] for i in sel_idx]

    tiles: list[np.ndarray] = []
    id_to_tracks = {fid: tracks for fid, tracks in zip(sampled_frame_ids, sampled_tracks)}

    # Build trajectory history up to each selected frame.
    chosen_set = set(chosen_ids)
    track_points: dict[int, list[tuple[int, int]]] = {}
    history_by_frame: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for fid, tracks in zip(sampled_frame_ids, sampled_tracks):
        for t in tracks:
            tid = int(t.track_id)
            track_points.setdefault(tid, []).append((int(t.cx), int(t.cy)))
        if fid in chosen_set:
            history_by_frame[fid] = {tid: pts.copy() for tid, pts in track_points.items()}

    for fid in chosen_ids:
        frame_ref = sampled_frames.get(fid)
        if isinstance(frame_ref, str):
            frame = cv2.imread(frame_ref)
        else:
            frame = frame_ref
        if frame is None:
            continue
        vis = frame.copy()

        # Draw trajectories first so current boxes stay visually on top.
        for tid, pts in history_by_frame.get(fid, {}).items():
            if len(pts) < 2:
                continue
            color = ((tid * 37) % 255, (tid * 67) % 255, (tid * 97) % 255)
            for p1, p2 in zip(pts[:-1], pts[1:]):
                cv2.line(vis, p1, p2, color, 2)

        for t in id_to_tracks.get(fid, []):
            x1, y1, x2, y2 = map(int, t.bbox_xyxy)
            tid = int(t.track_id)
            color = ((tid * 37) % 255, (tid * 67) % 255, (tid * 97) % 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                f"id={t.track_id} {t.cls_name} {t.score:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        vis = cv2.resize(vis, (thumb_w, thumb_h))
        cv2.putText(vis, f"frame={fid}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 255), 2, cv2.LINE_AA)
        tiles.append(vis)

    if not tiles:
        canvas = np.full((thumb_h, thumb_w, 3), 240, dtype=np.uint8)
        cv2.putText(canvas, "No sampled frames", (20, thumb_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(save_path, canvas)
        return save_path

    rows = []
    for i in range(0, len(tiles), cols):
        row_tiles = tiles[i : i + cols]
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))
        rows.append(np.hstack(row_tiles))

    montage = np.vstack(rows)
    cv2.imwrite(save_path, montage)
    return save_path
