from __future__ import annotations

from pathlib import Path

import numpy as np

from src.schemas import EventNode


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


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
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
            best_iou = max(best_iou, interval_iou((s, e), (ps, pe)))
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
