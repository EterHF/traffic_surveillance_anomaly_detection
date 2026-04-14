from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.utils import ensure_dir, write_json
from src.eval.metrics import (
    average_precision_score_binary,
    binary_accuracy_at_threshold,
    equal_error_rate_binary,
    roc_auc_score_binary,
)
from src.pipeline.pure_vlm_baseline import PureVLMBaselinePipeline
from src.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure VLM baseline on frame-folder manifest")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True, help="txt manifest path")
    parser.add_argument(
        "--frames-root",
        default="",
        help="Root folder for frame dirs when manifest lines are like 'v1 total s e'. "
        "Default: <manifest_dir>/frames/abnormal",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="output root for per-video jsons and summary; default: <cfg.output.output_dir>/results/pure_vlm_baseline",
    )
    parser.add_argument("--threshold", type=float, default=None, help="override baseline threshold")
    parser.add_argument("--resume", action="store_true", help="skip videos with existing per-video json")
    parser.add_argument("--max-videos", type=int, default=0, help="limit number of videos to run (0 means all)")
    parser.add_argument("--window-size", type=int, default=0, help="override pure_vlm_baseline.window_size")
    parser.add_argument("--window-stride", type=int, default=0, help="override pure_vlm_baseline.window_stride")
    parser.add_argument(
        "--sampled-frames-per-chunk",
        type=int,
        default=0,
        help="override pure_vlm_baseline.sampled_frames_per_chunk",
    )
    parser.add_argument("--max-new-tokens", type=int, default=0, help="override vlm.max_new_tokens")
    parser.add_argument("--vlm-device", default="", help="override vlm.device, e.g. cuda or cpu")
    parser.add_argument("--vlm-dtype", default="", help="override vlm.dtype, e.g. float16")
    parser.add_argument("--max-image-size", type=int, default=0, help="override pure_vlm_baseline.max_image_size")
    parser.add_argument(
        "--max-stride-ratio",
        type=float,
        default=0.0,
        help="override pure_vlm_baseline.max_stride_ratio, e.g. 0.5",
    )
    return parser.parse_args()


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "video"


def _parse_intervals_token(token: str) -> list[list[int]]:
    token = token.strip()
    if not token:
        return []

    if token.startswith("["):
        obj = json.loads(token)
        if not isinstance(obj, list):
            return []
        intervals = []
        for pair in obj:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            s, e = int(pair[0]), int(pair[1])
            if e < s:
                s, e = e, s
            intervals.append([s, e])
        return intervals

    # Fallback: "10-20,40-55"
    out = []
    for part in token.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" not in p:
            continue
        s_str, e_str = p.split("-", 1)
        s, e = int(s_str.strip()), int(e_str.strip())
        if e < s:
            s, e = e, s
        out.append([s, e])
    return out


def _resolve_frames_root(manifest_path: str, frames_root: str) -> Path:
    if frames_root:
        root = Path(frames_root)
    else:
        root = Path(manifest_path).resolve().parent / "frames" / "abnormal"
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"frames_root not found or not a dir: {root}")
    return root


def _parse_id_total_intervals_line(text: str, frames_root: Path) -> dict[str, Any] | None:
    # Supports: v1 579 400 495 [600 620 ...]
    parts = text.split()
    if len(parts) < 2:
        return None

    vid = parts[0]
    if not vid:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", vid):
        return None

    try:
        total_frames_declared = int(parts[1])
    except ValueError:
        return None

    nums: list[int] = []
    for token in parts[2:]:
        try:
            nums.append(int(token))
        except ValueError:
            return None

    if len(nums) % 2 != 0:
        raise ValueError(f"odd interval token count for id-style line: {text}")

    intervals: list[list[int]] = []
    for i in range(0, len(nums), 2):
        s, e = nums[i], nums[i + 1]
        if e < s:
            s, e = e, s
        intervals.append([s, e])

    frame_source = str((frames_root / vid).resolve())
    return {
        "video_id": vid,
        "frame_source": frame_source,
        "gt_intervals": intervals,
        "gt_total_frames": total_frames_declared,
    }


def parse_manifest(manifest_path: str, frames_root: str = "") -> list[dict[str, Any]]:
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    resolved_frames_root = _resolve_frames_root(manifest_path=manifest_path, frames_root=frames_root)

    items: list[dict[str, Any]] = []
    for line_idx, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue

        try:
            if text.startswith("{"):
                obj = json.loads(text)
                source = str(obj.get("frame_dir") or obj.get("frame_source") or "").strip()
                if not source:
                    raise ValueError("missing frame_dir/frame_source")
                video_id = str(obj.get("video_id") or Path(source).stem)
                intervals = obj.get("gt_intervals", [])
                if not isinstance(intervals, list):
                    intervals = []
                parsed_intervals = []
                for pair in intervals:
                    if isinstance(pair, list) and len(pair) == 2:
                        s, e = int(pair[0]), int(pair[1])
                        if e < s:
                            s, e = e, s
                        parsed_intervals.append([s, e])
                items.append(
                    {
                        "video_id": video_id,
                        "frame_source": source,
                        "gt_intervals": parsed_intervals,
                        "gt_total_frames": int(obj.get("total_frames", 0)) if str(obj.get("total_frames", "")).isdigit() else 0,
                    }
                )
                continue

            # TU-DAT style: id total_frames s1 e1 [s2 e2 ...]
            id_line = _parse_id_total_intervals_line(text, resolved_frames_root)
            if id_line is not None:
                items.append(id_line)
                continue

            cols = [c.strip() for c in text.split("\t") if c.strip()]
            if len(cols) == 1:
                source = cols[0]
                intervals: list[list[int]] = []
            else:
                source = cols[0]
                intervals = _parse_intervals_token(cols[1])
            items.append(
                {
                    "video_id": Path(source).stem,
                    "frame_source": source,
                    "gt_intervals": intervals,
                    "gt_total_frames": 0,
                }
            )
        except Exception as e:
            raise ValueError(f"Manifest parse error at line {line_idx}: {e}") from e

    if not items:
        raise ValueError("No valid videos found in manifest")
    return items


def intervals_to_binary_labels(num_frames: int, intervals: list[list[int]]) -> list[int]:
    y = [0] * max(0, int(num_frames))
    for s, e in intervals:
        s = max(0, int(s))
        e = min(num_frames - 1, int(e))
        if e < s:
            continue
        for i in range(s, e + 1):
            y[i] = 1
    return y


def main() -> None:
    args = parse_args()
    cfg = load_settings(args.config)

    overrides = cfg.to_dict()
    changed = False

    if args.threshold is not None:
        overrides.setdefault("pure_vlm_baseline", {})
        overrides["pure_vlm_baseline"]["threshold"] = float(args.threshold)
        changed = True
    if args.window_size and args.window_size > 0:
        overrides.setdefault("pure_vlm_baseline", {})
        overrides["pure_vlm_baseline"]["window_size"] = int(args.window_size)
        changed = True
    if args.window_stride and args.window_stride > 0:
        overrides.setdefault("pure_vlm_baseline", {})
        overrides["pure_vlm_baseline"]["window_stride"] = int(args.window_stride)
        changed = True
    if args.sampled_frames_per_chunk and args.sampled_frames_per_chunk > 0:
        overrides.setdefault("pure_vlm_baseline", {})
        overrides["pure_vlm_baseline"]["sampled_frames_per_chunk"] = int(args.sampled_frames_per_chunk)
        changed = True
    if args.max_new_tokens and args.max_new_tokens > 0:
        overrides.setdefault("vlm", {})
        overrides["vlm"]["max_new_tokens"] = int(args.max_new_tokens)
        changed = True
    if args.max_image_size and args.max_image_size > 0:
        overrides.setdefault("pure_vlm_baseline", {})
        overrides["pure_vlm_baseline"]["max_image_size"] = int(args.max_image_size)
        changed = True
    if args.max_stride_ratio and args.max_stride_ratio > 0:
        overrides.setdefault("pure_vlm_baseline", {})
        overrides["pure_vlm_baseline"]["max_stride_ratio"] = float(args.max_stride_ratio)
        changed = True
    if args.vlm_device:
        overrides.setdefault("vlm", {})
        overrides["vlm"]["device"] = str(args.vlm_device)
        changed = True
    if args.vlm_dtype:
        overrides.setdefault("vlm", {})
        overrides["vlm"]["dtype"] = str(args.vlm_dtype)
        changed = True

    if changed:
        cfg = load_settings(args.config, overrides=overrides)

    items = parse_manifest(args.manifest, frames_root=args.frames_root)
    if args.max_videos and args.max_videos > 0:
        items = items[: int(args.max_videos)]
    out_root = args.output_dir or str(Path(cfg.output.output_dir) / "results" / "pure_vlm_baseline")
    ensure_dir(out_root)

    # Make a copy of real cfg, manifest and in output for reference.
    write_json(str(Path(out_root) / "config_used.json"), cfg.to_dict())
    with open(str(Path(out_root) / "manifest_used.txt"), "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    pipeline = PureVLMBaselinePipeline(cfg)

    frame_y_true_all: list[int] = []
    frame_y_score_all: list[float] = []
    video_y_true: list[int] = []
    video_y_score: list[float] = []

    failed: list[dict[str, str]] = []
    written_files: list[str] = []
    run_t0 = time.perf_counter()

    for item in tqdm(items, desc="Pure VLM baseline"):
        video_id = str(item["video_id"])
        frame_source = str(item["frame_source"])
        gt_intervals = list(item.get("gt_intervals", []))
        gt_total_frames = int(item.get("gt_total_frames", 0) or 0)

        json_name = _sanitize_name(video_id) + ".json"
        out_path = str(Path(out_root) / json_name)
        if args.resume and Path(out_path).exists():
            written_files.append(out_path)
            continue

        try:
            result = pipeline.run_video(video_id=video_id, frame_source=frame_source)
            result["gt_intervals"] = gt_intervals
            result["gt_total_frames"] = gt_total_frames
            if gt_total_frames > 0 and int(result.get("num_frames", 0)) != gt_total_frames:
                result["gt_frame_count_mismatch"] = {
                    "declared": gt_total_frames,
                    "actual": int(result.get("num_frames", 0)),
                }

            if gt_intervals:
                y_true = intervals_to_binary_labels(result["num_frames"], gt_intervals)
                y_score = [float(x) for x in result["frame_scores_smooth"]]
                frame_y_true_all.extend(y_true)
                frame_y_score_all.extend(y_score)
                frame_auc = roc_auc_score_binary(y_true, y_score)
                frame_ap = average_precision_score_binary(y_true, y_score)
                frame_eer = equal_error_rate_binary(y_true, y_score)
                result["frame_auc_roc"] = frame_auc
                result["frame_ruc"] = frame_auc
                result["frame_ap"] = frame_ap
                result["frame_eer"] = frame_eer["eer"] if frame_eer else None
                result["frame_edr"] = frame_eer["edr"] if frame_eer else None
                result["frame_accuracy"] = binary_accuracy_at_threshold(y_true, y_score, threshold=0.5)

            video_y_true.append(1 if gt_intervals else 0)
            video_y_score.append(float(result.get("video_score_max", 0.0)))

            write_json(out_path, result)
            written_files.append(out_path)
        except Exception as e:
            failed.append({"video_id": video_id, "frame_source": frame_source, "error": str(e)})

    run_total_sec = float(time.perf_counter() - run_t0)
    video_auc = roc_auc_score_binary(video_y_true, video_y_score)
    video_ap = average_precision_score_binary(video_y_true, video_y_score)
    video_eer = equal_error_rate_binary(video_y_true, video_y_score)
    frame_auc = roc_auc_score_binary(frame_y_true_all, frame_y_score_all)
    frame_ap = average_precision_score_binary(frame_y_true_all, frame_y_score_all)
    frame_eer = equal_error_rate_binary(frame_y_true_all, frame_y_score_all)

    summary = {
        "num_videos": len(items),
        "num_success": len(written_files),
        "num_failed": len(failed),
        "results_dir": out_root,
        "timing": {
            "run_total_sec": run_total_sec,
            "avg_video_sec": float(run_total_sec / max(1, len(written_files))),
        },
        "metrics": {
            "video_auc_roc": video_auc,
            "video_ruc": video_auc,
            "video_ap": video_ap,
            "video_eer": video_eer["eer"] if video_eer else None,
            "video_edr": video_eer["edr"] if video_eer else None,
            "video_accuracy": binary_accuracy_at_threshold(video_y_true, video_y_score, threshold=0.5),
            "frame_auc_roc": frame_auc,
            "frame_ruc": frame_auc,
            "frame_ap": frame_ap,
            "frame_eer": frame_eer["eer"] if frame_eer else None,
            "frame_edr": frame_eer["edr"] if frame_eer else None,
            "frame_accuracy": binary_accuracy_at_threshold(frame_y_true_all, frame_y_score_all, threshold=0.5),
        },
        "failed": failed,
    }
    write_json(str(Path(out_root) / "metrics_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()