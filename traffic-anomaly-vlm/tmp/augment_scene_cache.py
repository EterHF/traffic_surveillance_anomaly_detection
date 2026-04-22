from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.feature_components.scene import SceneFeatureConfig, SceneFeatureExtractor, align_pair_signal


def _top_mean(values: np.ndarray, ratio: float = 0.25) -> float:
    if values.size == 0:
        return 0.0
    keep = max(1, int(np.ceil(values.size * float(ratio))))
    return float(np.mean(np.sort(values)[-keep:]))


def _robust_scale(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    if values.size == 0:
        return values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = max(1e-6, 1.4826 * mad)
    z = np.maximum((values - median) / sigma, 0.0)
    hi = max(float(np.percentile(z, 95)), 1e-6)
    return np.clip(z / hi, 0.0, 1.0).astype(np.float32)


def _window_pair_summary(pair_values: np.ndarray, window_len: int) -> float:
    signal = align_pair_signal(pair_values.astype(np.float32), window_len)
    return _top_mean(_robust_scale(signal), ratio=0.25)


def _load_cfg(cfg_path: str) -> SceneFeatureConfig:
    with Path(cfg_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return SceneFeatureConfig(**dict(raw.get("scene_features", {})))


def _augment_video_cache(
    payload: dict[str, Any],
    extractor: SceneFeatureExtractor,
    precompute_clip: bool,
    precompute_raft: bool,
) -> dict[str, Any]:
    frame_paths = [str(p) for p in payload["sampled_frame_paths"]]
    images = [cv2.imread(path) for path in frame_paths]
    if any(img is None for img in images):
        missing = next(path for path, img in zip(frame_paths, images) if img is None)
        raise FileNotFoundError(f"Failed to read cached sampled frame: {missing}")

    pair_signals = extractor.compute_pair_signals(
        window_tracks=None,
        images_bgr=images,
        clip_weight=1.0 if precompute_clip else 0.0,
        raft_weight=1.0 if precompute_raft else 0.0,
    )
    clip_pair = pair_signals["clip_pair"]
    raft_pair = pair_signals["raft_pair"]

    window_size = int(payload["window_size"])
    window_step = int(payload["window_step"])
    for idx, row in enumerate(payload["window_rows"]):
        window_start = idx * window_step
        window_end = window_start + window_size - 1
        feature_dict = dict(row["feature_dict"])
        if precompute_clip:
            clip_seg = clip_pair[window_start:window_end]
            feature_dict["clip_eventness"] = _window_pair_summary(clip_seg, window_size) if clip_seg.size else 0.0
        if precompute_raft:
            raft_seg = raft_pair[window_start:window_end]
            feature_dict["raft_eventness"] = _window_pair_summary(raft_seg, window_size) if raft_seg.size else 0.0
        row["feature_dict"] = feature_dict

    payload["scene_pair_augmented"] = {
        "clip_eventness": bool(precompute_clip),
        "raft_eventness": bool(precompute_raft),
    }
    return payload


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment cached raw-score features with clip/raft scene eventness")
    parser.add_argument("--config", default="/nvme2/VAD_yemao/traffic-anomaly-vlm/configs/default.yaml")
    parser.add_argument("--base-cache-dir", required=True)
    parser.add_argument("--out-cache-dir", required=True)
    parser.add_argument("--precompute-clip", action="store_true")
    parser.add_argument("--precompute-raft", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--video-ids", nargs="*", default=None)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not args.precompute_clip and not args.precompute_raft:
        raise ValueError("At least one of --precompute-clip / --precompute-raft must be enabled")

    scene_cfg = _load_cfg(args.config)
    extractor = SceneFeatureExtractor(scene_cfg)
    base_root = Path(args.base_cache_dir)
    out_root = Path(args.out_cache_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    video_ids = sorted(p.name for p in base_root.iterdir() if p.is_dir())
    if args.video_ids:
        keep = set(args.video_ids)
        video_ids = [video_id for video_id in video_ids if video_id in keep]

    for video_id in tqdm(video_ids, desc="augment-cache", dynamic_ncols=True):
        src_path = base_root / video_id / "feature_rows.json"
        dst_dir = out_root / video_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / "feature_rows.json"
        if dst_path.exists() and not args.force:
            continue
        with src_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload = _augment_video_cache(
            payload=payload,
            extractor=extractor,
            precompute_clip=bool(args.precompute_clip),
            precompute_raft=bool(args.precompute_raft),
        )
        with dst_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(build_args())
