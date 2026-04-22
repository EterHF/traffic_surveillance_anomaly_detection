from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.schemas import TrackObject


@dataclass
class SceneFeatureConfig:
    lowres_size: int = 32
    occupancy_grid: int = 4

    # Move to feature_builder.py for easier global weighting control:
    # clip_weight: float = 0.0
    # raft_weight: float = 0.0

    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_model_revision: str | None = None
    clip_device: str = "cuda"
    clip_batch_size: int = 16
    clip_resize_long_side: int = 640
    clip_resize_interpolation: str = "area"

    raft_root: str = ""
    raft_weights: str = ""
    raft_device: str = "cuda"
    raft_iters: int = 12
    raft_resize_long_side: int = 640
    raft_resize_interpolation: str = "area"
    raft_flow_proj_dim: int = 128
    raft_flow_proj_seed: int = 42


def align_pair_signal(pair_values: np.ndarray, num_frames: int) -> np.ndarray:
    out = np.zeros((max(0, int(num_frames))), dtype=np.float32)
    if out.size <= 1 or pair_values.size == 0:
        return out
    usable = min(out.size - 1, pair_values.size)
    out[1 : usable + 1] = pair_values[:usable]
    return out


def pair_errors_from_embeddings(embs: np.ndarray) -> dict[str, np.ndarray]:
    if embs.shape[0] < 2:
        empty = np.array([], dtype=np.float32)
        return {"sq_l2_error": empty, "cosine_error": empty, "combined_error": empty}

    prev = embs[:-1]
    curr = embs[1:]
    diff = curr - prev
    sq_l2 = np.sum(diff * diff, axis=1).astype(np.float32)
    prev_n = prev / (np.linalg.norm(prev, axis=1, keepdims=True) + 1e-6)
    curr_n = curr / (np.linalg.norm(curr, axis=1, keepdims=True) + 1e-6)
    cosine_error = (1.0 - np.sum(prev_n * curr_n, axis=1)).astype(np.float32)
    return {
        "sq_l2_error": sq_l2,
        "cosine_error": cosine_error,
        "combined_error": (sq_l2 + cosine_error).astype(np.float32),
    }


def build_lowres_embeddings(images_bgr: list[np.ndarray], lowres_size: int) -> np.ndarray:
    size = max(8, int(lowres_size))
    descs: list[np.ndarray] = []
    for img in images_bgr:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        grad_x = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        desc = np.concatenate([small.reshape(-1), grad_mag.reshape(-1)], axis=0)
        desc = desc / (np.linalg.norm(desc) + 1e-6)
        descs.append(desc.astype(np.float32))
    dim = size * size * 2
    return np.stack(descs, axis=0) if descs else np.zeros((0, dim), dtype=np.float32)


def build_track_layout_embeddings(
    tracks_per_frame: list[list[TrackObject]],
    occupancy_grid: int,
) -> np.ndarray:
    grid = max(2, int(occupancy_grid))
    descs: list[np.ndarray] = []
    for frame_tracks in tracks_per_frame:
        occ = np.zeros((grid, grid), dtype=np.float32)
        area = np.zeros((grid, grid), dtype=np.float32)
        count = float(len(frame_tracks))
        fw, fh = _frame_dims_from_tracks(frame_tracks)
        frame_area = max(1.0, fw * fh)
        total_area = 0.0

        for track in frame_tracks:
            gx = min(grid - 1, max(0, int(float(track.cx) / max(1.0, fw) * grid)))
            gy = min(grid - 1, max(0, int(float(track.cy) / max(1.0, fh) * grid)))
            occ[gy, gx] += 1.0
            area_ratio = float(track.area) / frame_area
            area[gy, gx] += area_ratio
            total_area += area_ratio

        vec = np.concatenate(
            [occ.reshape(-1), area.reshape(-1), np.array([count, total_area], dtype=np.float32)],
            axis=0,
        )
        vec = vec / (np.linalg.norm(vec) + 1e-6)
        descs.append(vec.astype(np.float32))
    dim = grid * grid * 2 + 2
    return np.stack(descs, axis=0) if descs else np.zeros((0, dim), dtype=np.float32)


def _frame_dims_from_tracks(frame_tracks: list[TrackObject]) -> tuple[float, float]:
    if not frame_tracks:
        return 1.0, 1.0
    fw = max((float(track.frame_w) for track in frame_tracks), default=0.0)
    fh = max((float(track.frame_h) for track in frame_tracks), default=0.0)
    if fw > 0.0 and fh > 0.0:
        return fw, fh
    x2_max = max((float(track.bbox_xyxy[2]) for track in frame_tracks), default=1.0)
    y2_max = max((float(track.bbox_xyxy[3]) for track in frame_tracks), default=1.0)
    return max(1.0, x2_max), max(1.0, y2_max)


def _frame_density(frame_tracks: list[TrackObject]) -> float:
    if not frame_tracks:
        return 0.0
    frame_w, frame_h = _frame_dims_from_tracks(frame_tracks)
    frame_area = max(1.0, frame_w * frame_h)
    return float(sum(float(track.area) for track in frame_tracks) / frame_area)


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=True)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = max(1e-6, 1.4826 * mad)
    z = np.maximum((values - median) / sigma, 0.0)
    high = max(float(np.percentile(z, 95)), 1e-6)
    return np.clip(z / high, 0.0, 1.0).astype(np.float32)


def _top_mean(values: np.ndarray, ratio: float = 0.25) -> float:
    if values.size == 0:
        return 0.0
    keep = max(1, int(np.ceil(values.size * float(ratio))))
    return float(np.mean(np.sort(values)[-keep:]))


def _summarize_pair_change(pair_values: np.ndarray, num_frames: int) -> float:
    signal = align_pair_signal(pair_values.astype(np.float32), num_frames)
    return _top_mean(_robust_unit_scale(signal), ratio=0.25)


class SceneFeatureExtractor:
    """Compute scene-level statistics and optional visual/layout change cues."""

    def __init__(self, cfg: SceneFeatureConfig | None = None):
        # Force cfg 
        self.cfg = cfg # or SceneFeatureConfig()
        self._clip_extractor = None
        self._raft_extractor = None

    def _get_clip_extractor(self):
        if self._clip_extractor is not None:
            return self._clip_extractor
        from src.features.feature_components.clip import CLIPFeatureConfig, CLIPFeatureExtractor

        self._clip_extractor = CLIPFeatureExtractor(
            CLIPFeatureConfig(
                model_name=self.cfg.clip_model_name,
                model_revision=self.cfg.clip_model_revision,
                device=self.cfg.clip_device,
                batch_size=self.cfg.clip_batch_size,
                resize_long_side=self.cfg.clip_resize_long_side,
                resize_interpolation=self.cfg.clip_resize_interpolation,
            )
        )
        return self._clip_extractor

    def _get_raft_extractor(self):
        if self._raft_extractor is not None:
            return self._raft_extractor
        from src.features.feature_components.raft import RAFTFeatureConfig, RAFTFeatureExtractor

        self._raft_extractor = RAFTFeatureExtractor(
            RAFTFeatureConfig(
                raft_root=self.cfg.raft_root,
                weights_path=self.cfg.raft_weights,
                device=self.cfg.raft_device,
                iters=self.cfg.raft_iters,
                resize_long_side=self.cfg.raft_resize_long_side,
                resize_interpolation=self.cfg.raft_resize_interpolation,
                flow_proj_dim=self.cfg.raft_flow_proj_dim,
                flow_proj_seed=self.cfg.raft_flow_proj_seed,
            )
        )
        return self._raft_extractor

    def compute_pair_signals(
        self,
        window_tracks: list[list[TrackObject]] | None = None,
        images_bgr: list[np.ndarray] | None = None,
        lowres_weight: float | None = None,
        layout_weight: float | None = None,
        clip_weight: float | None = None,
        raft_weight: float | None = None,
    ) -> dict[str, np.ndarray]:
        window_tracks = window_tracks or []
        images_bgr = images_bgr or []

        lowres_pair = np.array([], dtype=np.float32)
        if float(lowres_weight or 0.0) > 0.0 and images_bgr:
            lowres_pair = pair_errors_from_embeddings(
                build_lowres_embeddings(images_bgr, self.cfg.lowres_size)
            )["combined_error"]

        layout_pair = np.array([], dtype=np.float32)
        if float(layout_weight or 0.0) > 0.0 and window_tracks:
            layout_pair = pair_errors_from_embeddings(
                build_track_layout_embeddings(window_tracks, self.cfg.occupancy_grid)
            )["combined_error"]

        clip_pair = np.array([], dtype=np.float32)
        if float(clip_weight or 0.0) > 0.0 and images_bgr:
            clip_pair = self._get_clip_extractor().compute_pair_errors(images_bgr)["eventvad_combined_error"]

        raft_pair = np.array([], dtype=np.float32)
        if (
            float(raft_weight or 0.0) > 0.0
            and images_bgr
            and self.cfg.raft_root
            and self.cfg.raft_weights
        ):
            raft_pair = self._get_raft_extractor().compute_pair_errors(images_bgr)["eventvad_combined_error"]

        return {
            "lowres_pair": lowres_pair.astype(np.float32, copy=False),
            "layout_pair": layout_pair.astype(np.float32, copy=False),
            "clip_pair": clip_pair.astype(np.float32, copy=False),
            "raft_pair": raft_pair.astype(np.float32, copy=False),
        }

    def compute(
        self,
        window_tracks: list[list[TrackObject]],
        images_bgr: list[np.ndarray] | None = None,
        lowres_weight: float | None = None,
        layout_weight: float | None = None,
        clip_weight: float | None = None,
        raft_weight: float | None = None,
        lowres_pair: np.ndarray | None = None,
        layout_pair: np.ndarray | None = None,
        clip_pair: np.ndarray | None = None,
        raft_pair: np.ndarray | None = None,
    ) -> dict[str, float]:
        if not window_tracks and not images_bgr:
            return {
                "cur_count": 0.0,
                "count_mean": 0.0,
                "count_std": 0.0,
                "count_delta": 0.0,
                "cur_density": 0.0,
                "density_mean": 0.0,
                "density_std": 0.0,
                "density_delta": 0.0,
                "lowres_eventness": 0.0,
                "track_layout_eventness": 0.0,
                "clip_eventness": 0.0,
                "raft_eventness": 0.0,
            }

        density_vals = [_frame_density(frame_tracks) for frame_tracks in window_tracks] if window_tracks else [0.0]
        per_frame_counts = [len(frame_tracks) for frame_tracks in window_tracks] if window_tracks else [0]

        if (
            lowres_pair is None
            and layout_pair is None
            and clip_pair is None
            and raft_pair is None
        ):
            pair_signals = self.compute_pair_signals(
                window_tracks=window_tracks,
                images_bgr=images_bgr,
                lowres_weight=lowres_weight,
                layout_weight=layout_weight,
                clip_weight=clip_weight,
                raft_weight=raft_weight,
            )
            lowres_pair = pair_signals["lowres_pair"]
            layout_pair = pair_signals["layout_pair"]
            clip_pair = pair_signals["clip_pair"]
            raft_pair = pair_signals["raft_pair"]

        lowres_eventness = 0.0
        if float(lowres_weight or 0.0) > 0.0 and lowres_pair is not None:
            lowres_eventness = _summarize_pair_change(lowres_pair, len(window_tracks) or len(images_bgr or []))

        track_layout_eventness = 0.0
        if float(layout_weight or 0.0) > 0.0 and layout_pair is not None:
            track_layout_eventness = _summarize_pair_change(layout_pair, len(window_tracks))

        clip_eventness = 0.0
        if clip_pair is not None:
            clip_eventness = _summarize_pair_change(clip_pair, len(window_tracks) or len(images_bgr or []))

        raft_eventness = 0.0
        if raft_pair is not None:
            raft_eventness = _summarize_pair_change(raft_pair, len(window_tracks) or len(images_bgr or []))

        return {
            "cur_count": float(per_frame_counts[-1]) if per_frame_counts else 0.0,
            "count_mean": float(np.mean(per_frame_counts)) if per_frame_counts else 0.0,
            "count_std": float(np.std(per_frame_counts)) if per_frame_counts else 0.0,
            "count_delta": float(per_frame_counts[-1] - per_frame_counts[0]) if len(per_frame_counts) >= 2 else 0.0,
            "cur_density": float(density_vals[-1]) if density_vals else 0.0,
            "density_mean": float(np.mean(density_vals)) if density_vals else 0.0,
            "density_std": float(np.std(density_vals)) if density_vals else 0.0,
            "density_delta": float(density_vals[-1] - density_vals[0]) if len(density_vals) >= 2 else 0.0,
            "lowres_eventness": float(lowres_eventness),
            "track_layout_eventness": float(track_layout_eventness),
            "clip_eventness": float(clip_eventness),
            "raft_eventness": float(raft_eventness),
        }
