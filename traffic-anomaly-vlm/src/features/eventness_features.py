from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.signal import savgol_filter

from src.schemas import TrackObject


@dataclass
class EventnessFeatureConfig:
    use_clip: bool = False
    use_raft: bool = False
    use_lowres: bool = True
    use_track_layout: bool = True

    lowres_size: int = 32
    occupancy_grid: int = 4
    smooth_window: int = 9
    smooth_polyorder: int = 2

    clip_weight: float = 0.45
    raft_weight: float = 0.15
    lowres_weight: float = 0.20
    track_weight: float = 0.20
    signal_weight: float = 0.30

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


@dataclass
class EventnessResult:
    frame_scores: np.ndarray
    cue_scores: dict[str, np.ndarray] = field(default_factory=dict)
    raw_cue_scores: dict[str, np.ndarray] = field(default_factory=dict)
    pair_scores: dict[str, np.ndarray] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)


def _safe_savgol(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    if values.size < 3:
        return values.astype(np.float32, copy=True)

    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window > values.size:
        window = values.size if values.size % 2 == 1 else values.size - 1
    if window < 3:
        return values.astype(np.float32, copy=True)

    polyorder = max(1, min(int(polyorder), window - 1))
    if window <= polyorder:
        return values.astype(np.float32, copy=True)
    return savgol_filter(values, window_length=window, polyorder=polyorder).astype(np.float32)


def _median(vals: np.ndarray) -> float:
    if vals.size == 0:
        return 0.0
    return float(np.median(vals))


def _mad(vals: np.ndarray, med: float) -> float:
    if vals.size == 0:
        return 0.0
    return float(np.median(np.abs(vals - med)))


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    if values.size == 0:
        return values

    med = _median(values)
    sigma = max(1e-6, 1.4826 * _mad(values, med))
    z = np.maximum((values - med) / sigma, 0.0)
    hi = float(np.percentile(z, 95)) if z.size > 0 else 0.0
    hi = max(hi, 1e-6)
    return np.clip(z / hi, 0.0, 1.0).astype(np.float32)


def _align_pair_signal(pair_values: np.ndarray, num_frames: int) -> np.ndarray:
    out = np.zeros((max(0, int(num_frames))), dtype=np.float32)
    if out.size <= 1 or pair_values.size == 0:
        return out
    usable = min(out.size - 1, pair_values.size)
    out[1 : usable + 1] = pair_values[:usable]
    return out


def _pair_errors_from_embeddings(embs: np.ndarray) -> dict[str, np.ndarray]:
    if embs.shape[0] < 2:
        return {
            "sq_l2_error": np.array([], dtype=np.float32),
            "cosine_error": np.array([], dtype=np.float32),
            "combined_error": np.array([], dtype=np.float32),
        }

    prev = embs[:-1]
    curr = embs[1:]
    diff = curr - prev
    sq_l2 = np.sum(diff * diff, axis=1).astype(np.float32)
    prev_n = prev / (np.linalg.norm(prev, axis=1, keepdims=True) + 1e-6)
    curr_n = curr / (np.linalg.norm(curr, axis=1, keepdims=True) + 1e-6)
    cosine_error = (1.0 - np.sum(prev_n * curr_n, axis=1)).astype(np.float32)
    combined = (sq_l2 + cosine_error).astype(np.float32)
    return {
        "sq_l2_error": sq_l2,
        "cosine_error": cosine_error,
        "combined_error": combined,
    }


def _build_lowres_embeddings(images_bgr: list[np.ndarray], lowres_size: int) -> np.ndarray:
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
    return np.stack(descs, axis=0) if descs else np.zeros((0, size * size * 2), dtype=np.float32)


def _build_track_layout_embeddings(
    tracks_per_frame: list[list[TrackObject]],
    occupancy_grid: int,
) -> np.ndarray:
    grid = max(2, int(occupancy_grid))
    descs: list[np.ndarray] = []
    for frame_tracks in tracks_per_frame:
        occ = np.zeros((grid, grid), dtype=np.float32)
        area = np.zeros((grid, grid), dtype=np.float32)
        count = float(len(frame_tracks))

        fw = max((float(t.frame_w) for t in frame_tracks), default=1.0)
        fh = max((float(t.frame_h) for t in frame_tracks), default=1.0)
        frame_area = max(1.0, fw * fh)
        total_area = 0.0

        for t in frame_tracks:
            gx = min(grid - 1, max(0, int(float(t.cx) / max(1.0, fw) * grid)))
            gy = min(grid - 1, max(0, int(float(t.cy) / max(1.0, fh) * grid)))
            occ[gy, gx] += 1.0
            a = float(t.area) / frame_area
            area[gy, gx] += a
            total_area += a

        vec = np.concatenate(
            [
                occ.reshape(-1),
                area.reshape(-1),
                np.array([count, total_area], dtype=np.float32),
            ],
            axis=0,
        )
        vec = vec / (np.linalg.norm(vec) + 1e-6)
        descs.append(vec.astype(np.float32))
    dim = grid * grid * 2 + 2
    return np.stack(descs, axis=0) if descs else np.zeros((0, dim), dtype=np.float32)


def _normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    valid = {k: max(0.0, float(v)) for k, v in raw_weights.items() if float(v) > 0.0}
    total = sum(valid.values())
    if total <= 0.0:
        return {}
    return {k: float(v / total) for k, v in valid.items()}


class EventnessFeatureExtractor:
    """Compute eventness cues from generic frames and optional object layouts."""

    def __init__(self, cfg: EventnessFeatureConfig | None = None):
        self.cfg = cfg or EventnessFeatureConfig()
        self._clip_extractor = None
        self._raft_extractor = None

    def _get_clip_extractor(self):
        if self._clip_extractor is not None:
            return self._clip_extractor
        from src.features.clip_features import CLIPFeatureConfig, CLIPFeatureExtractor

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
        from src.features.raft_features import RAFTFeatureConfig, RAFTFeatureExtractor

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

    def compute(
        self,
        images_bgr: list[np.ndarray],
        tracks_per_frame: list[list[TrackObject]] | None = None,
        auxiliary_signals: dict[str, np.ndarray] | None = None,
    ) -> EventnessResult:
        num_frames = len(images_bgr)
        raw_cue_scores: dict[str, np.ndarray] = {}
        pair_scores: dict[str, np.ndarray] = {}
        cue_weights: dict[str, float] = {}

        if self.cfg.use_lowres and images_bgr:
            lowres_embs = _build_lowres_embeddings(images_bgr, self.cfg.lowres_size)
            lowres_err = _pair_errors_from_embeddings(lowres_embs)["combined_error"]
            pair_scores["lowres_eventness"] = lowres_err
            raw_cue_scores["lowres_eventness"] = _align_pair_signal(lowres_err, num_frames)
            cue_weights["lowres_eventness"] = float(self.cfg.lowres_weight)

        if self.cfg.use_track_layout and tracks_per_frame is not None:
            layout_embs = _build_track_layout_embeddings(tracks_per_frame, self.cfg.occupancy_grid)
            layout_err = _pair_errors_from_embeddings(layout_embs)["combined_error"]
            pair_scores["track_layout_eventness"] = layout_err
            raw_cue_scores["track_layout_eventness"] = _align_pair_signal(layout_err, num_frames)
            cue_weights["track_layout_eventness"] = float(self.cfg.track_weight)

        if self.cfg.use_clip and images_bgr:
            clip_err = self._get_clip_extractor().compute_pair_errors(images_bgr)
            clip_combined = clip_err["eventvad_combined_error"].astype(np.float32)
            pair_scores["clip_eventness"] = clip_combined
            raw_cue_scores["clip_eventness"] = _align_pair_signal(clip_combined, num_frames)
            cue_weights["clip_eventness"] = float(self.cfg.clip_weight)

        if self.cfg.use_raft and images_bgr and num_frames >= 2 and self.cfg.raft_root and self.cfg.raft_weights:
            raft_err = self._get_raft_extractor().compute_pair_errors(images_bgr)
            raft_combined = raft_err["eventvad_combined_error"].astype(np.float32)
            pair_scores["raft_eventness"] = raft_combined
            raw_cue_scores["raft_eventness"] = _align_pair_signal(raft_combined, num_frames)
            cue_weights["raft_eventness"] = float(self.cfg.raft_weight)

        if auxiliary_signals:
            for name, values in auxiliary_signals.items():
                signal = np.asarray(values, dtype=np.float32).reshape(-1)
                if signal.size != num_frames:
                    raise ValueError(f"auxiliary signal '{name}' length does not match num_frames")
                raw_cue_scores[name] = signal.astype(np.float32, copy=True)
                cue_weights[name] = float(self.cfg.signal_weight)

        cue_scores: dict[str, np.ndarray] = {}
        for name, signal in raw_cue_scores.items():
            smooth = _safe_savgol(signal.astype(np.float32), self.cfg.smooth_window, self.cfg.smooth_polyorder)
            cue_scores[name] = _robust_unit_scale(smooth)

        weights = _normalize_weights(cue_weights)
        if cue_scores and weights:
            fused = np.zeros((num_frames,), dtype=np.float32)
            for name, w in weights.items():
                fused += cue_scores.get(name, 0.0) * float(w)
            fused = _safe_savgol(fused, self.cfg.smooth_window, self.cfg.smooth_polyorder)
            fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
        else:
            fused = np.zeros((num_frames,), dtype=np.float32)

        return EventnessResult(
            frame_scores=fused,
            cue_scores=cue_scores,
            raw_cue_scores=raw_cue_scores,
            pair_scores=pair_scores,
            weights=weights,
        )
