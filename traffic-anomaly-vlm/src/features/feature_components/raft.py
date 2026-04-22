from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import torch
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _ensure_raft_imports(raft_root: Path) -> None:
    core_dir = raft_root / "core"
    for p in (str(raft_root), str(core_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)


@dataclass
class RAFTFeatureConfig:
    raft_root: str
    weights_path: str
    device: str = "cuda"
    iters: int = 12
    resize_long_side: int = 640
    resize_interpolation: str = "area"
    flow_proj_dim: int = 128
    flow_proj_seed: int = 42


class RAFTFeatureExtractor:
    """Extract RAFT motion features from image arrays."""

    def __init__(self, cfg: RAFTFeatureConfig):
        self.cfg = cfg
        self.raft_root = Path(cfg.raft_root).resolve()
        _ensure_raft_imports(self.raft_root)

        from raft import RAFT  # type: ignore
        from utils.utils import InputPadder  # type: ignore

        self._InputPadder = InputPadder

        req_device = str(cfg.device).lower()
        if req_device.startswith("cuda") and not torch.cuda.is_available():
            req_device = "cpu"
        self.device = torch.device(req_device)

        model_args = argparse.Namespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False,
            dropout=0.0,
        )
        model = RAFT(model_args)

        ckpt_path = Path(cfg.weights_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"RAFT weights not found: {ckpt_path}")

        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if isinstance(state, dict):
            clean = {}
            for k, v in state.items():
                nk = k[7:] if k.startswith("module.") else k
                clean[nk] = v
            model.load_state_dict(clean, strict=True)
        else:
            model.load_state_dict(state, strict=True)

        self.model = model.to(self.device)
        self.model.eval()
        self.flow_proj = self._init_random_ortho(2, int(cfg.flow_proj_dim), int(cfg.flow_proj_seed))

    @staticmethod
    def _init_random_ortho(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        m = max(in_dim, out_dim)
        q, _ = np.linalg.qr(rng.standard_normal((m, m)))
        return q[:in_dim, :out_dim].astype(np.float32)

    def _preprocess_image(self, bgr: np.ndarray) -> np.ndarray:
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Expected image with shape [H, W, 3] in BGR format")

        if int(self.cfg.resize_long_side) > 0:
            h, w = bgr.shape[:2]
            long_side = max(h, w)
            if long_side > int(self.cfg.resize_long_side):
                scale = float(self.cfg.resize_long_side) / float(long_side)
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                interp_map = {
                    "nearest": cv2.INTER_NEAREST,
                    "linear": cv2.INTER_LINEAR,
                    "area": cv2.INTER_AREA,
                    "cubic": cv2.INTER_CUBIC,
                    "lanczos4": cv2.INTER_LANCZOS4,
                }
                interp = interp_map.get(str(self.cfg.resize_interpolation).lower(), cv2.INTER_AREA)
                bgr = cv2.resize(bgr, (nw, nh), interpolation=interp)
        return bgr

    def _to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ten = torch.from_numpy(rgb).permute(2, 0, 1).float()[None]
        return ten.to(self.device)

    def _compute_flow_pair(self, img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> np.ndarray:
        img1 = self._to_tensor(self._preprocess_image(img1_bgr))
        img2 = self._to_tensor(self._preprocess_image(img2_bgr))

        padder = self._InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        _, flow_up = self.model(img1, img2, iters=int(self.cfg.iters), test_mode=True)
        flow_up = padder.unpad(flow_up)
        return flow_up[0].detach().float().cpu().numpy().astype(np.float32)

    def extract_eventvad_flow_features(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """
        EventVAD-style flow features:
        1) dense flow for adjacent frames
        2) global average pool to 2D vector (dx, dy)
        3) random orthogonal projection to configurable dim
        4) prepend zero vector to align frame count
        Returns shape: [num_frames, flow_proj_dim]
        """
        n = len(images_bgr)
        proj_dim = int(self.cfg.flow_proj_dim)
        if n == 0:
            return np.zeros((0, proj_dim), dtype=np.float32)
        if n == 1:
            return np.zeros((1, proj_dim), dtype=np.float32)

        pooled: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(n - 1):
                flow = self._compute_flow_pair(images_bgr[i], images_bgr[i + 1])
                pooled.append(np.mean(flow, axis=(1, 2)).astype(np.float32))

        pooled_arr = np.stack(pooled, axis=0)  # [n-1, 2]
        projected = np.dot(pooled_arr, self.flow_proj).astype(np.float32)  # [n-1, proj_dim]
        first = np.zeros((1, proj_dim), dtype=np.float32)
        return np.concatenate([first, projected], axis=0)

    def compute_pair_errors(self, images_bgr: list[np.ndarray]) -> dict[str, Any]:
        if len(images_bgr) < 2:
            return {
                "flow_mag_mean": np.array([], dtype=np.float32),
                "flow_mag_p95": np.array([], dtype=np.float32),
                "flow_temporal_diff": np.array([], dtype=np.float32),
                "eventvad_sq_l2_error": np.array([], dtype=np.float32),
                "eventvad_cosine_error": np.array([], dtype=np.float32),
                "eventvad_combined_error": np.array([], dtype=np.float32),
            }

        mag_mean: list[float] = []
        mag_p95: list[float] = []
        temporal_diff: list[float] = []

        prev_flow: np.ndarray | None = None
        with torch.no_grad():
            for i in range(len(images_bgr) - 1):
                flow = self._compute_flow_pair(images_bgr[i], images_bgr[i + 1])
                flow_mag = np.linalg.norm(flow, axis=0)

                mag_mean.append(float(np.mean(flow_mag)))
                mag_p95.append(float(np.percentile(flow_mag, 95)))

                if prev_flow is None:
                    temporal_diff.append(0.0)
                else:
                    h = min(prev_flow.shape[1], flow.shape[1])
                    w = min(prev_flow.shape[2], flow.shape[2])
                    pf = prev_flow[:, :h, :w]
                    cf = flow[:, :h, :w]
                    diff = np.linalg.norm(cf - pf, axis=0)
                    temporal_diff.append(float(np.mean(diff)))

                prev_flow = flow

        flow_feats = self.extract_eventvad_flow_features(images_bgr)
        prev = flow_feats[:-1]
        curr = flow_feats[1:]
        diff = curr - prev
        sq_l2 = np.sum(diff * diff, axis=1).astype(np.float32)
        prev_n = prev / (np.linalg.norm(prev, axis=1, keepdims=True) + 1e-6)
        curr_n = curr / (np.linalg.norm(curr, axis=1, keepdims=True) + 1e-6)
        cos_err = (1.0 - np.sum(prev_n * curr_n, axis=1)).astype(np.float32)
        combined = (sq_l2 + cos_err).astype(np.float32)

        return {
            "flow_mag_mean": np.array(mag_mean, dtype=np.float32),
            "flow_mag_p95": np.array(mag_p95, dtype=np.float32),
            "flow_temporal_diff": np.array(temporal_diff, dtype=np.float32),
            "eventvad_sq_l2_error": sq_l2,
            "eventvad_cosine_error": cos_err,
            "eventvad_combined_error": combined,
        }


if __name__ == "__main__":
    from src.settings import load_settings
    cfg = load_settings()

    from src.core.cuda_compat import ensure_cuda_runtime_compat
    ensure_cuda_runtime_compat()

    raft_cfg = RAFTFeatureConfig(
        raft_root=cfg.features.raft.root,
        weights_path=cfg.features.raft.weights,
        device=cfg.features.raft.device,
        iters=cfg.features.raft.iters,
        resize_long_side=cfg.features.raft.resize_long_side,
        resize_interpolation=cfg.features.raft.resize_interpolation,
    )
    extractor = RAFTFeatureExtractor(raft_cfg)
    print("RAFT feature extractor initialized successfully")
    # test if raft is correctly imported and can process dummy images
    dummy_img_1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    dummy_img_2 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    feature = extractor.extract_eventvad_flow_features([dummy_img_1, dummy_img_2])
    errors = extractor.compute_pair_errors([dummy_img_1, dummy_img_2])
    print("RAFT feature extraction and error computation successful")
    print("Extracted feature shape:", feature.shape)
    print("Computed errors:", {k: v.item() for k, v in errors.items()})