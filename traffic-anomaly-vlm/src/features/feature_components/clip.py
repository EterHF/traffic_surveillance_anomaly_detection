from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor


@dataclass
class CLIPFeatureConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    model_revision: str | None = None
    device: str = "cuda"
    batch_size: int = 32
    resize_long_side: int = 0
    resize_interpolation: str = "area"


class CLIPFeatureExtractor:
    """Extract CLIP features from image arrays and compute frame-pair errors."""

    def __init__(self, cfg: CLIPFeatureConfig | None = None):
        self.cfg = cfg or CLIPFeatureConfig()
        req_device = str(self.cfg.device).lower()
        if req_device.startswith("cuda") and not torch.cuda.is_available():
            req_device = "cpu"
        self.device = torch.device(req_device)

        proc_kwargs: dict[str, Any] = {}
        if self.cfg.model_revision:
            proc_kwargs["revision"] = self.cfg.model_revision
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_name, use_fast=True, **proc_kwargs)

        model_kwargs: dict[str, Any] = {}
        if self.cfg.model_revision:
            model_kwargs["revision"] = self.cfg.model_revision

        # Follow official loading path:
        # AutoProcessor.from_pretrained(...)
        # AutoModelForZeroShotImageClassification.from_pretrained(...)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(
            self.cfg.model_name,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()

    def _image_features_from_model(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.model, "get_image_features"):
            return self.model.get_image_features(**inputs)
        clip_model = getattr(self.model, "clip_model", None)
        if clip_model is not None and hasattr(clip_model, "get_image_features"):
            return clip_model.get_image_features(**inputs)
        raise RuntimeError("Loaded CLIP model does not expose get_image_features")

    @staticmethod
    def _resize_image_if_needed(
        bgr: np.ndarray,
        resize_long_side: int,
        resize_interpolation: str,
    ) -> np.ndarray:
        if int(resize_long_side) <= 0:
            return bgr

        h, w = bgr.shape[:2]
        long_side = max(h, w)
        if long_side <= int(resize_long_side):
            return bgr

        scale = float(resize_long_side) / float(long_side)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))

        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
        }
        interp = interp_map.get(str(resize_interpolation).lower(), cv2.INTER_AREA)
        return cv2.resize(bgr, (nw, nh), interpolation=interp)

    @staticmethod
    def _to_pil_images(
        images_bgr: list[np.ndarray],
        resize_long_side: int,
        resize_interpolation: str,
    ) -> list[Image.Image]:
        out: list[Image.Image] = []
        for img in images_bgr:
            if img is None:
                continue
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("Expected image with shape [H, W, 3] in BGR format")
            img = CLIPFeatureExtractor._resize_image_if_needed(img, resize_long_side, resize_interpolation)
            out.append(Image.fromarray(img[:, :, ::-1]))
        return out

    def extract_embeddings(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        images = self._to_pil_images(
            images_bgr,
            resize_long_side=int(self.cfg.resize_long_side),
            resize_interpolation=str(self.cfg.resize_interpolation),
        )
        if not images:
            return np.zeros((0, 512), dtype=np.float32)

        embs: list[np.ndarray] = []
        bs = max(1, int(self.cfg.batch_size))
        with torch.no_grad():
            for i in range(0, len(images), bs):
                batch_images = images[i : i + bs]
                inputs = self.processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self._image_features_from_model(inputs)
                if isinstance(feats, dict):
                    if "image_embeds" in feats and feats["image_embeds"] is not None:
                        feats = feats["image_embeds"]
                    elif "pooler_output" in feats and feats["pooler_output"] is not None:
                        feats = feats["pooler_output"]
                    else:
                        feats = next(iter(feats.values()))
                elif hasattr(feats, "image_embeds"):
                    feats = feats.image_embeds
                elif hasattr(feats, "pooler_output"):
                    feats = feats.pooler_output
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                embs.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(embs, axis=0)

    def compute_pair_errors(self, images_bgr: list[np.ndarray]) -> dict[str, np.ndarray]:
        embs = self.extract_embeddings(images_bgr)
        if embs.shape[0] < 2:
            return {
                "cosine_error": np.array([], dtype=np.float32),
                "l2_error": np.array([], dtype=np.float32),
                "sq_l2_error": np.array([], dtype=np.float32),
                "eventvad_combined_error": np.array([], dtype=np.float32),
                "delta_cosine_error": np.array([], dtype=np.float32),
            }

        prev = embs[:-1]
        curr = embs[1:]
        diff = curr - prev
        sq_l2_error = np.sum(diff * diff, axis=1).astype(np.float32)
        cosine_sim = np.sum(prev * curr, axis=1)
        cosine_error = (1.0 - cosine_sim).astype(np.float32)
        l2_error = np.sqrt(np.maximum(sq_l2_error, 0.0)).astype(np.float32)
        combined = (sq_l2_error + cosine_error).astype(np.float32)
        delta = np.zeros_like(cosine_error)
        delta[1:] = np.abs(cosine_error[1:] - cosine_error[:-1])
        return {
            "cosine_error": cosine_error,
            "l2_error": l2_error,
            "sq_l2_error": sq_l2_error,
            "eventvad_combined_error": combined,
            "delta_cosine_error": delta,
        }


if __name__ == "__main__":
    from src.settings import load_settings
    cfg = load_settings()

    from src.core.cuda_compat import ensure_cuda_runtime_compat
    ensure_cuda_runtime_compat()

    clip_cfg = CLIPFeatureConfig(
        model_name=cfg.features.clip.model_name,
        model_revision=cfg.features.clip.model_revision,
        device=cfg.features.clip.device,
        batch_size=cfg.features.clip.batch_size,
        resize_long_side=cfg.features.clip.resize_long_side,
        resize_interpolation=cfg.features.clip.resize_interpolation,
    )
    extractor = CLIPFeatureExtractor(clip_cfg)
    print("CLIPFeatureExtractor initialized successfully")
    # test if clip is correctly loaded and can process a dummy image
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    features = extractor.extract_embeddings([dummy_image])
    print("Extracted features shape:", features.shape)  
