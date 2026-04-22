from __future__ import annotations

from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor


@dataclass
class VLMConfig:
  enable: bool = True
  model_path: str = "/nvme2/VAD_yemao/traffic-anomaly-vlm/weights/qwen3-vl-4bi"
  device: str = "cuda:0"
  dtype: str = "float16"
  max_new_tokens: int = 4096
  max_image_size: int = 640


class LocalVLM:
    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        self.model_path = cfg.model_path
        self.device = cfg.device
        self.max_new_tokens = cfg.max_new_tokens
        self.max_image_size = cfg.max_image_size

        if cfg.dtype == "auto":
            torch_dtype = "auto"
        elif cfg.dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                cfg.model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        self.model.to(cfg.device)
        self.model.eval()
