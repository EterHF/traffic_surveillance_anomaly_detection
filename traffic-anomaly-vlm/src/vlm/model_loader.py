from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor


class LocalVLM:
    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "auto"):
        self.model_path = model_path
        self.device = device

        if dtype == "auto":
            torch_dtype = "auto"
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        self.model.to(device)
        self.model.eval()
