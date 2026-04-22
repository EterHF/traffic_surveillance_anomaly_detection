from __future__ import annotations

from PIL import Image

from src.vlm.model_loader import LocalVLM


def _limit_image_resolution(img: Image.Image, max_image_size: int) -> Image.Image:
    if max_image_size <= 0:
        return img
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_image_size:
        return img
    scale = float(max_image_size) / float(long_side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.BICUBIC)


def run_inference(
    vlm: LocalVLM,
    prompt: str,
    image_paths: list[str]
) -> str:
    images = []
    max_image_size = vlm.max_image_size
    for p in image_paths:
        if not p:
            continue
        img = Image.open(p).convert("RGB")
        if max_image_size > 0:
            img = _limit_image_resolution(img, max_image_size=max_image_size)
        images.append(img)

    # Qwen3-VL style input: build chat-template text with explicit image placeholders.
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": prompt}],
        }
    ]
    text_input = vlm.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = vlm.processor(text=[text_input], images=images, return_tensors="pt")
    inputs = {k: v.to(vlm.device) for k, v in inputs.items()}

    try:
        output_ids = vlm.model.generate(
            **inputs,
            max_new_tokens=vlm.max_new_tokens,
            use_cache=False,
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        retry_tokens = max(16, int(vlm.max_new_tokens // 2))
        output_ids = vlm.model.generate(
            **inputs,
            max_new_tokens=retry_tokens,
            use_cache=False,
        )

    # Trim prompt tokens and decode only newly generated part.
    input_ids = inputs.get("input_ids", None)
    if input_ids is not None:
        trimmed = [out[len(inp) :] for inp, out in zip(input_ids, output_ids)]
    else:
        trimmed = output_ids

    text = vlm.processor.batch_decode(trimmed, skip_special_tokens=True)
    first = text[0].strip() if text else ""
    if first:
        return first

    # Some model/processor pairs may return already-trimmed generations.
    full = vlm.processor.batch_decode(output_ids, skip_special_tokens=True)
    first_full = full[0].strip() if full else ""
    return first_full
