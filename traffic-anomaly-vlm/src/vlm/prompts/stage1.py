from __future__ import annotations

import json


def _evidence_desc(evidence_method: str) -> str:
    if evidence_method == "montage":
        return "a montage image composed of sampled frames in time order"
    if evidence_method == "frames":
        return "multiple sampled frames in time order"
    if evidence_method == "enhanced":
        return "multiple sampled frames with object tracks, boxes, ids, and trajectory overlays"
    return "visual evidence from a candidate traffic segment"


def build_stage1_prompt(summary: dict, evidence_method: str = "montage") -> str:
    return (
        "You are an expert traffic analyst.\n\n"
        "You are given a candidate traffic segment.\n"
        f"The visual evidence is { _evidence_desc(evidence_method) }.\n"
        "Do not decide anomaly score yet. First summarize what is happening.\n\n"
        f"Structured summary: {json.dumps(summary, ensure_ascii=False)}\n\n"
        "Return JSON only with keys:\n"
        "- scene_summary: one short sentence\n"
        "- main_objects: short list of visible road users or objects\n"
        "- motion_summary: one short sentence about movement and interaction\n"
        "- anomaly_hint: one of [\"normal\", \"possibly_abnormal\", \"clearly_abnormal\"]\n"
        "- evidence_tags: short list such as [\"setup\", \"impact\", \"aftermath\", \"context\", \"occlusion\"]\n"
    )


def build_pure_stage1_prompt(chunks_per_window: int = 4, sampled_frames_per_chunk: int = 4) -> str:
    return (
        "You are a traffic surveillance video analyst.\n\n"
        f"You are given {chunks_per_window * sampled_frames_per_chunk} frames in temporal order from a short "
        "traffic surveillance clip. "
        f"These frames come from {chunks_per_window} consecutive temporal chunks, and each chunk contains "
        f"{sampled_frames_per_chunk} sampled frames.\n\n"
        "Your task in this step is NOT to output the final anomaly score yet.\n\n"
        "Please analyze the clip and output a JSON object with:\n"
        "- main_objects: major traffic participants visible in the clip\n"
        "- scene_summary: a concise summary of the whole clip\n"
        f"- chunk_descriptions: a list of {chunks_per_window} short descriptions, one for each temporal chunk\n"
        "- noticeable_change: whether there is any sudden motion, risky interaction, scene hazard, or "
        "abnormal transition\n"
        f"- likely_abnormal_chunks: a list of chunk indices among [1..{chunks_per_window}]\n"
        "- risk_hint: one of [\"normal\", \"possibly_risky\", \"clearly_risky\"]\n\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- Do not output anything outside JSON.\n"
    )
