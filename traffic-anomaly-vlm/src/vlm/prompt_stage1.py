from __future__ import annotations
import json


def build_stage1_prompt(summary: dict) -> str:
    return (
        "You are an expert traffic analyst. "
        "Describe the scene neutrally: actors, motion, and key spatial-temporal relations. "
        f"Structured summary: {summary}"
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


def build_pure_stage1_with_context(context: dict) -> str:
    return (
        "You are a traffic surveillance video analyst. "
        "Analyze the clip and return JSON only. "
        f"Context: {json.dumps(context, ensure_ascii=False)}"
    )
