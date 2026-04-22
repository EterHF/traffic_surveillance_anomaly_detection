from __future__ import annotations

import json


def build_span_score_prompt(summary: dict) -> str:
    """Prompt for reusable span scoring on both coarse and fine candidates."""

    return (
        "You are a traffic anomaly scoring assistant.\n\n"
        "You are given a candidate segment from a short traffic surveillance clip.\n"
        "The image is a montage of sampled frames in chronological order.\n"
        "Your task is to estimate how likely this segment contains the core abnormal traffic event.\n\n"
        f"Structured summary: {json.dumps(summary, ensure_ascii=False)}\n\n"
        "Return JSON only with keys:\n"
        "- anomaly_score: float in [0,1], where 0 means normal/context and 1 means clearly abnormal core segment\n"
        "- confidence: float in [0,1]\n"
        "- reason: one short sentence\n\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- Use the prior_score in the summary only as a weak hint, not as the final answer.\n"
        "- Score high when the sampled frames likely contain the abnormal body, setup or aftermath.\n"
        "- If the segment looks mostly normal or ambiguous, return a moderate or low anomaly_score.\n"
    )

