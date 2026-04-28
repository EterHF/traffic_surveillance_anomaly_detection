from __future__ import annotations

import json


def _evidence_desc(evidence_method: str) -> str:
    if evidence_method == "montage":
        return "a single montage image composed of sampled frames in chronological order"
    if evidence_method == "frames":
        return "multiple sampled frames in chronological order"
    if evidence_method == "enhanced":
        return "multiple sampled frames with tracked objects, ids, boxes, and trajectory overlays"
    return "chronological visual evidence from a candidate traffic segment"


def build_span_score_prompt(summary: dict, evidence_method: str = "montage") -> str:
    """Single-stage prompt for direct span-level anomaly scoring."""

    return (
        "You are a traffic anomaly scoring assistant.\n\n"
        "You are given a candidate traffic segment.\n"
        f"The visual evidence is { _evidence_desc(evidence_method) }.\n"
        "This segment may show setup, impact, aftermath, or only normal context.\n\n"
        f"Structured summary: {json.dumps(summary, ensure_ascii=False)}\n\n"
        "Return JSON only with keys:\n"
        "- anomaly_score: float in [0,1]\n"
        "- confidence: float in [0,1]\n"
        "- reason: one short sentence\n\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- High score means the segment likely contains meaningful abnormal evidence, not necessarily only the impact moment.\n"
        "- Use the summary only as a weak hint.\n"
    )
