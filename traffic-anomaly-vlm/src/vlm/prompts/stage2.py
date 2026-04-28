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


def build_stage2_prompt(stage1_text: dict | str, summary: dict | None = None, evidence_method: str = "montage") -> str:
    return (
        "You are a traffic anomaly judge.\n\n"
        "You are given a candidate traffic segment.\n"
        f"The visual evidence is { _evidence_desc(evidence_method) }.\n"
        f"Structured summary: {json.dumps(summary or {}, ensure_ascii=False)}\n"
        f"Stage-1 analysis: {json.dumps(stage1_text, ensure_ascii=False) if isinstance(stage1_text, dict) else str(stage1_text)}\n\n"
        "Return JSON only with keys:\n"
        "- anomaly_score: float in [0,1]\n"
        "- confidence: float in [0,1]\n"
        "- reason: one short sentence\n"
        "- evidence_tags: short list such as [\"setup\", \"impact\", \"aftermath\", \"context\"]\n"
    )


def build_pure_stage2_prompt(stage1_output: dict, chunks_per_window: int = 4) -> str:
    return (
        "You are a traffic anomaly detection judge.\n\n"
        "You are given:\n"
        "1. the same traffic surveillance clip\n"
        "2. a structured neutral description generated in the previous step\n\n"
        f"The clip is divided into {chunks_per_window} temporal chunks in order.\n"
        f"Stage-1 structured description: {json.dumps(stage1_output, ensure_ascii=False)}\n\n"
        "Please output a JSON object with:\n"
        "- is_anomaly: true or false\n"
        "- overall_score: anomaly score in [0,1], where 0 means clearly normal and 1 means clearly anomalous\n"
        f"- chunk_scores: a list of {chunks_per_window} anomaly scores in [0,1] (same direction: higher means more anomalous)\n"
        "- anomaly_type: one of [\"collision_like\", \"dangerous_interaction\", \"abnormal_stop\", "
        "\"wrong_direction\", \"possible_overspeed\", \"fire_or_smoke\", \"road_obstruction\", "
        "\"normal\", \"unknown\"]\n"
        f"- abnormal_chunks: a list of chunk indices among [1..{chunks_per_window}]\n"
        "- confidence: a float in [0,1]\n"
        "- short_reason: one sentence\n"
        "- supporting_evidence: a short list of evidence phrases\n\n"
        "Rules:\n"
        "- If the clip is normal, set anomaly_type to \"normal\".\n"
        "- If is_anomaly is false or anomaly_type is \"normal\", overall_score and all chunk_scores should be low (typically <= 0.3).\n"
        "- If is_anomaly is true, anomaly_type should not be \"normal\" and at least one chunk score should be high (typically >= 0.6).\n"
        f"- chunk_scores must contain exactly {chunks_per_window} numbers.\n"
        "- Output JSON only.\n"
    )
