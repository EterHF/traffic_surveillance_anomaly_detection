from __future__ import annotations

import json


def build_stage2_prompt(stage1_text: str) -> str:
    return (
        "Given the neutral description below, decide if this is a traffic anomaly. "
        "Return strict JSON with keys: is_anomaly, event_type, confidence, summary, supporting_evidence, counter_evidence. "
        f"Description: {stage1_text}"
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
