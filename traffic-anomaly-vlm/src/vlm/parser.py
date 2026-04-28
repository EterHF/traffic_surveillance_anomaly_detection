from __future__ import annotations

import json
from typing import Any


def _extract_json_obj(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        payload = raw[start : end + 1] if start >= 0 and end > start else raw
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}



def parse_vlm_output(raw: str):
    pass
    
# For pure vlm pipeline, use with build_pure_stage1_prompt
def parse_stage1_output(raw: str, chunks: int = 4) -> dict[str, Any]:
    obj = _extract_json_obj(raw)
    chunk_desc = obj.get("chunk_descriptions", [])
    if not isinstance(chunk_desc, list):
        chunk_desc = []
    if len(chunk_desc) < chunks:
        chunk_desc = list(chunk_desc) + [""] * (chunks - len(chunk_desc))
    chunk_desc = [str(x) for x in chunk_desc[:chunks]]

    likely = obj.get("likely_abnormal_chunks", [])
    if not isinstance(likely, list):
        likely = []
    likely = [int(x) for x in likely if isinstance(x, (int, float, str)) and str(x).isdigit()]
    likely = [x for x in likely if 1 <= x <= chunks]

    risk_hint = str(obj.get("risk_hint", "normal"))
    if risk_hint not in {"normal", "possibly_risky", "clearly_risky"}:
        risk_hint = "normal"

    return {
        "main_objects": [str(x) for x in obj.get("main_objects", [])] if isinstance(obj.get("main_objects"), list) else [],
        "scene_summary": str(obj.get("scene_summary", "")),
        "chunk_descriptions": chunk_desc,
        "noticeable_change": bool(obj.get("noticeable_change", False)),
        "likely_abnormal_chunks": likely,
        "risk_hint": risk_hint,
    }

# For pure vlm pipeline, use with build_pure_stage2_prompt
def parse_stage2_output(raw: str, chunks: int = 4) -> dict[str, Any]:
    obj = _extract_json_obj(raw)

    scores = obj.get("chunk_scores", [])
    if not isinstance(scores, list):
        scores = []
    parsed_scores: list[float] = []
    for val in scores:
        try:
            parsed_scores.append(float(val))
        except Exception:
            parsed_scores.append(0.0)
    if len(parsed_scores) < chunks:
        parsed_scores += [0.0] * (chunks - len(parsed_scores))
    parsed_scores = [min(1.0, max(0.0, x)) for x in parsed_scores[:chunks]]

    abnormal_chunks = obj.get("abnormal_chunks", [])
    if not isinstance(abnormal_chunks, list):
        abnormal_chunks = []
    normalized_abnormal_chunks = []
    for c in abnormal_chunks:
        if isinstance(c, int):
            ci = c
        elif isinstance(c, float):
            ci = int(c)
        elif isinstance(c, str) and c.isdigit():
            ci = int(c)
        else:
            continue
        if 1 <= ci <= chunks:
            normalized_abnormal_chunks.append(ci)

    anomaly_type = str(obj.get("anomaly_type", "unknown"))
    allowed = {
        "collision_like",
        "dangerous_interaction",
        "abnormal_stop",
        "wrong_direction",
        "possible_overspeed",
        "fire_or_smoke",
        "road_obstruction",
        "normal",
        "unknown",
    }
    if anomaly_type not in allowed:
        anomaly_type = "unknown"

    try:
        overall_score = float(obj.get("overall_score", 0.0))
    except Exception:
        overall_score = 0.0
    overall_score = min(1.0, max(0.0, overall_score))

    try:
        confidence = float(obj.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    is_anomaly = bool(obj.get("is_anomaly", overall_score >= 0.5))

    # Some generations drift to "normal + high score" where score behaves like normality.
    # Re-orient scores to anomaly direction when categorical fields strongly indicate normal.
    if (not is_anomaly or anomaly_type == "normal") and (overall_score > 0.5 or any(s > 0.5 for s in parsed_scores)):
        overall_score = 1.0 - overall_score
        parsed_scores = [1.0 - s for s in parsed_scores]

    return {
        "is_anomaly": is_anomaly,
        "overall_score": overall_score,
        "chunk_scores": parsed_scores,
        "anomaly_type": anomaly_type,
        "abnormal_chunks": sorted(set(normalized_abnormal_chunks)),
        "confidence": confidence,
        "short_reason": str(obj.get("short_reason", "")),
        "supporting_evidence": [str(x) for x in obj.get("supporting_evidence", [])]
        if isinstance(obj.get("supporting_evidence"), list)
        else [],
    }

# Use with build_span_score_prompt, single-stage VLM scoring of spans
def parse_span_score_output(raw: str) -> dict[str, Any]:
    obj = _extract_json_obj(raw)
    try:
        anomaly_score = float(obj.get("anomaly_score", obj.get("score", obj.get("overall_score", 0.0))))
    except Exception:
        anomaly_score = 0.0
    anomaly_score = min(max(anomaly_score, 0.0), 1.0)

    try:
        confidence = float(obj.get("confidence", anomaly_score))
    except Exception:
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    return {
        "anomaly_score": anomaly_score,
        "confidence": confidence,
        "reason": str(obj.get("reason", "")),
    }


def parse_span_score_output_for_stage1(raw: str) -> dict[str, Any]:
    obj = _extract_json_obj(raw)
    anomaly_hint = str(obj.get("anomaly_hint", "normal"))
    if anomaly_hint not in {"normal", "possibly_abnormal", "clearly_abnormal"}:
        anomaly_hint = "normal"
    return {
        "scene_summary": str(obj.get("scene_summary", "")),
        "main_objects": [str(x) for x in obj.get("main_objects", [])]
        if isinstance(obj.get("main_objects"), list)
        else [],
        "motion_summary": str(obj.get("motion_summary", "")),
        "anomaly_hint": anomaly_hint,
        "evidence_tags": [str(x) for x in obj.get("evidence_tags", [])]
        if isinstance(obj.get("evidence_tags"), list)
        else [],
    }

def parse_span_score_output_for_stage2(raw: str) -> dict[str, Any]:
    parsed = parse_span_score_output(raw)
    obj = _extract_json_obj(raw)
    parsed["evidence_tags"] = [str(x) for x in obj.get("evidence_tags", [])] if isinstance(obj.get("evidence_tags"), list) else []
    return parsed
