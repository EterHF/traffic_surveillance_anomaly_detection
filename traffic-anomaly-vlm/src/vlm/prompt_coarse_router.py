from __future__ import annotations

import json


def build_coarse_router_prompt(summary: dict) -> str:
    return (
        "You are a traffic anomaly routing assistant.\n\n"
        "You are given a coarse candidate segment from a short traffic surveillance clip, together with:\n"
        "1. a raw montage of keyframes in time order\n"
        "2. an overlay montage with detected objects and focus track hints\n"
        "3. a local crop montage around the likely interaction region\n\n"
        "Your job is NOT to produce the final anomaly verdict yet. "
        "Instead, decide whether this coarse segment contains the main anomaly body, and whether the next stage should "
        "keep the coarse node or split it into finer nodes.\n\n"
        f"Structured summary: {json.dumps(summary, ensure_ascii=False)}\n\n"
        "Return JSON only with keys:\n"
        "- contains_core_anomaly: bool\n"
        "- role: one of [\"core\", \"context\", \"transition\", \"uncertain\"]\n"
        "- temporal_pattern: one of [\"localized\", \"sustained\", \"multi_stage\", \"uncertain\"]\n"
        "- focus_region: [start_ratio, end_ratio] within [0,1], describing where the anomaly body most likely lies inside this coarse segment\n"
        "- split_recommendation: one of [\"keep_coarse\", \"split_2\", \"split_3\"]\n"
        "- boundary_adjust: {\"left\": int, \"right\": int}, small frame offsets to expand or shrink the current segment\n"
        "- confidence: float in [0,1]\n"
        "- reason: one short sentence\n\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- If the anomaly appears to persist through most of the segment, use role=\"core\", temporal_pattern=\"sustained\", and split_recommendation=\"keep_coarse\".\n"
        "- If the segment mainly contains setup or aftermath instead of the anomaly body, use role=\"context\" or \"transition\".\n"
        "- If the anomaly seems concentrated around a short moment, use temporal_pattern=\"localized\" and a narrow focus_region.\n"
        "- Keep boundary_adjust conservative, usually within [-12, 12].\n"
    )
