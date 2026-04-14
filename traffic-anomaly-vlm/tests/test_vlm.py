from src.vlm.parser import parse_vlm_output
from src.vlm.parser import parse_stage1_output, parse_stage2_output


def test_vlm_parser_json():
    raw = '{"is_anomaly": true, "event_type": "test", "confidence": 0.8, "summary": "ok"}'
    res = parse_vlm_output(raw)
    assert res.is_anomaly is True
    assert res.event_type == "test"


def test_stage1_parser_defaults():
    raw = '{"scene_summary": "normal traffic"}'
    out = parse_stage1_output(raw, chunks=4)
    assert out["scene_summary"] == "normal traffic"
    assert len(out["chunk_descriptions"]) == 4


def test_stage2_parser_chunk_fixup():
    raw = '{"is_anomaly": true, "overall_score": 0.9, "chunk_scores": [0.1, 0.2]}'
    out = parse_stage2_output(raw, chunks=4)
    assert out["is_anomaly"] is True
    assert len(out["chunk_scores"]) == 4
    assert out["chunk_scores"][2] == 0.0
