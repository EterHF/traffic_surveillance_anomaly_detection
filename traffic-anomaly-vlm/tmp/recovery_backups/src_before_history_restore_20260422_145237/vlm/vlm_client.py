from src.schemas import EvidencePack, VLMResult
from src.vlm.base import BaseVLMClient
from src.vlm.prompt_stage1 import Stage1Prompt
from src.vlm.prompt_stage2 import Stage2Prompt
from src.vlm.parser import VLMParser
import time

class DummyVLMClient(BaseVLMClient):
    """
    Dummy client for the skeleton to simulate VLM latency and response.
    Replace with actual API call (e.g. Qwen-VL, GPT-4V) in real usage.
    """
    def __init__(self, model_name: str = "dummy"):
        self.model_name = model_name

    def infer_two_stage(self, evidence: EvidencePack) -> VLMResult:
        # Stage 1
        p1 = Stage1Prompt.generate(evidence.structured_summary)
        print(f"[VLM] Running stage 1 on event {evidence.event_id}...")
        time.sleep(0.5)
        mock_s1 = "A vehicle is seen moving rapidly towards a pedestrian crossing."
        
        # Stage 2
        p2 = Stage2Prompt.generate(mock_s1)
        print(f"[VLM] Running stage 2 on event {evidence.event_id}...")
        time.sleep(0.5)
        mock_s2_json = '''
        {
            "is_anomaly": true,
            "event_type": "dangerous_approach",
            "confidence": 0.85,
            "summary": "Vehicle approached crosswalk without yielding.",
            "supporting_evidence": ["High speed near crosswalk"]
        }
        '''
        
        return VLMParser.parse_stage2(mock_s2_json)
