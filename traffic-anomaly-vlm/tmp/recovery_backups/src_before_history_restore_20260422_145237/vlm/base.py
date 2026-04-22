from abc import ABC, abstractmethod
from src.schemas import EvidencePack, VLMResult

class BaseVLMClient(ABC):
    @abstractmethod
    def infer_two_stage(self, evidence: EvidencePack) -> VLMResult:
        """
        Execute two-stage prompting Strategy:
        1. Contextual reading.
        2. Diagnosis/Anomaly Verdict.
        """
        pass
