from typing import List, Optional
from src.schemas import EventProposal, EvidencePack, WindowFeature
from src.evidence.keyframe_selector import KeyframeSelector
from src.evidence.trajectory_sketch import TrajectorySketcher
from src.evidence.structured_summary import StructuredSummaryBuilder

class EvidencePackBuilder:
    def __init__(self):
        self.kf_selector = KeyframeSelector()
        self.sketcher = TrajectorySketcher()
        self.summary_builder = StructuredSummaryBuilder()
        
    def build(self, proposal: EventProposal, features: List[WindowFeature], save_dir: str) -> EvidencePack:
        """
        Orchestrate the creation of all evidence assets.
        In a real run, this would save extracted image patches/sketches to disk 
        and return the paths.
        """
        keyframes = self.kf_selector.select(proposal)
        
        # Mock paths for keyframes
        kf_paths = [f"{save_dir}/{proposal.event_id}_frame_{kf}.jpg" for kf in keyframes]
        
        summary = self.summary_builder.build(proposal, features)
        
        return EvidencePack(
            event_id=proposal.event_id,
            keyframe_paths=kf_paths,
            position_context_path=None,
            temporal_context_path=None,
            trajectory_sketch_path=None,
            structured_summary=summary
        )
