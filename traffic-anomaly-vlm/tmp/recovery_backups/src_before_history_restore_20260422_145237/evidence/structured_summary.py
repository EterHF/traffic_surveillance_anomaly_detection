from typing import Dict, Any
from src.schemas import EventProposal, WindowFeature

class StructuredSummaryBuilder:
    def build(self, proposal: EventProposal, features: List[WindowFeature]) -> Dict[str, Any]:
        """
        Build a JSON-friendly structured summary for the VLM prompt.
        """
        summary = {
            "event_id": proposal.event_id,
            "duration_frames": proposal.end_frame - proposal.start_frame,
            "main_actor_id": proposal.main_track_id,
            "max_severity_score": proposal.scores.get("max_trigger", 0.0),
            "description": "Auto-generated structured summary template."
        }
        return summary
