from src.schemas import TrackObject
from typing import Dict, List, Any

def extract_track_primitives(tracks: List[TrackObject]) -> Dict[str, Any]:
    """
    Extract basic primitive features from a list of track objects belonging to the same ID.
    """
    if not tracks:
        return {}
        
    cxs = [t.cx for t in tracks]
    cys = [t.cy for t in tracks]
    
    # Calculate simple velocities
    if len(tracks) > 1:
        vx = cxs[-1] - cxs[0]
        vy = cys[-1] - cys[0]
        speed = (vx**2 + vy**2)**0.5
    else:
        vx, vy, speed = 0.0, 0.0, 0.0
        
    return {
        "start_pos": (cxs[0], cys[0]),
        "end_pos": (cxs[-1], cys[-1]),
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "avg_area": sum(t.area for t in tracks) / len(tracks)
    }
