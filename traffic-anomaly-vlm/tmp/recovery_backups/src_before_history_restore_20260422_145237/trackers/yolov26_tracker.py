import numpy as np
from typing import List
from src.schemas import TrackObject

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class YOLOv26Tracker:
    """
    Unified Detector & Tracker using Ultralytics tracking API (e.g. model.track).
    This processes both detection and tracking in a single forward pass.
    """
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.5, tracker_type: str = "bytetrack.yaml"):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.tracker_type = tracker_type
        
        if YOLO is None:
            print("Warning: ultralytics is not installed. YOLOv26Tracker will not work.")
            self.model = None
        else:
            self.model = YOLO(model_path)
            
    def track(self, frame: np.ndarray, frame_id: int) -> List[TrackObject]:
        if self.model is None:
            return []
            
        # persist=True is required for frame-by-frame tracking in a continuous video stream
        results = self.model.track(
            frame, 
            persist=True, 
            conf=self.conf_thres, 
            iou=self.iou_thres, 
            tracker=self.tracker_type,
            verbose=False
        )
        
        tracked_objects = []
        for r in results:
            boxes = r.boxes
            # If no tracks are found or tracking IDs are missing
            if boxes is None or boxes.id is None:
                continue
                
            for box, track_id in zip(boxes, boxes.id):
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                t_id = int(track_id.cpu().numpy())
                cls_name = self.model.names[cls_id]
                
                x1, y1, x2, y2 = xyxy
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                area = w * h
                
                tracked_objects.append(TrackObject(
                    frame_id=frame_id,
                    track_id=t_id,
                    bbox_xyxy=xyxy,
                    score=conf,
                    cls_name=cls_name,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    area=area
                ))
                
        return tracked_objects
