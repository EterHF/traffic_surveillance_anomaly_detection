import cv2
import numpy as np
from src.schemas import Detection
from src.detectors.base import BaseDetector

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class CustomYOLODetector(BaseDetector):
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.5):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        if YOLO is None:
            print("Warning: ultralytics is not installed. YOLO detector will not work.")
            self.model = None
        else:
            self.model = YOLO(self.model_path)
        
    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.model is None:
            return []
            
        results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls_id]
                
                detections.append(Detection(
                    bbox_xyxy=xyxy,
                    score=conf,
                    cls_name=cls_name
                ))
                
        return detections
