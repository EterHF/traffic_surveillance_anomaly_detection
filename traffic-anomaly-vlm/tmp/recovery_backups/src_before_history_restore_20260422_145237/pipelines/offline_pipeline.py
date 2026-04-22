from typing import Any
import os
from src.schemas import FinalEventResult
from src.io.video_reader import VideoReader
from src.io.frame_sampler import FrameSampler
from src.trackers.yolov26_tracker import YOLOv26Tracker
from src.features.window_feature_builder import WindowFeatureBuilder
from src.triggers.boundary_detector import BoundaryDetector
from src.triggers.event_builder import EventBuilder
from src.evidence.evidence_pack_builder import EvidencePackBuilder
from src.vlm.vlm_client import DummyVLMClient

class OfflinePipeline:
    def __init__(self, config: Any):
        self.config = config
        print(f"Initializing OfflinePipeline with joint detector & tracker (YOLO tracking API)")
        
        # IO
        self.sampler = FrameSampler(source_fps=30.0, target_fps=self.config.video.fps_sample)
        
        # Perception (Joint Detection + Tracking)
        self.tracker = YOLOv26Tracker(
            model_path=self.config.detector.model_path,
            conf_thres=self.config.detector.conf_thres,
            iou_thres=self.config.detector.iou_thres,
            tracker_type="bytetrack.yaml" # or "botsort.yaml"
        )
        
        # Features & Triggers
        self.feature_builder = WindowFeatureBuilder(window_size=self.config.video.window_size)
        self.boundary_detector = BoundaryDetector(
            high_thresh=self.config.trigger.boundary.high,
            low_thresh=self.config.trigger.boundary.low
        )
        self.event_builder = EventBuilder()

        
        # Evidence & VLM
        self.evidence_builder = EvidencePackBuilder()
        self.vlm_client = DummyVLMClient(model_name=self.config.vlm.model_name)
        
        # Outputs
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.save_dir = os.path.join(self.config.output_dir, "events")
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self, video_path: str) -> list[FinalEventResult]:
        print(f"Running pipeline on video: {video_path}")
        
        frame_buffer = []
        track_buffer = []
        all_features = []
        
        print("Simulating real video processing loop...")
        # Since we might not have a real video accessible, we demonstrate the loop
        # using the integrated tracker. But we will fallback to Dummy features for testing.
        
        # Real implementation would be something like:
        # reader = VideoReader(video_path)
        # for frame_id, frame in self.sampler.sample(reader):
        #     # Joint Detection & Tracking
        #     tracks = self.tracker.track(frame, frame_id)
        #     
        #     frame_buffer.append(frame)
        #     track_buffer.append(tracks)
        #     
        #     if self.feature_builder.ready(frame_buffer, track_buffer):
        #         feat = self.feature_builder.build(frame_buffer, track_buffer)
        #         all_features.append(feat)
        #         frame_buffer.pop(0)
        #         track_buffer.pop(0)

        # --------------------- Dummy fallback for execution testing ---------------------
        print("Running Boundary Detection & Event Building using integrated data...")
        from src.schemas import WindowFeature
        dummy_features = [
            WindowFeature(i, i*4, i*4+16, {}, max(0, 3.0 - abs(i-5)*0.5)) for i in range(10)
        ]
        
        scores = [f.trigger_score for f in dummy_features]
        boundaries = self.boundary_detector.detect(scores)
        proposals = self.event_builder.build(boundaries, dummy_features)

        
        final_results = []
        for prop in proposals:
            print(f"Building evidence for proposal {prop.event_id}...")
            ev_pack = self.evidence_builder.build(prop, dummy_features, self.save_dir)
            
            print(f"Calling VLM for event {prop.event_id}...")
            vlm_res = self.vlm_client.infer_two_stage(ev_pack)
            
            final_res = FinalEventResult(
                event_id=prop.event_id,
                proposal=prop,
                evidence=ev_pack,
                vlm_result=vlm_res
            )
            final_results.append(final_res)
            
            print(f"--> Event Result: {vlm_res.event_type} (Anomalous? {vlm_res.is_anomaly})")
            
        return final_results
