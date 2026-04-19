from __future__ import annotations

from src.schemas import TrackObject


def parse_ultralytics_results(results, frame_id: int) -> list[TrackObject]:
    out: list[TrackObject] = []
    for r in results:
        boxes = r.boxes
        if boxes is None or boxes.id is None:
            continue

        frame_h, frame_w = 0.0, 0.0
        if hasattr(r, "orig_shape") and r.orig_shape is not None and len(r.orig_shape) >= 2:
            frame_h = float(r.orig_shape[0])
            frame_w = float(r.orig_shape[1])

        for box, tid in zip(boxes, boxes.id):
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            track_id = int(tid.cpu().numpy())

            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            out.append(
                TrackObject(
                    frame_id=frame_id,
                    track_id=track_id,
                    cls_id=cls_id,
                    cls_name=str(r.names.get(cls_id, cls_id)),
                    score=conf,
                    bbox_xyxy=xyxy,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    area=w * h,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )
            )
    return out
