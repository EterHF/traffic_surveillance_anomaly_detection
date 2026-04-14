from src.schemas import TrackObject


def test_track_object_fields():
    t = TrackObject(
        frame_id=1,
        track_id=2,
        cls_id=0,
        cls_name="person",
        score=0.9,
        bbox_xyxy=[0.0, 0.0, 10.0, 20.0],
        cx=5.0,
        cy=10.0,
        w=10.0,
        h=20.0,
        area=200.0,
    )
    assert t.track_id == 2
