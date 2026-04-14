from src.perception.track_cache import TrackCache


def test_track_cache_empty():
    c = TrackCache(max_frames=3)
    assert c.get_window(2) == []
