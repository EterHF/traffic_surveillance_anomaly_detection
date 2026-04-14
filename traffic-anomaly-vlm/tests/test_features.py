from src.features.track_features import build_track_features


def test_track_features_empty():
    feats = build_track_features([])
    assert isinstance(feats, dict)
