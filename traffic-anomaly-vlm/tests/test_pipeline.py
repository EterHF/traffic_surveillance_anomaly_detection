from src.proposals.boundary_detector import BoundaryDetector


def test_boundary_detect():
    b = BoundaryDetector(high=1.0, low=0.5, method="by_thres", peak_expand=(0, 0), min_span_len=1, merge_gap=0)
    out = b.detect([0.1, 1.2, 0.8, 0.3])
    assert out == [(1, 2)]
