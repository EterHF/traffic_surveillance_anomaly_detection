from src.triggers.boundary import BoundaryDetector


def test_boundary_detect():
    b = BoundaryDetector(high=1.0, low=0.5)
    out = b.detect([0.1, 1.2, 0.8, 0.3])
    assert out == [(1, 2)]
