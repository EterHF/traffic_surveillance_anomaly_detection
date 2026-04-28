from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


Span = tuple[int, int]


@dataclass
class BoundaryDetection:
    """Peak-centered proposal result."""

    peaks: list[int]
    spans: list[Span]
    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class BoundaryDetectorConfig:
    """Minimal config for peak-centered boundary proposals."""

    # Start from a robust threshold, then relax/trim to keep roughly 5-10 peaks.
    high_z: float = 0.85
    min_peak_score: float = 0.15
    min_peaks: int = 5
    max_peaks: int = 10

    # All units are sampled-frame indices.
    peak_gap: int = 2
    peak_expand: tuple[int, int] = (8, 12)
    merge_gap: int = 0


def _median(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.median(values.astype(np.float32)))


def _clamp_span(start: int, end: int, n: int) -> Span:
    start = max(0, min(int(start), n - 1))
    end = max(0, min(int(end), n - 1))
    if start > end:
        start, end = end, start
    return start, end


class BoundaryDetector:
    """Convert a 1D anomaly score curve into compact peak-centered spans."""

    def __init__(self, cfg: BoundaryDetectorConfig | None = None):
        self.cfg = cfg or BoundaryDetectorConfig()
        self.high_z = float(self.cfg.high_z)
        self.min_peak_score = max(0.0, float(self.cfg.min_peak_score))
        self.min_peaks = max(1, int(self.cfg.min_peaks))
        self.max_peaks = max(self.min_peaks, int(self.cfg.max_peaks))
        self.peak_gap = max(1, int(self.cfg.peak_gap))
        self.peak_expand = self._coerce_pair(self.cfg.peak_expand, default=(8, 12))
        self.merge_gap = max(0, int(self.cfg.merge_gap))

    @staticmethod
    def _coerce_pair(value: tuple[int, int] | list[int], default: tuple[int, int]) -> tuple[int, int]:
        try:
            if len(value) >= 2:
                return max(0, int(value[0])), max(0, int(value[1]))
        except Exception:
            pass
        return default

    def detect(self, scores: list[float] | np.ndarray) -> list[Span]:
        return self.detect_details(scores).spans

    def detect_details(self, scores: list[float] | np.ndarray) -> BoundaryDetection:
        values = self._prepare_scores(scores)
        if values.size == 0:
            return BoundaryDetection(peaks=[], spans=[], thresholds={})

        thresholds = self._thresholds(values)
        thresholds["max_score"] = float(np.max(values))
        if float(np.max(values)) < self.min_peak_score:
            # Null-aware gate upstream can legitimately produce no reliable
            # evidence. In that case do not force proposals from noise.
            thresholds["peak_high"] = float(thresholds["high"])
            thresholds["num_peaks"] = 0.0
            return BoundaryDetection(peaks=[], spans=[], thresholds=thresholds)

        peaks = self._adaptive_peaks(values, thresholds)
        spans = [self._span_around_peak(peak, n=int(values.size)) for peak in peaks]
        kept = self._greedy_keep(peaks, spans, values)

        thresholds["num_peaks"] = float(len(kept))
        return BoundaryDetection(
            peaks=[peak for peak, _ in kept],
            spans=[span for _, span in kept],
            thresholds=thresholds,
        )

    @staticmethod
    def _prepare_scores(scores: list[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float32)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _thresholds(self, values: np.ndarray) -> dict[str, float]:
        med = _median(values)
        mad = _median(np.abs(values.astype(np.float32) - med))
        sigma = max(1e-6, 1.4826 * mad)
        high = med + self.high_z * sigma
        return {
            "median": float(med),
            "sigma": float(sigma),
            "high": float(high),
        }

    @staticmethod
    def _local_peaks(values: np.ndarray) -> list[int]:
        n = int(values.size)
        if n == 0:
            return []
        if n == 1:
            return [0]

        peaks: list[int] = []
        for idx in range(n):
            left = float(values[idx - 1]) if idx > 0 else float("-inf")
            right = float(values[idx + 1]) if idx + 1 < n else float("-inf")
            center = float(values[idx])
            if center >= left and center >= right:
                peaks.append(idx)
        return peaks

    def _adaptive_peaks(self, values: np.ndarray, thresholds: dict[str, float]) -> list[int]:
        candidates = self._local_peaks(values)
        if not candidates:
            thresholds["peak_high"] = float(thresholds["high"])
            return []

        high = float(thresholds["high"])
        peaks = [idx for idx in candidates if float(values[idx]) >= max(high, self.min_peak_score)]
        peaks = self._suppress_close_peaks(peaks, values)

        thresholds["peak_high"] = float(min((float(values[p]) for p in peaks), default=high))
        return sorted(int(p) for p in peaks)

    def _suppress_close_peaks(self, peaks: list[int], values: np.ndarray) -> list[int]:
        kept: list[int] = []
        for peak in sorted(peaks, key=lambda idx: (float(values[idx]), -idx), reverse=True):
            if all(abs(int(peak) - int(prev)) > self.peak_gap for prev in kept):
                kept.append(int(peak))
        return kept

    def _span_around_peak(self, peak: int, n: int) -> Span:
        left, right = self.peak_expand
        return _clamp_span(int(peak) - left, int(peak) + right, n)

    def _greedy_keep(
        self,
        peaks: list[int],
        spans: list[Span],
        values: np.ndarray,
    ) -> list[tuple[int, Span]]:
        kept: list[tuple[int, Span]] = []
        items = sorted(
            zip(peaks, spans),
            key=lambda item: (float(values[int(item[0])]), -int(item[0])),
            reverse=True,
        )
        for peak, span in items:
            # If expanded intervals overlap or are too close, keep the stronger one.
            if any(self._too_close(span, kept_span) for _, kept_span in kept):
                continue
            kept.append((int(peak), span))
            # if len(kept) >= self.max_peaks:
            #     break
        return sorted(kept, key=lambda item: item[1][0])

    def _too_close(self, a: Span, b: Span) -> bool:
        return a[0] <= b[1] + self.merge_gap and b[0] <= a[1] + self.merge_gap
