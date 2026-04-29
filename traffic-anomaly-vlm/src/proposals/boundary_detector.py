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

    # Peak units use frame IDs when frame_ids are passed to detect/detect_details.
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

    def detect(self, scores: list[float] | np.ndarray, frame_ids: list[int] | np.ndarray | None = None) -> list[Span]:
        return self.detect_details(scores, frame_ids=frame_ids).spans

    def detect_details(
        self,
        scores: list[float] | np.ndarray,
        frame_ids: list[int] | np.ndarray | None = None,
    ) -> BoundaryDetection:
        values = self._prepare_scores(scores)
        if values.size == 0:
            return BoundaryDetection(peaks=[], spans=[], thresholds={})
        frame_positions = self._prepare_frame_ids(frame_ids, n=int(values.size))

        thresholds = self._thresholds(values)
        thresholds["max_score"] = float(np.max(values))
        if float(np.max(values)) < self.min_peak_score:
            # Null-aware gate upstream can legitimately produce no reliable
            # peaks; keep only fixed anchors in that case.
            thresholds["peak_high"] = float(thresholds["high"])
            thresholds["num_peaks"] = 0.0
            anchors = self._anchor_spans(n=int(values.size), frame_ids=frame_positions)
            thresholds["num_anchor_spans"] = float(len(anchors))
            return BoundaryDetection(peaks=[], spans=anchors, thresholds=thresholds)

        peaks = self._adaptive_peaks(values, thresholds, frame_positions)
        spans = [
            self._span_around_peak(peak, n=int(values.size), frame_ids=frame_positions)
            for peak in peaks
        ]
        kept = self._greedy_keep(peaks, spans, values, frame_positions)
        kept_spans = [span for _, span in kept]
        out_spans = self._dedupe_anchor_spans(
            kept_spans,
            n=int(values.size),
            frame_ids=frame_positions,
        )

        thresholds["num_peaks"] = float(len(kept))
        thresholds["num_anchor_spans"] = float(len(out_spans) - len(kept_spans))
        return BoundaryDetection(
            peaks=[peak for peak, _ in kept],
            spans=out_spans,
            thresholds=thresholds,
        )

    @staticmethod
    def _prepare_scores(scores: list[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float32)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _prepare_frame_ids(frame_ids: list[int] | np.ndarray | None, n: int) -> np.ndarray | None:
        if frame_ids is None:
            return None
        positions = np.asarray(frame_ids, dtype=np.int64)
        if positions.size != int(n):
            raise ValueError(f"frame_ids length ({positions.size}) must match scores length ({int(n)})")
        if positions.size > 1 and bool(np.any(np.diff(positions) < 0)):
            raise ValueError("frame_ids must be sorted in ascending order")
        return positions

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

    def _adaptive_peaks(
        self,
        values: np.ndarray,
        thresholds: dict[str, float],
        frame_ids: np.ndarray | None = None,
    ) -> list[int]:
        candidates = self._local_peaks(values)
        if not candidates:
            thresholds["peak_high"] = float(thresholds["high"])
            return []

        high = float(thresholds["high"])
        peaks = [idx for idx in candidates if float(values[idx]) >= max(high, self.min_peak_score)]
        peaks = self._suppress_close_peaks(peaks, values, frame_ids)

        thresholds["peak_high"] = float(min((float(values[p]) for p in peaks), default=high))
        return sorted(int(p) for p in peaks)

    def _suppress_close_peaks(
        self,
        peaks: list[int],
        values: np.ndarray,
        frame_ids: np.ndarray | None = None,
    ) -> list[int]:
        kept: list[int] = []
        for peak in sorted(peaks, key=lambda idx: (float(values[idx]), -idx), reverse=True):
            if all(
                abs(self._peak_position(peak, frame_ids) - self._peak_position(prev, frame_ids)) > self.peak_gap
                for prev in kept
            ):
                kept.append(int(peak))
        return kept

    def _span_around_peak(self, peak: int, n: int, frame_ids: np.ndarray | None = None) -> Span:
        left, right = self.peak_expand
        if frame_ids is not None:
            peak_idx = max(0, min(int(peak), int(n) - 1))
            center = int(frame_ids[peak_idx])
            start_frame = center - int(left)
            end_frame = center + int(right)
            start_idx = int(np.searchsorted(frame_ids, start_frame, side="left"))
            end_idx = int(np.searchsorted(frame_ids, end_frame, side="right")) - 1
            start_idx = max(0, min(start_idx, int(n) - 1))
            end_idx = max(0, min(end_idx, int(n) - 1))
            if start_idx > peak_idx:
                start_idx = peak_idx
            if end_idx < peak_idx:
                end_idx = peak_idx
            return _clamp_span(start_idx, end_idx, n)
        return _clamp_span(int(peak) - left, int(peak) + right, n)

    def _anchor_spans(self, n: int, frame_ids: np.ndarray | None = None) -> list[Span]:
        if n <= 0:
            return []
        anchors = self._anchor_indices(n=int(n), frame_ids=frame_ids)
        spans: list[Span] = []
        for anchor in anchors:
            span = self._span_around_peak(peak=int(anchor), n=int(n), frame_ids=frame_ids)
            if span not in spans:
                spans.append(span)
        return spans

    @staticmethod
    def _anchor_indices(n: int, frame_ids: np.ndarray | None = None) -> list[int]:
        if n <= 0:
            return []
        if frame_ids is None:
            return [int(idx) for idx in np.linspace(0, int(n) - 1, num=4, dtype=int)]

        targets = np.linspace(int(frame_ids[0]), int(frame_ids[-1]), num=4)
        frame_values = frame_ids.astype(np.float64)
        anchors: list[int] = []
        for target in targets:
            idx = int(np.argmin(np.abs(frame_values - float(target))))
            if idx not in anchors:
                anchors.append(idx)
        return anchors

    def _dedupe_anchor_spans(
        self,
        spans: list[Span],
        n: int,
        frame_ids: np.ndarray | None = None,
    ) -> list[Span]:
        out = list(spans)
        for anchor in self._anchor_spans(n, frame_ids=frame_ids):
            if any(self._too_close(anchor, span, frame_ids) for span in out):
                continue
            out.append(anchor)
        return sorted(out, key=lambda span: span[0])

    def _greedy_keep(
        self,
        peaks: list[int],
        spans: list[Span],
        values: np.ndarray,
        frame_ids: np.ndarray | None = None,
    ) -> list[tuple[int, Span]]:
        kept: list[tuple[int, Span]] = []
        items = sorted(
            zip(peaks, spans),
            key=lambda item: (float(values[int(item[0])]), -int(item[0])),
            reverse=True,
        )
        for peak, span in items:
            # If expanded intervals overlap or are too close, keep the stronger one.
            if any(self._too_close(span, kept_span, frame_ids) for _, kept_span in kept):
                continue
            kept.append((int(peak), span))
            # if len(kept) >= self.max_peaks:
            #     break
        return sorted(kept, key=lambda item: item[1][0])

    def _too_close(self, a: Span, b: Span, frame_ids: np.ndarray | None = None) -> bool:
        a_start, a_end = self._span_positions(a, frame_ids)
        b_start, b_end = self._span_positions(b, frame_ids)
        return a_start <= b_end + self.merge_gap and b_start <= a_end + self.merge_gap

    @staticmethod
    def _peak_position(peak: int, frame_ids: np.ndarray | None = None) -> int:
        if frame_ids is None:
            return int(peak)
        peak_idx = max(0, min(int(peak), int(frame_ids.size) - 1))
        return int(frame_ids[peak_idx])

    @staticmethod
    def _span_positions(span: Span, frame_ids: np.ndarray | None = None) -> Span:
        if frame_ids is None:
            return int(span[0]), int(span[1])
        start_idx = max(0, min(int(span[0]), int(frame_ids.size) - 1))
        end_idx = max(0, min(int(span[1]), int(frame_ids.size) - 1))
        return int(frame_ids[start_idx]), int(frame_ids[end_idx])
