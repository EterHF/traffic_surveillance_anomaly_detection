from __future__ import annotations

from typing import Literal

from scipy.signal import savgol_filter


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    n = len(xs)
    m = n // 2
    if n % 2 == 1:
        return float(xs[m])
    return float((xs[m - 1] + xs[m]) * 0.5)


def _mad(vals: list[float], med: float) -> float:
    if not vals:
        return 0.0
    dev = [abs(v - med) for v in vals]
    return _median(dev)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    merged: list[tuple[int, int]] = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


class BoundaryDetector:
    def __init__(
        self,
        high: float | None = None,
        low: float | None = None,
        online: bool = False,
        use_savgol_filter: bool = False,
        method: Literal["by_peeks", "by_thres"] = "by_peeks",
        # Robust z-score parameters for adaptive thresholding.
        high_z: float = 1.0,
        low_z: float = 0.5,
        # Online mode generally needs a lower threshold factor for recall.
        factor_online: float = 0.3,
        # Peak/span post-process parameters.
        peak_gap: int = 5,
        peak_expand: tuple[int, int] = (12, 25),
        min_span_len: int = 12,
        merge_gap: int = 25,
        savgol_window: int = 7,
        savgol_polyorder: int = 3,
    ):
        self.high = high
        self.low = low
        self.online = online
        self.use_savgol_filter = use_savgol_filter
        self.savgol_window = max(5, int(savgol_window) | 1)
        self.savgol_polyorder = max(1, int(savgol_polyorder))
        self.method = method
        self.high_z = float(high_z)
        self.low_z = float(low_z)
        self.factor_online = float(factor_online)
        self.peak_gap = max(1, int(peak_gap))
        self.peak_expand = (max(0, int(peak_expand[0])), max(0, int(peak_expand[1])))
        self.min_span_len = max(1, int(min_span_len))
        self.merge_gap = max(0, int(merge_gap))

    def detect(self, scores: list[float]) -> list[tuple[int, int]]:
        if not scores:
            return []

        if self.use_savgol_filter and len(scores) >= self.savgol_window:
            polyorder = min(self.savgol_polyorder, self.savgol_window - 1)
            scores = savgol_filter(scores, self.savgol_window, polyorder).tolist()

        if self.method == "by_peeks":
            peeks = self._detect_peeks(scores)
            spans = self._peeks_to_spans(peeks, gap=self.peak_gap, expand=self.peak_expand, n=len(scores))
        else:
            spans = self._detect_spans_by_adaptive_threshold(scores)
        refined_spans = self._refine_spans(spans, n=len(scores))

        if self.online:
            recent = [len(scores) - 10, len(scores) - 1]
            for i, (s, e) in enumerate(refined_spans):
                if s <= recent[1] <= e:
                    return True
        return refined_spans

    def _detect_peeks(self, scores: list[float]) -> list[int]:
        n = len(scores)
        if n == 1:
            return [0]

        med = _median(scores)
        sigma = max(1e-6, 1.4826 * _mad(scores, med))
        peak_thr = float(self.high) if self.high is not None else (med + self.high_z * sigma)

        peaks: list[int] = []
        for i in range(n):
            left = scores[i - 1] if i > 0 else float("-inf")
            right = scores[i + 1] if i + 1 < n else float("-inf")
            if scores[i] >= left and scores[i] >= right and scores[i] >= peak_thr:
                peaks.append(i)

        # Fallback: keep at least one candidate peak for potential anomaly proposals.
        if not peaks:
            peaks = [int(max(range(n), key=lambda j: scores[j]))]
        return peaks

    def _peeks_to_spans(self, peeks: list[int], gap: int, expand: tuple[int, int], n: int) -> list[tuple[int, int]]:
        if not peeks:
            return []

        peeks = sorted(peeks)
        clusters: list[list[int]] = [[peeks[0]]]
        for p in peeks[1:]:
            if p - clusters[-1][-1] <= gap:
                clusters[-1].append(p)
            else:
                clusters.append([p])

        left_expand, right_expand = expand
        spans: list[tuple[int, int]] = []
        for c in clusters:
            s = max(0, c[0] - left_expand)
            e = min(n - 1, c[-1] + right_expand)
            spans.append((s, e))
        return spans

    def _detect_spans_by_adaptive_threshold(self, scores: list[float]) -> list[tuple[int, int]]:
        median = _median(scores)
        mad = _mad(scores, median)
        sigma = max(1e-6, 1.4826 * mad)

        high_thr = float(self.high) if self.high is not None else (median + self.high_z * sigma)
        low_thr = float(self.low) if self.low is not None else (median + self.low_z * sigma)
        low_thr = min(low_thr, high_thr)

        spans: list[tuple[int, int]] = []
        in_span = False
        start = 0
        for i, s in enumerate(scores):
            if not in_span and s >= high_thr:
                in_span = True
                start = i
            elif in_span and s < low_thr:
                spans.append((start, i - 1))
                in_span = False
        if in_span:
            spans.append((start, len(scores) - 1))
        
        expanded_spans: []
        for i, (s, e) in enumerate(spans):
            s = max(0, s - self.peak_expand[0] // 2)
            e = min(len(scores) - 1, e + self.peak_expand[1] // 2)
            expanded_spans.append((s, e))
            
        return expanded_spans

    def _refine_spans(self, spans: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
        if not spans:
            return []

        # Normalize and clamp span boundaries.
        norm: list[tuple[int, int]] = []
        for s, e in spans:
            s = max(0, min(int(s), n - 1))
            e = max(0, min(int(e), n - 1))
            if e < s:
                s, e = e, s
            norm.append((s, e))

        merged = _merge_spans(norm)

        # Merge nearby segments for stable proposal boundaries.
        if self.merge_gap > 0 and merged:
            compact: list[tuple[int, int]] = [merged[0]]
            for s, e in merged[1:]:
                ps, pe = compact[-1]
                if s - pe <= self.merge_gap:
                    compact[-1] = (ps, max(pe, e))
                else:
                    compact.append((s, e))
            merged = compact

        # Drop tiny spans.
        return [se for se in merged if (se[1] - se[0] + 1) >= self.min_span_len]
        