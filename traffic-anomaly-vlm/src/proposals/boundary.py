from __future__ import annotations


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
        min_rise: float | None = None,
        high_z: float = 2.5,
        low_z: float = 0.5,
        rise_z: float = 2.0,
        fallback_span: int = 24,
    ):
        self.high = high
        self.low = low
        self.min_rise = min_rise
        self.high_z = high_z
        self.low_z = low_z
        self.rise_z = rise_z
        self.fallback_span = max(4, int(fallback_span))

    def detect(self, scores: list[float]) -> list[tuple[int, int]]:
        if not scores:
            return []

        stat = self._compute_thresholds(scores)
        high_thr = stat["high"]
        rise_thr = stat["rise"]
        low_thr = stat["low"]

        peak_idx = self._find_steep_peaks(scores, high_thr=high_thr, rise_thr=rise_thr)
        if not peak_idx:
            peak_idx = self._fallback_candidates(scores)

        spans = [self._expand_peak(scores, i, low_thr=low_thr) for i in peak_idx]
        return _merge_spans(spans)

    def _compute_thresholds(self, scores: list[float]) -> dict[str, float]:
        med = _median(scores)
        sigma = max(1e-6, 1.4826 * _mad(scores, med))

        diffs = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
        d_med = _median(diffs)
        d_sigma = max(1e-6, 1.4826 * _mad(diffs, d_med))

        adaptive_high = med + float(self.high_z) * sigma
        adaptive_low = med + float(self.low_z) * sigma
        adaptive_rise = d_med + float(self.rise_z) * d_sigma

        high_thr = float(self.high) if self.high is not None else adaptive_high
        low_thr = float(self.low) if self.low is not None else adaptive_low
        rise_thr = float(self.min_rise) if self.min_rise is not None else adaptive_rise

        return {
            "high": high_thr,
            "low": min(low_thr, high_thr),
            "rise": rise_thr,
        }

    def _find_steep_peaks(self, scores: list[float], high_thr: float, rise_thr: float) -> list[int]:
        n = len(scores)
        if n == 1:
            return [0] if scores[0] >= high_thr else []

        peaks: list[int] = []
        for i in range(n):
            left = scores[i - 1] if i > 0 else float("-inf")
            right = scores[i + 1] if i + 1 < n else float("-inf")
            local_peak = scores[i] >= left and scores[i] >= right
            if not local_peak:
                continue

            prev = scores[i - 1] if i > 0 else scores[i]
            rise = scores[i] - prev
            if scores[i] >= high_thr and rise >= rise_thr:
                peaks.append(i)
        return peaks

    def _fallback_candidates(self, scores: list[float]) -> list[int]:
        n = len(scores)
        if n == 1:
            return [0]

        k = max(1, n // self.fallback_span)
        if n >= self.fallback_span:
            k += 1

        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        chosen: list[int] = []
        min_gap = max(1, n // max(2, 2 * k))
        for idx in order:
            if all(abs(idx - j) >= min_gap for j in chosen):
                chosen.append(idx)
            if len(chosen) >= k:
                break

        return sorted(chosen)

    def _expand_peak(self, scores: list[float], idx: int, low_thr: float) -> tuple[int, int]:
        n = len(scores)
        s = idx
        e = idx
        low = float(low_thr)

        while s > 0 and scores[s - 1] >= low:
            s -= 1
        while e + 1 < n and scores[e + 1] >= low:
            e += 1

        if s == idx and e == idx:
            radius = max(1, n // 40)
            s = max(0, idx - radius)
            e = min(n - 1, idx + radius)
        return (s, e)
