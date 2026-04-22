from __future__ import annotations

import numpy as np

from src.proposals.boundary_detector import BoundaryDetector, BoundaryDetectorConfig

from .config import CoarseSpanConfig
from .helpers import peak_idx_for_span, top_mean
from .models import EventNode


def _make_coarse_span(
    span_id: str,
    start_idx: int,
    end_idx: int,
    frame_ids: list[int],
    seed_scores: np.ndarray,
) -> EventNode:
    peak_idx = peak_idx_for_span(seed_scores, start_idx, end_idx)
    seed_seg = seed_scores[start_idx : end_idx + 1]
    seed_peak = float(np.max(seed_seg)) if seed_seg.size else 0.0
    seed_mean = float(top_mean(seed_seg, ratio=0.25)) if seed_seg.size else 0.0
    return EventNode(
        node_id=span_id,
        level=1,
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        peak_idx=int(peak_idx),
        start_frame=int(frame_ids[start_idx]),
        end_frame=int(frame_ids[end_idx]),
        peak_frame=int(frame_ids[peak_idx]),
        eventness_peak=seed_peak,
        eventness_mean=seed_mean,
        span_prior_score=seed_peak,
    )


def build_coarse_spans(
    frame_ids: list[int],
    seed_scores: list[float] | np.ndarray,
    track_scores: list[float] | np.ndarray,
    object_scores: list[float] | np.ndarray,
    cfg: CoarseSpanConfig | None = None,
) -> list[EventNode]:
    """Build coarse spans directly from the raw anomaly score curve."""

    cfg = cfg or CoarseSpanConfig()
    seed_arr = np.asarray(seed_scores, dtype=np.float32)
    del track_scores, object_scores
    detector = BoundaryDetector(
        cfg=BoundaryDetectorConfig(
            use_savgol_filter=False,
            method="by_peeks",
            high_z=float(cfg.high_z),
            peak_gap=int(cfg.peak_gap),
            peak_expand=tuple(int(v) for v in cfg.peak_expand),
            min_span_len=int(cfg.min_span_len),
            merge_gap=int(cfg.merge_gap),
        )
    )
    spans = detector.detect(seed_arr.astype(np.float32).tolist())
    if not isinstance(spans, list):
        return []

    coarse_spans = [
        _make_coarse_span(
            span_id=f"coarse.{idx}",
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            frame_ids=frame_ids,
            seed_scores=seed_arr,
        )
        for idx, (start_idx, end_idx) in enumerate(spans, start=1)
        if end_idx >= start_idx
    ]
    coarse_spans = sorted(
        coarse_spans,
        key=lambda span: (float(span.span_prior_score), float(span.eventness_peak), -int(span.span_len())),
        reverse=True,
    )
    if int(cfg.max_proposals) > 0:
        coarse_spans = coarse_spans[: int(cfg.max_proposals)]
    return coarse_spans
