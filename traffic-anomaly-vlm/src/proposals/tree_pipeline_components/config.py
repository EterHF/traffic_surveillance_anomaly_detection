from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TreeBuildConfig:
    """Bottom-up merge settings for pure event-tree construction."""

    max_depth: int = 3
    min_span_len: int = 8
    split_min_len: int = 16
    peak_gap: int = 5
    peak_expand: tuple[int, int] = (10, 15)
    merge_gap: int = 10
    per_level_high_z: tuple[float, ...] = (1.0, 0.6, 0.3)
    min_output_span: int = 8

    refine_coarse: bool = True
    prompt_method: str = "single_stage"


@dataclass
class NodeSelectorConfig:
    """Ranking and de-duplication settings for final candidate nodes."""

    top_k: int = 6
    min_node_len: int = 8
    overlap_iou: float = 0.85
    gate_floor: float = 0.55
    reference_span: int = 96
