from __future__ import annotations

import numpy as np

from src.proposals.event_tree import EventTreeConfig, build_event_tree, flatten_event_tree
from src.proposals.node_selector import NodeSelectorConfig, select_salient_nodes


def test_event_tree_builds_nodes_from_peaks():
    frame_ids = list(range(40))
    scores = np.zeros((40,), dtype=np.float32)
    scores[8:12] = [0.1, 0.6, 0.9, 0.5]
    scores[24:29] = [0.2, 0.7, 1.0, 0.8, 0.3]

    root = build_event_tree(
        frame_ids,
        scores,
        EventTreeConfig(
            max_depth=2,
            min_span_len=4,
            split_min_len=8,
            peak_gap=3,
            peak_expand=(2, 3),
            merge_gap=2,
            per_level_high_z=(0.6, 0.4),
        ),
    )
    nodes = flatten_event_tree(root)

    assert len(nodes) >= 2
    assert any(7 <= n.peak_frame <= 12 for n in nodes)
    assert any(24 <= n.peak_frame <= 28 for n in nodes)


def test_node_selector_prefers_event_risk_overlap():
    frame_ids = list(range(30))
    scores = np.zeros((30,), dtype=np.float32)
    scores[10:15] = [0.1, 0.6, 1.0, 0.7, 0.2]
    scores[20:24] = [0.1, 0.7, 0.8, 0.3]
    track_risk = np.zeros((30,), dtype=np.float32)
    track_risk[11:14] = [0.4, 0.9, 0.5]

    root = build_event_tree(
        frame_ids,
        scores,
        EventTreeConfig(
            max_depth=2,
            min_span_len=3,
            split_min_len=6,
            peak_gap=2,
            peak_expand=(1, 2),
            merge_gap=1,
            per_level_high_z=(0.4, 0.2),
        ),
    )
    selected = select_salient_nodes(
        root,
        eventness_scores=scores,
        track_risk_scores=track_risk,
        cfg=NodeSelectorConfig(top_k=1, min_node_len=3, overlap_iou=0.9, gate_floor=0.35),
    )

    assert len(selected) == 1
    assert 10 <= selected[0].peak_frame <= 14
