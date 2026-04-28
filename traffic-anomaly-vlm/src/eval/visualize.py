from pathlib import Path
import math
import cv2

from matplotlib import pyplot as plt
import numpy as np
from src.schemas import EventNode


def plot_peaks(
    scores: list[float] | np.ndarray,
    frame_ids: list[int],
    peaks: list[int],
    title: str = "Scores_over_Time",
    output_dir: str = "outputs",
) -> plt.Figure:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dense_scores = np.asarray(scores, dtype=np.float32)
    peak_frames = [frame_ids[idx] for idx in peaks]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame_ids, dense_scores, linewidth=1.3, label="scores", color="blue")
    ax.scatter(peak_frames, dense_scores[peaks], s=60, color="red", label="peaks", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path / f"{title.replace(' ', '_').lower()}.png", dpi=160)
    return fig


def plot_scores(
    scores: list[float] | np.ndarray,
    frame_ids: list[int],
    pred_spans: list[tuple[int, int]] | None = None,
    gt_intervals: list[list[int]] | None = None,
    title: str = "Scores over Time",
    output_dir: str = "outputs",
) -> plt.Figure:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dense_trigger = np.asarray(scores, dtype=np.float32)
    pred_spans = pred_spans or []
    gt_intervals = gt_intervals or []

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame_ids, dense_trigger, linewidth=1.3, label="trigger_score", color="blue")
    for start, end in gt_intervals:
        ax.axvspan(start, end, color="red", alpha=0.10)
    for start, end in pred_spans:
        ax.axvspan(start, end, color="#ffb74d", alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path / f"{title.replace(' ', '_').lower()}.png", dpi=160)
    return fig


def plot_event_nodes(nodes: list[EventNode], frame_ids: list[int], title: str = "Event_Nodes", output_dir: str = "outputs") -> plt.Figure:
    def _collect_nodes(input_nodes: list[EventNode]) -> dict[str, EventNode]:
        by_id: dict[str, EventNode] = {}

        def _walk(node: EventNode) -> None:
            if node.node_id in by_id:
                return
            by_id[node.node_id] = node
            for child in node.children:
                _walk(child)

        for node in input_nodes:
            _walk(node)
        return by_id

    node_by_id = _collect_nodes(nodes)
    all_nodes = sorted(
        node_by_id.values(),
        key=lambda node: (int(node.level), int(node.start_frame), int(node.end_frame), str(node.node_id)),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not all_nodes:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No event nodes", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.suptitle(title)
        plt.tight_layout()
        fig.savefig(output_path / f"{title.replace(' ', '_').lower()}.png")
        return fig

    max_level = max(int(node.level) for node in all_nodes)
    min_frame = min(frame_ids) if frame_ids else min(int(node.start_frame) for node in all_nodes)
    max_frame = max(frame_ids) if frame_ids else max(int(node.end_frame) for node in all_nodes)
    level_ticks = list(range(1, max_level + 1))
    cmap = plt.cm.get_cmap("tab10", max(2, max_level + 1))

    fig_height = max(4.0, 1.25 * max_level + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    edges: set[tuple[str, str]] = set()
    for node in all_nodes:
        for child in node.children:
            if child.node_id in node_by_id:
                edges.add((node.node_id, child.node_id))

        if "." in node.node_id:
            parent_id = node.node_id.rsplit(".", 1)[0]
            if parent_id in node_by_id:
                edges.add((parent_id, node.node_id))

    for parent_id, child_id in sorted(edges):
        parent = node_by_id[parent_id]
        child = node_by_id[child_id]
        ax.plot(
            [float(parent.peak_frame), float(child.peak_frame)],
            [float(parent.level), float(child.level)],
            color="0.65",
            linewidth=1.2,
            zorder=1,
        )

    for node in all_nodes:
        y = float(node.level)
        color = cmap(int(node.level) % cmap.N)
        ax.hlines(
            y=y,
            xmin=float(node.start_frame),
            xmax=float(node.end_frame),
            color=color,
            linewidth=8,
            alpha=0.95,
            zorder=2,
        )
        ax.scatter(
            float(node.peak_frame),
            y,
            s=46,
            color="black",
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

        label = str(node.node_id)
        if float(node.fused_score) > 0.0:
            label = f"{label} ({float(node.fused_score):.2f})"
        ax.text(
            float(node.start_frame),
            y - 0.12,
            label,
            fontsize=9,
            ha="left",
            va="top",
            color="black",
            zorder=4,
        )

    ax.set_title(title)
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Tree Level")
    ax.set_xlim(float(min_frame), float(max_frame))
    ax.set_ylim(max_level + 0.7, 0.3)
    ax.set_yticks(level_ticks)
    ax.set_yticklabels([f"L{level}" for level in level_ticks])
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.grid(True, axis="y", linestyle=":", alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path / f"{title.replace(' ', '_').lower()}.png")
    return fig


def plot_all_subscores(
    frame_ids: list[int],
    subscores: dict[str, list[float] | np.ndarray],
    gt_intervals: list[list[int]] | None = None,
    title: str = "All Subscores",
    output_dir: str = "outputs",
    filename: str | None = None,
    max_cols: int = 2,
) -> plt.Figure:
    """Plot each raw subscore in its own subplot over time."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    gt_intervals = gt_intervals or []

    valid_items: list[tuple[str, np.ndarray]] = []
    for name, values in subscores.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            continue
        valid_items.append((str(name), arr))

    if not valid_items:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No subscores", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.suptitle(title)
        fig.tight_layout()
        target_name = filename or f"{title.replace(' ', '_').lower()}.png"
        fig.savefig(output_path / target_name, dpi=160)
        plt.close(fig)
        return fig

    max_cols = max(1, int(max_cols))
    n_items = len(valid_items)
    n_cols = min(max_cols, n_items)
    n_rows = int(math.ceil(n_items / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 2.6 * n_rows), sharex=True)
    axes_arr = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for ax, (name, curve) in zip(axes_arr.ravel(), valid_items):
        ax.plot(frame_ids, curve, linewidth=1.15, color="#1f77b4")
        for start, end in gt_intervals:
            ax.axvspan(start, end, color="red", alpha=0.10)
        ax.set_title(name)
        ax.grid(alpha=0.25)
        ax.set_ylabel("value")

    for ax in axes_arr.ravel()[len(valid_items) :]:
        ax.set_axis_off()

    for ax in axes_arr[-1]:
        ax.set_xlabel("Frame ID")

    fig.suptitle(title)
    fig.tight_layout()
    target_name = filename or f"{title.replace(' ', '_').lower()}.png"
    fig.savefig(output_path / target_name, dpi=160)
    plt.close(fig)
    return fig


def visualize_track_by_id(tracks: list[tuple[float, float]], vis_h: int, vis_w: int, output_path: str = "debug.jpg"):
    canvas = np.full((vis_h, vis_w, 3), 255, dtype=np.uint8)
    prev_x, prev_y = int(tracks[0][0]), int(tracks[0][1])
    # draw the first point as a red circle
    cv2.circle(canvas, (prev_x, prev_y), 6, (0, 0, 255), 2)
    for idx, (cx, cy) in enumerate(tracks):
        cv2.line(canvas, (prev_x, prev_y), (int(cx), int(cy)), (0, 0, 255), 2)
        prev_x, prev_y = int(cx), int(cy)
    # cv2.imwrite(output_path, canvas)
    return canvas
