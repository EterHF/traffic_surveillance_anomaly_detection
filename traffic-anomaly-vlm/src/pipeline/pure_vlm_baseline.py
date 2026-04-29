from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from src.core.video_io import ImageSequenceReader, build_reader
from src.vlm.infer import run_inference
from src.vlm.model_loader import LocalVLM
from src.vlm.parser import parse_stage1_output, parse_stage2_output
from src.vlm.prompts import build_pure_stage1_prompt, build_pure_stage2_prompt


def uniform_sample_indices(length: int, n: int) -> list[int]:
    if length <= 0:
        return []
    if n <= 1:
        return [0]
    if length == 1:
        return [0] * n
    values = np.linspace(0, length - 1, num=n)
    return [int(round(v)) for v in values]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    window = max(1, int(window))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def extract_intervals(mask: np.ndarray, min_len: int = 1, merge_gap: int = 0) -> list[list[int]]:
    intervals: list[list[int]] = []
    start = None
    for i, is_on in enumerate(mask.astype(bool)):
        if is_on and start is None:
            start = i
        if not is_on and start is not None:
            intervals.append([start, i - 1])
            start = None
    if start is not None:
        intervals.append([start, len(mask) - 1])

    if not intervals:
        return []

    merged = [intervals[0]]
    for s, e in intervals[1:]:
        prev = merged[-1]
        if s - prev[1] - 1 <= merge_gap:
            prev[1] = e
        else:
            merged.append([s, e])

    if min_len <= 1:
        return merged
    return [[s, e] for s, e in merged if (e - s + 1) >= min_len]


def interval_indices_to_frame_ids(intervals: list[list[int]], frame_ids: list[int]) -> list[list[int]]:
    if not frame_ids:
        return []
    out: list[list[int]] = []
    last_idx = len(frame_ids) - 1
    for start, end in intervals:
        start_idx = max(0, min(int(start), last_idx))
        end_idx = max(0, min(int(end), last_idx))
        out.append([int(frame_ids[start_idx]), int(frame_ids[end_idx])])
    return out


class PureVLMBaselinePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.working_fps = int(getattr(getattr(cfg, "video", object()), "fps_sample", 25) or 25)
        self.window_size = 80
        self.window_stride = 20
        self.chunks_per_window = 4
        self.frames_per_chunk = self.window_size // self.chunks_per_window
        self.sampled_frames_per_chunk = 4
        self.max_image_size = 896
        self.max_stride_ratio = 0.5

        self.smoothing_mode = "savgol"
        self.savgol_window = 11
        self.savgol_polyorder = 2
        self.ma_window = 9
        self.threshold = 0.5
        self.min_interval_len = 10
        self.merge_gap = 8

        baseline_cfg = getattr(cfg, "pure_vlm_baseline", None)
        if baseline_cfg is not None:
            self.window_size = int(getattr(baseline_cfg, "window_size", self.window_size))
            self.window_stride = int(getattr(baseline_cfg, "window_stride", self.window_stride))
            self.chunks_per_window = int(getattr(baseline_cfg, "chunks_per_window", self.chunks_per_window))
            self.sampled_frames_per_chunk = int(
                getattr(baseline_cfg, "sampled_frames_per_chunk", self.sampled_frames_per_chunk)
            )
            self.frames_per_chunk = self.window_size // max(1, self.chunks_per_window)
            self.smoothing_mode = str(getattr(baseline_cfg, "smoothing_mode", self.smoothing_mode))
            self.savgol_window = int(getattr(baseline_cfg, "savgol_window", self.savgol_window))
            self.savgol_polyorder = int(getattr(baseline_cfg, "savgol_polyorder", self.savgol_polyorder))
            self.ma_window = int(getattr(baseline_cfg, "ma_window", self.ma_window))
            self.threshold = float(getattr(baseline_cfg, "threshold", self.threshold))
            self.min_interval_len = int(getattr(baseline_cfg, "min_interval_len", self.min_interval_len))
            self.merge_gap = int(getattr(baseline_cfg, "merge_gap", self.merge_gap))
            self.max_image_size = int(getattr(baseline_cfg, "max_image_size", self.max_image_size))
            self.max_stride_ratio = float(getattr(baseline_cfg, "max_stride_ratio", self.max_stride_ratio))

        self.window_size = max(4, int(self.window_size))
        self.window_stride_requested = max(1, int(self.window_stride))
        max_stride = max(1, int(round(self.window_size * max(0.05, self.max_stride_ratio))))
        self.window_stride = min(self.window_stride_requested, max_stride)
        self.window_stride_clamped = self.window_stride != self.window_stride_requested

        vlm_cfg = getattr(cfg, "vlm", None)
        if vlm_cfg is None:
            raise ValueError("Missing cfg.vlm section for pure baseline")
        self.vlm = LocalVLM(
            model_path=vlm_cfg.model_path,
            device=getattr(vlm_cfg, "device", "cuda"),
            dtype=getattr(vlm_cfg, "dtype", "auto"),
        )
        self.max_new_tokens = int(getattr(vlm_cfg, "max_new_tokens", 256))

    def _collect_frame_items(self, frame_source: str) -> list[tuple[int, str]]:
        reader = build_reader(frame_source, input_fps=float(self.working_fps))
        if not isinstance(reader, ImageSequenceReader):
            raise ValueError(
                "Pure VLM baseline expects frame-image folder/list input. "
                "Please provide a directory of frames or an image list txt/lst file."
            )

        frame_items = [(int(fid), str(path)) for fid, path in reader.frame_items]
        if not frame_items:
            raise ValueError(f"No frames found from source: {frame_source}")
        return frame_items

    def _smooth_scores(self, frame_scores: np.ndarray) -> np.ndarray:
        if frame_scores.size == 0:
            return frame_scores

        if self.smoothing_mode == "savgol":
            window = max(3, int(self.savgol_window))
            if window % 2 == 0:
                window += 1
            if window > frame_scores.size:
                window = frame_scores.size if frame_scores.size % 2 == 1 else frame_scores.size - 1
            if window >= 3 and window > self.savgol_polyorder:
                smooth = savgol_filter(frame_scores, window_length=window, polyorder=self.savgol_polyorder)
            else:
                smooth = moving_average(frame_scores, window=self.ma_window)
        else:
            smooth = moving_average(frame_scores, window=self.ma_window)

        return np.clip(smooth.astype(np.float32), 0.0, 1.0)

    def _compute_video_topk(self, scores: np.ndarray, ratio: float = 0.1) -> float:
        if scores.size == 0:
            return 0.0
        k = max(1, int(math.ceil(scores.size * ratio)))
        topk = np.sort(scores)[-k:]
        return float(np.mean(topk))

    def run_video(self, video_id: str, frame_source: str) -> dict[str, Any]:
        video_t0 = time.perf_counter()
        frame_items = self._collect_frame_items(frame_source)
        frame_ids = [fid for fid, _ in frame_items]
        frame_paths = [path for _, path in frame_items]
        total_frames = len(frame_items)

        if total_frames < self.window_size:
            raise ValueError(
                f"Not enough frames for windowing: got={total_frames}, required={self.window_size}"
            )

        window_results: list[dict[str, Any]] = []
        frame_scores_raw = np.zeros(total_frames, dtype=np.float32)
        frame_covered = np.zeros(total_frames, dtype=bool)

        stage1_secs: list[float] = []
        stage2_secs: list[float] = []
        window_secs: list[float] = []

        window_id = 0
        for start in range(0, total_frames - self.window_size + 1, self.window_stride):
            end = start + self.window_size
            window_frame_ids = frame_ids[start:end]
            window_paths = frame_paths[start:end]

            sampled_paths: list[str] = []
            for chunk_idx in range(self.chunks_per_window):
                c_start = chunk_idx * self.frames_per_chunk
                c_end = c_start + self.frames_per_chunk
                chunk_paths = window_paths[c_start:c_end]
                for idx in uniform_sample_indices(len(chunk_paths), self.sampled_frames_per_chunk):
                    sampled_paths.append(chunk_paths[idx])

            stage1_prompt = build_pure_stage1_prompt(
                chunks_per_window=self.chunks_per_window,
                sampled_frames_per_chunk=self.sampled_frames_per_chunk,
            )
            t_w0 = time.perf_counter()
            stage1_raw = run_inference(
                self.vlm,
                stage1_prompt,
                sampled_paths,
                self.max_new_tokens,
                max_image_size=self.max_image_size,
            )
            t_w1 = time.perf_counter()
            stage1_output = parse_stage1_output(stage1_raw)

            stage2_prompt = build_pure_stage2_prompt(stage1_output)
            stage2_raw = run_inference(
                self.vlm,
                stage2_prompt,
                sampled_paths,
                self.max_new_tokens,
                max_image_size=self.max_image_size,
            )
            t_w2 = time.perf_counter()
            stage2_output = parse_stage2_output(stage2_raw, chunks=self.chunks_per_window)

            s1 = float(t_w1 - t_w0)
            s2 = float(t_w2 - t_w1)
            wt = float(t_w2 - t_w0)
            stage1_secs.append(s1)
            stage2_secs.append(s2)
            window_secs.append(wt)

            for chunk_idx, score in enumerate(stage2_output["chunk_scores"]):
                c_start = start + chunk_idx * self.frames_per_chunk
                c_end = min(start + (chunk_idx + 1) * self.frames_per_chunk, total_frames)
                frame_scores_raw[c_start:c_end] = np.maximum(frame_scores_raw[c_start:c_end], float(score))

            frame_covered[start:end] = True

            window_results.append(
                {
                    "window_id": window_id,
                    "start_frame": int(window_frame_ids[0]),
                    "end_frame": int(window_frame_ids[-1]),
                    "stage1_raw_text": stage1_raw,
                    "stage2_raw_text": stage2_raw,
                    "timing": {
                        "stage1_sec": s1,
                        "stage2_sec": s2,
                        "window_total_sec": wt,
                    },
                    "stage1_output": stage1_output,
                    "stage2_output": stage2_output,
                }
            )
            window_id += 1

        frame_scores_smooth = self._smooth_scores(frame_scores_raw)
        pred_mask = frame_scores_smooth > float(self.threshold)
        interval_indices = extract_intervals(
            pred_mask,
            min_len=max(1, int(self.min_interval_len)),
            merge_gap=max(0, int(self.merge_gap)),
        )
        intervals = interval_indices_to_frame_ids(interval_indices, frame_ids)

        video_total_sec = float(time.perf_counter() - video_t0)
        num_windows = len(window_results)
        covered_frames = int(frame_covered.sum())

        return {
            "video_id": video_id,
            "fps": int(self.working_fps),
            "window_size": int(self.window_size),
            "window_stride_requested": int(self.window_stride_requested),
            "window_stride": int(self.window_stride),
            "window_stride_clamped": bool(self.window_stride_clamped),
            "chunks_per_window": int(self.chunks_per_window),
            "frames_per_chunk": int(self.frames_per_chunk),
            "sampled_frames_per_chunk": int(self.sampled_frames_per_chunk),
            "max_image_size": int(self.max_image_size),
            "num_frames": int(total_frames),
            "frame_ids": frame_ids,
            "window_results": window_results,
            "frame_scores_raw": frame_scores_raw.tolist(),
            "frame_scores_smooth": frame_scores_smooth.tolist(),
            "predicted_interval_indices": interval_indices,
            "predicted_intervals": intervals,
            "coverage": {
                "covered_frames": covered_frames,
                "coverage_ratio": float(covered_frames / max(1, total_frames)),
            },
            "timing": {
                "video_total_sec": video_total_sec,
                "num_windows": num_windows,
                "avg_window_sec": float(np.mean(window_secs)) if window_secs else 0.0,
                "avg_stage1_sec": float(np.mean(stage1_secs)) if stage1_secs else 0.0,
                "avg_stage2_sec": float(np.mean(stage2_secs)) if stage2_secs else 0.0,
            },
            "video_score_max": float(np.max(frame_scores_smooth)) if frame_scores_smooth.size else 0.0,
            "video_score_top10": self._compute_video_topk(frame_scores_smooth, ratio=0.1),
        }
