from __future__ import annotations

from pathlib import Path
import re
from typing import Iterator

import cv2
import numpy as np


class _ResizeMixin:
    def __init__(self, resize_max_side: int = 0, resize_interpolation: str = "area"):
        self.resize_max_side = int(resize_max_side or 0)
        self.resize_interpolation = str(resize_interpolation or "area").lower()

    def _resize_frame_if_needed(self, frame: np.ndarray) -> np.ndarray:
        if self.resize_max_side <= 0:
            return frame

        h, w = frame.shape[:2]
        max_side = max(h, w)
        if max_side <= self.resize_max_side:
            return frame

        scale = float(self.resize_max_side) / float(max_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
        }
        interp = interp_map.get(self.resize_interpolation, cv2.INTER_AREA)
        return cv2.resize(frame, (new_w, new_h), interpolation=interp)


class VideoReader(_ResizeMixin):
    def __init__(self, source: str, resize_max_side: int = 0, resize_interpolation: str = "area"):
        super().__init__(resize_max_side=resize_max_side, resize_interpolation=resize_interpolation)
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open source: {source}")

    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        frame_id = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame_id, self._resize_frame_if_needed(frame)
            frame_id += 1

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()


class ImageSequenceReader(_ResizeMixin):
    def __init__(
        self,
        source: str,
        fps: float = 30.0,
        resize_max_side: int = 0,
        resize_interpolation: str = "area",
        exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ):
        super().__init__(resize_max_side=resize_max_side, resize_interpolation=resize_interpolation)
        self.source = source
        self._fps = float(fps)
        self.exts = tuple(e.lower() for e in exts)
        self.frame_paths = self._resolve_frame_paths(source)
        self.frame_items = self._build_frame_items(self.frame_paths)
        if not self.frame_paths:
            raise ValueError(f"No frame images found from source: {source}")

    def _resolve_frame_paths(self, source: str) -> list[Path]:
        p = Path(source)
        if p.is_dir():
            files = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in self.exts]
            return sorted(files, key=self._path_sort_key)

        if p.is_file() and p.suffix.lower() in {".txt", ".lst"}:
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            out: list[Path] = []
            for ln in lines:
                lp = Path(ln)
                if not lp.is_absolute():
                    lp = p.parent / lp
                if lp.exists() and lp.suffix.lower() in self.exts:
                    out.append(lp)
            return out

        return []

    @staticmethod
    def _parse_numeric_id(stem: str) -> int | None:
        # Prefer the trailing number block, e.g. "img_000123" -> 123.
        m = re.search(r"(\d+)$", stem)
        if not m:
            return None
        return int(m.group(1))

    def _path_sort_key(self, p: Path) -> tuple[int, int | str, str]:
        num = self._parse_numeric_id(p.stem)
        if num is not None:
            return (0, num, p.name)
        return (1, p.stem, p.name)

    def _build_frame_items(self, paths: list[Path]) -> list[tuple[int, Path]]:
        parsed_ids = [self._parse_numeric_id(p.stem) for p in paths]
        if all(v is not None for v in parsed_ids):
            ids = [int(v) for v in parsed_ids]
            if len(set(ids)) == len(ids):
                return list(zip(ids, paths))
        # Fallback to stable sequential ids when filenames are non-numeric or duplicated.
        return list(enumerate(paths))

    def fps(self) -> float:
        return self._fps

    def frame_count(self) -> int:
        return len(self.frame_items)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        for frame_id, fp in self.frame_items:
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            yield frame_id, self._resize_frame_if_needed(frame)

    def release(self) -> None:
        return


def build_reader(
    source: str,
    input_fps: float = 30.0,
    resize_max_side: int = 0,
    resize_interpolation: str = "area",
):
    p = Path(source)
    if p.is_dir() or (p.is_file() and p.suffix.lower() in {".txt", ".lst"}):
        return ImageSequenceReader(
            source=source,
            fps=input_fps,
            resize_max_side=resize_max_side,
            resize_interpolation=resize_interpolation,
        )
    return VideoReader(
        source,
        resize_max_side=resize_max_side,
        resize_interpolation=resize_interpolation,
    )
