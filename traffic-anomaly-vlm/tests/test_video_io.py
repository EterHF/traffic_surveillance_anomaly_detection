from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.core.video_io import build_reader


def test_image_sequence_reader_from_dir(tmp_path: Path):
    for i in range(3):
        img = np.full((32, 32, 3), i * 40, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{i:03d}.jpg"), img)

    reader = build_reader(str(tmp_path), input_fps=25.0)
    frames = list(iter(reader))

    assert len(frames) == 3
    assert frames[0][0] == 0
    assert reader.fps() == 25.0
