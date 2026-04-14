from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _natural_key(p: Path) -> tuple:
    """Sort paths naturally so 2.jpg comes before 10.jpg."""

    s = p.stem
    parts = re.split(r"(\d+)", s)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    key.append((2, p.suffix.lower()))
    return tuple(key)


def _is_readable_image(path: Path) -> bool:
    """Fast sanity check via ffprobe to skip broken image files."""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode == 0 and bool(res.stdout.strip())


def _list_sequence_dirs(root: Path) -> list[Path]:
    """Find all directories that directly contain image files."""

    dirs: list[Path] = []
    for d in sorted((p for p in root.rglob("*") if p.is_dir())):
        has_images = any((f.is_file() and f.suffix.lower() in IMAGE_EXTS) for f in d.iterdir())
        if has_images:
            dirs.append(d)
    return dirs


def _write_concat_list(frames: Iterable[Path], fps: float) -> Path:
    """Create ffmpeg concat list with fixed per-frame duration for stable CFR output."""

    frame_list = list(frames)
    duration = 1.0 / fps

    with tempfile.NamedTemporaryFile("w", suffix=".txt", prefix="ffconcat_", delete=False) as f:
        for img in frame_list:
            # Keep absolute path + quote escaping for robustness.
            escaped = str(img.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
            f.write(f"duration {duration:.10f}\n")

        # Concat demuxer needs the final file repeated to apply the last duration.
        last_escaped = str(frame_list[-1].resolve()).replace("'", "'\\''")
        f.write(f"file '{last_escaped}'\n")

        return Path(f.name)


def _run_ffmpeg_concat(frames: list[Path], out_path: Path, fps: float, overwrite: bool) -> tuple[bool, str]:
    """Fast stable path: use an explicit ordered concat list (no per-frame probe)."""

    list_file = _write_concat_list(frames, fps)
    try:
        cmd = [
            "ffmpeg",
            "-y" if overwrite else "-n",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-vsync",
            "cfr",
            "-r",
            str(fps),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            return False, proc.stderr.strip()
        return True, ""
    finally:
        list_file.unlink(missing_ok=True)


def _run_ffmpeg_glob(seq_dir: Path, out_path: Path, fps: float, overwrite: bool) -> tuple[bool, str]:
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        str(seq_dir / "*.jpg"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return False, proc.stderr.strip()
    return True, ""


def _run_ffmpeg_concat_with_probe(frames: list[Path], out_path: Path, fps: float, overwrite: bool) -> tuple[bool, str]:
    """Slow but robust fallback: probe each frame and build concat list for CFR output."""

    valid_frames = [f for f in frames if _is_readable_image(f)]
    if not valid_frames:
        return False, "all frames unreadable"

    list_file = _write_concat_list(valid_frames, fps)
    try:
        cmd = [
            "ffmpeg",
            "-y" if overwrite else "-n",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-vsync",
            "cfr",
            "-r",
            str(fps),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            return False, proc.stderr.strip()
        return True, ""
    finally:
        list_file.unlink(missing_ok=True)


def convert_one_dir(
    seq_dir: Path,
    input_root: Path,
    output_root: Path,
    fps: float,
    overwrite: bool,
    fallback_repair: bool,
) -> tuple[bool, str]:
    rel = seq_dir.relative_to(input_root)
    out_path = (output_root / rel).with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return True, f"skip existing: {out_path}"

    frames = sorted(
        [f for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS],
        key=_natural_key,
    )
    if not frames:
        return False, f"no image frames: {seq_dir}"

    # Primary path uses explicit natural ordering to avoid occasional frame-order flicker.
    ok, err = _run_ffmpeg_concat(frames=frames, out_path=out_path, fps=fps, overwrite=overwrite)
    if ok:
        return True, f"ok: {out_path}"

    if not fallback_repair:
        return False, f"ffmpeg failed: {seq_dir} -> {out_path}\n{err}"

    ok2, err2 = _run_ffmpeg_concat_with_probe(frames=frames, out_path=out_path, fps=fps, overwrite=overwrite)
    if ok2:
        return True, f"ok(repaired): {out_path}"

    return False, f"ffmpeg failed after repair: {seq_dir} -> {out_path}\nfast_err: {err}\nrepair_err: {err2}"


def _convert_one_dir_job(job: tuple[str, str, str, float, bool, bool]) -> tuple[bool, str]:
    seq_dir, input_root, output_root, fps, overwrite, fallback_repair = job
    return convert_one_dir(
        seq_dir=Path(seq_dir),
        input_root=Path(input_root),
        output_root=Path(output_root),
        fps=fps,
        overwrite=overwrite,
        fallback_repair=fallback_repair,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert frame directories to MP4 videos.")
    parser.add_argument(
        "--input-root",
        default="/nvme2/VAD_yemao/data/TAD/frames",
        help="Root directory containing frame sequence folders.",
    )
    parser.add_argument(
        "--output-root",
        default="/nvme2/VAD_yemao/data/TAD/videos_25fps",
        help="Output root directory for generated videos.",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="Target output FPS.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output videos.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel ffmpeg workers. Increase for faster reruns (watch CPU/IO pressure).",
    )
    parser.add_argument(
        "--fallback-repair",
        action="store_true",
        help="If fast conversion fails, retry with per-frame probing + concat (slower but robust).",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found or not a directory: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    seq_dirs = _list_sequence_dirs(input_root)

    print(f"found sequence dirs: {len(seq_dirs)}")
    print(f"input_root={input_root}")
    print(f"output_root={output_root}")
    print(f"fps={args.fps}")
    print(f"workers={args.workers}")

    ok = 0
    failed = 0
    failures: list[str] = []

    jobs = [
        (str(d), str(input_root), str(output_root), float(args.fps), bool(args.overwrite), bool(args.fallback_repair))
        for d in seq_dirs
    ]

    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        future_map = {ex.submit(_convert_one_dir_job, j): j[0] for j in jobs}
        done_count = 0
        for fut in as_completed(future_map):
            done_count += 1
            success, msg = fut.result()
            prefix = f"[{done_count}/{len(seq_dirs)}]"
            if success:
                ok += 1
                print(prefix, msg)
            else:
                failed += 1
                failures.append(msg)
                print(prefix, "FAIL:", msg)

    print("\n=== summary ===")
    print(f"total={len(seq_dirs)} ok={ok} failed={failed}")
    if failures:
        log_path = output_root / "conversion_failures.log"
        log_path.write_text("\n\n".join(failures), encoding="utf-8")
        print(f"failure log: {log_path}")


if __name__ == "__main__":
    main()
