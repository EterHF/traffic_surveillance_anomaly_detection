# Convert all sub videos in input root to frame folders in output root.

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}


def _list_videos(root: Path) -> list[Path]:
    videos: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    return videos


def convert_one_video(
    video_path: Path,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
    target_fps: float | None,
    image_ext: str,
    jpeg_quality: int,
) -> tuple[bool, str]:
    rel = video_path.relative_to(input_root)
    out_dir = output_root / rel.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob(f"*.{image_ext}"))
    if existing and not overwrite:
        return True, f"skip existing: {out_dir}"

    # Clear old extracted frames when overwrite is enabled.
    if overwrite and existing:
        for f in existing:
            f.unlink(missing_ok=True)

    frame_pattern = out_dir / f"%06d.{image_ext}"

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-start_number",
        "0",
    ]

    if target_fps is not None:
        cmd.extend(["-vf", f"fps={target_fps}"])

    if image_ext == "jpg":
        # Better visual quality for JPEG extraction.
        cmd.extend(["-q:v", str(jpeg_quality)])

    cmd.append(str(frame_pattern))

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return False, f"ffmpeg failed: {video_path} -> {out_dir}\n{proc.stderr.strip()}"

    out_count = len(list(out_dir.glob(f"*.{image_ext}")))
    if out_count == 0:
        return False, f"no frames produced: {video_path} -> {out_dir}"

    return True, f"ok: {video_path.name} -> {out_dir} ({out_count} frames)"


def _convert_one_video_job(job: tuple[str, str, str, bool, float | None, str, int]) -> tuple[bool, str]:
    video_path, input_root, output_root, overwrite, target_fps, image_ext, jpeg_quality = job
    return convert_one_video(
        video_path=Path(video_path),
        input_root=Path(input_root),
        output_root=Path(output_root),
        overwrite=overwrite,
        target_fps=target_fps,
        image_ext=image_ext,
        jpeg_quality=jpeg_quality,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert videos to frame folders.")
    parser.add_argument(
        "--input-root",
        default="/nvme2/VAD_yemao/data/TU-DAT/videos_30fps/positive",
        help="Root directory containing videos.",
    )
    parser.add_argument(
        "--output-root",
        default="/nvme2/VAD_yemao/data/TU-DAT/frames/positive",
        help="Output root directory for extracted frames.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional target extraction FPS (default: keep source FPS).",
    )
    parser.add_argument(
        "--image-ext",
        default="jpg",
        choices=["jpg", "png", "bmp", "webp"],
        help="Output frame image format.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=2,
        help="JPEG quality for ffmpeg -q:v (lower is better, valid 2~31).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frame folders.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel ffmpeg workers.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found or not a directory: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    videos = _list_videos(input_root)

    print(f"found videos: {len(videos)}")
    print(f"input_root={input_root}")
    print(f"output_root={output_root}")
    print(f"fps={args.fps if args.fps is not None else 'source'}")
    print(f"image_ext={args.image_ext}")
    print(f"workers={args.workers}")

    ok = 0
    failed = 0
    failures: list[str] = []

    jobs = [
        (
            str(v),
            str(input_root),
            str(output_root),
            bool(args.overwrite),
            float(args.fps) if args.fps is not None else None,
            str(args.image_ext),
            int(args.jpeg_quality),
        )
        for v in videos
    ]

    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        future_map = {ex.submit(_convert_one_video_job, j): j[0] for j in jobs}
        done_count = 0
        for fut in as_completed(future_map):
            done_count += 1
            success, msg = fut.result()
            prefix = f"[{done_count}/{len(videos)}]"
            if success:
                ok += 1
                print(prefix, msg)
            else:
                failed += 1
                failures.append(msg)
                print(prefix, "FAIL:", msg)

    print("\n=== summary ===")
    print(f"total={len(videos)} ok={ok} failed={failed}")
    if failures:
        log_path = output_root / "conversion_failures.log"
        log_path.write_text("\n\n".join(failures), encoding="utf-8")
        print(f"failure log: {log_path}")


if __name__ == "__main__":
    main()
