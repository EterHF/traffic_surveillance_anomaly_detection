from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg", ".m4v"}


@dataclass
class VideoLabel:
    total_frames: int
    intervals: List[Tuple[int, int]]


def parse_existing_labels(label_path: Path) -> Dict[str, VideoLabel]:
    labels: Dict[str, VideoLabel] = {}
    if not label_path.exists():
        return labels

    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue

        parts = text.split()
        if len(parts) < 2:
            print(f"[WARN] Skip malformed line {line_no}: {text}")
            continue

        video_id = parts[0]
        try:
            total_frames = int(parts[1])
        except ValueError:
            print(f"[WARN] Invalid total_frames at line {line_no}: {text}")
            continue

        nums: List[int] = []
        ok = True
        for token in parts[2:]:
            try:
                nums.append(int(token))
            except ValueError:
                print(f"[WARN] Invalid frame id at line {line_no}: {text}")
                ok = False
                break
        if not ok:
            continue

        if len(nums) % 2 != 0:
            print(f"[WARN] Odd number of interval tokens at line {line_no}: {text}")
            continue

        intervals: List[Tuple[int, int]] = []
        for i in range(0, len(nums), 2):
            s, e = nums[i], nums[i + 1]
            if e < s:
                s, e = e, s
            intervals.append((s, e))

        labels[video_id] = VideoLabel(total_frames=total_frames, intervals=intervals)

    return labels


def write_labels(label_path: Path, labels: Dict[str, VideoLabel], ordered_ids: List[str]) -> None:
    lines: List[str] = []
    for vid in ordered_ids:
        if vid not in labels:
            continue
        item = labels[vid]
        row = [vid, str(int(item.total_frames))]
        for s, e in item.intervals:
            row.extend([str(int(s)), str(int(e))])
        lines.append(" ".join(row))

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def collect_videos(video_dir: Path) -> List[Path]:
    files = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(files, key=lambda x: x.name)


def draw_overlay(frame, video_name: str, video_idx: int, video_total: int, frame_idx: int, total_frames: int, intervals: List[Tuple[int, int]], pending_start: int | None, paused: bool):
    out = frame.copy()
    h, w = out.shape[:2]

    lines = [
        f"Video {video_idx}/{video_total}: {video_name}",
        f"Frame: {frame_idx}/{max(total_frames - 1, 0)}  ({frame_idx + 1}/{total_frames})",
        f"Intervals: {len(intervals)}" + (f"  pending_start={pending_start}" if pending_start is not None else ""),
        "Mode: PAUSE" if paused else "Mode: PLAY",
        "Keys: SPACE pause/play | A start | D end | U undo | C clear | ,/. prev/next frame (pause)",
        "      N next video(save) | S save file | Q quit(save)",
    ]

    y = 24
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 255, 30), 2, cv2.LINE_AA)
        y += 24

    # Draw a simple timeline bar at bottom.
    bar_x1, bar_y1 = 10, h - 30
    bar_x2, bar_y2 = w - 10, h - 10
    cv2.rectangle(out, (bar_x1, bar_y1), (bar_x2, bar_y2), (100, 100, 100), 1)

    width = max(bar_x2 - bar_x1, 1)
    if total_frames > 1:
        for s, e in intervals:
            s = max(0, min(s, total_frames - 1))
            e = max(0, min(e, total_frames - 1))
            xs = bar_x1 + int(width * (s / (total_frames - 1)))
            xe = bar_x1 + int(width * (e / (total_frames - 1)))
            cv2.rectangle(out, (xs, bar_y1 + 1), (max(xs + 1, xe), bar_y2 - 1), (0, 0, 255), -1)

        xcur = bar_x1 + int(width * (frame_idx / (total_frames - 1)))
        cv2.line(out, (xcur, bar_y1 - 3), (xcur, bar_y2 + 3), (255, 255, 0), 2)

    return out


class AnnotatorApp:
    def __init__(self, videos: List[Path], labels: Dict[str, VideoLabel], output_txt: Path, ordered_ids: List[str], resume: bool):
        self.videos = videos
        self.labels = labels
        self.output_txt = output_txt
        self.ordered_ids = ordered_ids
        self.resume = resume

        self.root = tk.Tk()
        self.root.title("Local Video Interval Annotator")
        self.root.geometry("1240x820")
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

        self.video_index = -1
        self.current_cap: cv2.VideoCapture | None = None
        self.current_video: Path | None = None
        self.current_total_frames = 0
        self.current_fps = 25.0
        self.frame_index = 0
        self.current_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.intervals: List[Tuple[int, int]] = []
        self.pending_start: int | None = None
        self.playing = True
        self.photo_ref = None
        self.seek_guard = False

        self.video_text = tk.StringVar(value="Video: -")
        self.frame_text = tk.StringVar(value="Frame: -")
        self.mode_text = tk.StringVar(value="Mode: PLAY")
        self.pending_text = tk.StringVar(value="Pending start: None")
        self.interval_text = tk.StringVar(value="Intervals: 0")

        self._build_ui()
        self._bind_shortcuts()
        self._load_next_video(initial=True)
        self._schedule_tick()

    def _build_ui(self) -> None:
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(self.root, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.video_panel = tk.Label(self.left_frame, bg="black")
        self.video_panel.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.seek_scale = tk.Scale(
            self.left_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            resolution=1,
            command=self.on_seek,
            label="Frame Seek",
        )
        self.seek_scale.pack(fill=tk.X, padx=8, pady=(0, 8))

        tk.Label(right, textvariable=self.video_text, anchor="w", justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(right, textvariable=self.frame_text, anchor="w", justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=4)
        tk.Label(right, textvariable=self.mode_text, anchor="w", justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=4)
        tk.Label(right, textvariable=self.pending_text, anchor="w", justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=4)
        tk.Label(right, textvariable=self.interval_text, anchor="w", justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=4)

        btn_frame = tk.Frame(right)
        btn_frame.pack(fill=tk.X, padx=8, pady=6)

        tk.Button(btn_frame, text="Play/Pause", command=self.toggle_play).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        tk.Button(btn_frame, text="Prev Frame", command=self.prev_frame).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        tk.Button(btn_frame, text="Next Frame", command=self.next_frame).grid(row=0, column=2, sticky="ew", padx=2, pady=2)

        tk.Button(btn_frame, text="Mark Start", command=self.mark_start).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        tk.Button(btn_frame, text="Mark End", command=self.mark_end).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        tk.Button(btn_frame, text="Undo", command=self.undo).grid(row=1, column=2, sticky="ew", padx=2, pady=2)

        tk.Button(btn_frame, text="Clear All", command=self.clear_all).grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        tk.Button(btn_frame, text="Save", command=self.save_all).grid(row=2, column=1, sticky="ew", padx=2, pady=2)
        tk.Button(btn_frame, text="Next Video", command=self.next_video).grid(row=2, column=2, sticky="ew", padx=2, pady=2)

        tk.Button(right, text="Quit (Save)", command=self.on_quit, bg="#f8d7da").pack(fill=tk.X, padx=8, pady=(8, 10))

        for i in range(3):
            btn_frame.grid_columnconfigure(i, weight=1)

        tk.Label(right, text="Intervals for current video:", anchor="w").pack(fill=tk.X, padx=8, pady=(8, 2))
        self.interval_listbox = tk.Listbox(right, height=20)
        self.interval_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<space>", self._on_key_toggle_play)
        self.root.bind_all("<Key-a>", self._on_key_mark_start)
        self.root.bind_all("<Key-A>", self._on_key_mark_start)
        self.root.bind_all("<Key-d>", self._on_key_mark_end)
        self.root.bind_all("<Key-D>", self._on_key_mark_end)
        self.root.bind_all("<Key-u>", self._on_key_undo)
        self.root.bind_all("<Key-U>", self._on_key_undo)
        self.root.bind_all("<Key-c>", self._on_key_clear)
        self.root.bind_all("<Key-C>", self._on_key_clear)
        self.root.bind_all("<comma>", self._on_key_prev_frame)
        self.root.bind_all("<period>", self._on_key_next_frame)
        self.root.bind_all("<Key-n>", self._on_key_next_video)
        self.root.bind_all("<Key-N>", self._on_key_next_video)
        self.root.bind_all("<Key-s>", self._on_key_save)
        self.root.bind_all("<Key-S>", self._on_key_save)
        self.root.bind_all("<Key-q>", self._on_key_quit)
        self.root.bind_all("<Key-Q>", self._on_key_quit)

    def _on_key_toggle_play(self, _event=None):
        self.toggle_play()
        return "break"

    def _on_key_mark_start(self, _event=None):
        self.mark_start()
        return "break"

    def _on_key_mark_end(self, _event=None):
        self.mark_end()
        return "break"

    def _on_key_undo(self, _event=None):
        self.undo()
        return "break"

    def _on_key_clear(self, _event=None):
        self.clear_all()
        return "break"

    def _on_key_prev_frame(self, _event=None):
        self.prev_frame()
        return "break"

    def _on_key_next_frame(self, _event=None):
        self.next_frame()
        return "break"

    def _on_key_next_video(self, _event=None):
        self.next_video()
        return "break"

    def _on_key_save(self, _event=None):
        self.save_all()
        return "break"

    def _on_key_quit(self, _event=None):
        self.on_quit()
        return "break"

    def _read_at(self, index: int) -> np.ndarray | None:
        if self.current_cap is None:
            return None
        self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, frame = self.current_cap.read()
        if not ok or frame is None:
            return None
        return frame

    def _read_next(self) -> np.ndarray | None:
        if self.current_cap is None:
            return None
        ok, frame = self.current_cap.read()
        if not ok or frame is None:
            return None
        return frame

    def _update_status(self) -> None:
        if self.current_video is None:
            return
        self.video_text.set(f"Video {self.video_index + 1}/{len(self.videos)}: {self.current_video.name}")
        self.frame_text.set(f"Frame: {self.frame_index}/{max(self.current_total_frames - 1, 0)}")
        self.mode_text.set("Mode: PLAY" if self.playing else "Mode: PAUSE")
        self.pending_text.set(
            "Pending start: None" if self.pending_start is None else f"Pending start: {self.pending_start}"
        )
        self.interval_text.set(f"Intervals: {len(self.intervals)}")

        self.interval_listbox.delete(0, tk.END)
        for i, (s, e) in enumerate(self.intervals, start=1):
            self.interval_listbox.insert(tk.END, f"{i:02d}. {s} {e}")

    def _update_video_panel(self) -> None:
        if self.current_video is None:
            return
        frame = draw_overlay(
            frame=self.current_frame,
            video_name=self.current_video.name,
            video_idx=self.video_index + 1,
            video_total=len(self.videos),
            frame_idx=self.frame_index,
            total_frames=self.current_total_frames,
            intervals=self.intervals,
            pending_start=self.pending_start,
            paused=not self.playing,
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        panel_w = self.left_frame.winfo_width() - 16
        panel_h = self.left_frame.winfo_height() - 70
        if panel_w <= 1:
            panel_w = 640
        if panel_h <= 1:
            panel_h = 360
        
        # Disable padding/expansion of the parent to break the feedback loop
        self.left_frame.pack_propagate(False)
        self.video_panel.pack_propagate(False)

        # OpenCV resize is much faster than PIL's Image.thumbnail for smooth playback
        rgb_resized = cv2.resize(rgb, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(rgb_resized)
        
        photo = ImageTk.PhotoImage(image=img)
        self.photo_ref = photo
        self.video_panel.configure(image=photo)

    def _sync_seek(self) -> None:
        self.seek_guard = True
        self.seek_scale.configure(to=max(self.current_total_frames - 1, 0))
        self.seek_scale.set(self.frame_index)
        self.seek_guard = False

    def _load_video(self, video_idx: int) -> bool:
        if video_idx < 0 or video_idx >= len(self.videos):
            return False

        if self.current_cap is not None:
            self.current_cap.release()
            self.current_cap = None

        video_path = self.videos[video_idx]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            messagebox.showerror("Open Video Failed", f"Cannot open video:\n{video_path}")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if total_frames <= 0:
            cap.release()
            messagebox.showwarning("Empty Video", f"Video has no frames:\n{video_path.name}")
            return False

        self.current_cap = cap
        self.current_video = video_path
        self.video_index = video_idx
        self.current_total_frames = total_frames
        self.current_fps = fps
        self.frame_index = 0
        self.pending_start = None
        self.playing = True

        existing = self.labels.get(video_path.stem)
        self.intervals = list(existing.intervals) if existing is not None else []

        frame = self._read_at(0)
        if frame is None:
            return False
        self.current_frame = frame
        self._sync_seek()
        self._update_status()
        self._update_video_panel()
        return True

    def _load_next_video(self, initial: bool = False) -> None:
        start = self.video_index + 1
        for idx in range(start, len(self.videos)):
            vid = self.videos[idx].stem
            if self.resume and vid in self.labels:
                continue
            if self._load_video(idx):
                return
        messagebox.showinfo("Done", "All videos finished.")
        self.on_quit()

    def _save_current_video_label(self) -> None:
        if self.current_video is None:
            return
        self.labels[self.current_video.stem] = VideoLabel(
            total_frames=int(self.current_total_frames), intervals=list(self.intervals)
        )

    def save_all(self) -> None:
        self._save_current_video_label()
        write_labels(self.output_txt, self.labels, self.ordered_ids)
        messagebox.showinfo("Saved", f"Saved labels to:\n{self.output_txt}")

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self._update_status()
        self._update_video_panel()

    def prev_frame(self) -> None:
        if self.current_video is None:
            return
        self.playing = False
        self.frame_index = max(0, self.frame_index - 1)
        frame = self._read_at(self.frame_index)
        if frame is not None:
            self.current_frame = frame
            self._sync_seek()
            self._update_status()
            self._update_video_panel()

    def next_frame(self) -> None:
        if self.current_video is None:
            return
        self.playing = False
        self.frame_index = min(self.current_total_frames - 1, self.frame_index + 1)
        frame = self._read_at(self.frame_index)
        if frame is not None:
            self.current_frame = frame
            self._sync_seek()
            self._update_status()
            self._update_video_panel()

    def mark_start(self) -> None:
        self.pending_start = self.frame_index
        self._update_status()

    def mark_end(self) -> None:
        if self.pending_start is None:
            messagebox.showwarning("No Start", "Please click 'Mark Start' first.")
            return
        s, e = self.pending_start, self.frame_index
        if e < s:
            s, e = e, s
        self.intervals.append((s, e))
        self.intervals.sort(key=lambda x: x[0])
        self.pending_start = None
        self._update_status()
        self._update_video_panel()

    def undo(self) -> None:
        if self.pending_start is not None:
            self.pending_start = None
        elif self.intervals:
            self.intervals.pop()
        self._update_status()
        self._update_video_panel()

    def clear_all(self) -> None:
        if not self.intervals and self.pending_start is None:
            return
        ok = messagebox.askyesno("Clear Intervals", "Clear all intervals in current video?")
        if not ok:
            return
        self.intervals.clear()
        self.pending_start = None
        self._update_status()
        self._update_video_panel()

    def next_video(self) -> None:
        if self.pending_start is not None:
            keep = messagebox.askyesno(
                "Pending Start",
                "There is a pending start without end. Discard it and continue?",
            )
            if not keep:
                return
            self.pending_start = None

        self._save_current_video_label()
        write_labels(self.output_txt, self.labels, self.ordered_ids)
        self._load_next_video(initial=False)

    def on_seek(self, value: str) -> None:
        if self.seek_guard or self.current_video is None:
            return
        idx = int(float(value))
        idx = max(0, min(idx, self.current_total_frames - 1))
        if idx == self.frame_index:
            return

        self.playing = False
        self.frame_index = idx
        frame = self._read_at(self.frame_index)
        if frame is not None:
            self.current_frame = frame
            self._update_status()
            self._update_video_panel()

    def _schedule_tick(self) -> None:
        t_start = time.time()
        
        if self.current_video is not None and self.playing:
            next_idx = self.frame_index + 1
            if next_idx < self.current_total_frames:
                frame = self._read_next()
                if frame is None:
                    # Fallback for codecs/containers where sequential read can fail intermittently.
                    frame = self._read_at(next_idx)
                if frame is not None:
                    self.frame_index = next_idx
                    self.current_frame = frame
                    self._sync_seek()
                    self._update_status()
                    self._update_video_panel()
            else:
                self.playing = False
                self._update_status()
                self._update_video_panel()

        ideal_delay = 1000.0 / max(self.current_fps, 1.0)
        elapsed_ms = (time.time() - t_start) * 1000.0
        
        # Determine actual delay by subtracting the processing time of the current frame
        delay = max(1, int(round(ideal_delay - elapsed_ms)))
        self.root.after(delay, self._schedule_tick)

    def on_quit(self) -> None:
        self._save_current_video_label()
        write_labels(self.output_txt, self.labels, self.ordered_ids)
        if self.current_cap is not None:
            self.current_cap.release()
            self.current_cap = None
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def select_video_dir_interactive() -> Path:
    try:
        root = tk.Tk()
        root.withdraw()
        directory = filedialog.askdirectory(title="Select a folder containing videos")
        root.destroy()
        if not directory:
            raise RuntimeError("No folder selected")
        return Path(directory)
    except Exception as e:
        raise RuntimeError(f"Failed to select folder interactively: {e}") from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple local video anomaly interval annotator")
    parser.add_argument(
        "--video-dir",
        default="",
        help="Folder containing videos. If omitted, a local folder picker is opened.",
    )
    parser.add_argument(
        "--output-txt",
        default="local_labeling_tools/labels_intervals.txt",
        help="Output txt path. Format: v1 total_frames s1 e1 s2 e2",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output txt and skip already labeled videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_dir = Path(args.video_dir) if args.video_dir else select_video_dir_interactive()
    if not video_dir.exists() or not video_dir.is_dir():
        raise SystemExit(f"Invalid video directory: {video_dir}")

    output_txt = Path(args.output_txt)
    if not output_txt.is_absolute():
        output_txt = Path.cwd() / output_txt

    labels = parse_existing_labels(output_txt)
    videos = collect_videos(video_dir)
    if not videos:
        raise SystemExit(f"No videos found in {video_dir}")

    ordered_ids = [p.stem for p in videos]

    print(f"[INFO] Video dir: {video_dir}")
    print(f"[INFO] Found videos: {len(videos)}")
    print(f"[INFO] Output txt: {output_txt}")
    print("[INFO] Annotation starts...")

    app = AnnotatorApp(
        videos=videos,
        labels=labels,
        output_txt=output_txt,
        ordered_ids=ordered_ids,
        resume=bool(args.resume),
    )
    app.run()
    print(f"\n[INFO] Done. Final labels saved to: {output_txt}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(130)
