# Local Video Interval Annotator

A minimal local tool for labeling anomaly intervals for videos in a folder.

## Features

- Open a folder and label videos one by one in filename order.
- Each video supports multiple anomaly intervals.
- Save all results into one txt file.
- Output line format:

```text
v1 total_frames s1 e1 s2 e2
```

Example:

```text
01_Accident_001 1850 320 390 1022 1110
```

## Run

From project root:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vad_yemao
python local_labeling_tools/annotate_video_intervals.py --video-dir /path/to/videos --output-txt local_labeling_tools/labels_intervals.txt
```

If your script is at project root (current repo layout), use:

```bash
python annotate_video_intervals.py --video-dir /path/to/videos --output-txt local_labeling_tools/labels_intervals.txt
```

Or omit `--video-dir` to open a local folder picker:

```bash
python local_labeling_tools/annotate_video_intervals.py --output-txt local_labeling_tools/labels_intervals.txt
```

Or in current repo layout:

```bash
python annotate_video_intervals.py --output-txt local_labeling_tools/labels_intervals.txt
```

Resume mode (skip videos already in txt):

```bash
python local_labeling_tools/annotate_video_intervals.py --video-dir /path/to/videos --output-txt local_labeling_tools/labels_intervals.txt --resume
```

Or in current repo layout:

```bash
python annotate_video_intervals.py --video-dir /path/to/videos --output-txt local_labeling_tools/labels_intervals.txt --resume
```

## UI Buttons

All operations are done by clicking buttons in the right panel:

- `Play/Pause`: start or stop playback
- `Prev Frame` / `Next Frame`: step frame-by-frame
- `Mark Start`: mark anomaly interval start at current frame
- `Mark End`: mark anomaly interval end at current frame and finalize one interval
- `Undo`: clear pending start first, then remove last interval
- `Clear All`: clear all intervals in current video
- `Save`: save all labels to txt immediately
- `Next Video`: save current video and move to next one
- `Quit (Save)`: save and exit

There is also a `Frame Seek` slider under the video for quick positioning.

Keyboard shortcuts are also supported:

- `SPACE`: play/pause
- `A`: mark start
- `D`: mark end
- `U`: undo
- `C`: clear all
- `,` / `.`: prev/next frame
- `N`: next video
- `S`: save
- `Q`: quit and save

## Notes

- Frame IDs are 0-based.
- Existing txt will be loaded automatically if found.
- The script saves after each video to avoid data loss.
