# traffic-anomaly-vlm

YOLO26 + Ultralytics `model.track(...)` for detection and tracking, plus local Transformers VLM for anomaly reasoning.

## Design

- Perception is isolated in `src/perception/` and uses a single `DetectorTracker` wrapper.
- VLM is isolated in `src/vlm/` and uses local `from_pretrained()` loading.
- Modules communicate through shared schemas in `src/schemas.py`.
- Trigger logic is plug-and-play in `src/triggers/`.

## Quick Start

```bash
cd traffic-anomaly-vlm
pip install -r requirements.txt
python scripts/smoke_test.py
```

Run offline pipeline:

```bash
python -m src.main --config configs/default.yaml --input-video /path/to/video.mp4
```

Run with frame-image dataset input (directory or list file):

```bash
python -m src.main --config configs/default.yaml --input-video /path/to/frames_dir
python -m src.main --config configs/default.yaml --input-video /path/to/frames_list.txt
```

Run pure VLM baseline (no YOLO/track/trigger modules):

```bash
python scripts/run_pure_vlm_baseline.py \
	--config configs/default.yaml \
	--manifest /path/to/baseline_manifest.txt
```

`--manifest` supports:

- One video per line with tab-separated fields: `frame_dir<TAB>[[start,end],[start,end]]`
- JSONL style line: `{"video_id":"vid001","frame_dir":"/abs/path/frames","gt_intervals":[[48,143]]}`
- If only `frame_dir` is provided, that video is treated as unlabeled (inference-only).

Outputs of pure baseline are saved to `outputs/results/pure_vlm_baseline/` by default:

- one JSON per video (window/chunk/frame scores and predicted intervals)
- `metrics_summary.json` with video/frame level ROC-AUC and AP (when labels are present)

## Config

Main config is in `configs/default.yaml`:

- `perception.yolo_model`: YOLO26 weights path
- `perception.tracker`: `bytetrack.yaml` or `botsort.yaml`
- `vlm.model_path`: local Transformers model directory
- `video.input_fps`: source fps for frame-image dataset (used by sampler)
- `pure_vlm_baseline.*`: optional baseline overrides (window/stride/chunks/smoothing/threshold)

## Visual Debug Outputs

Offline run will automatically save:

- `outputs/debug/window_score_curve.png`: trigger score curve over windows
- `outputs/debug/event_timeline.png`: event intervals and peak frame markers
- `outputs/debug/detection_preview_montage.jpg`: sampled frames with detection/tracking boxes
- `outputs/debug/keyframe_montage.jpg`: keyframe collage per event

## Run Without VLM

Set `vlm.enable: false` in config (default) to skip local VLM loading and inference.
In this mode, pipeline still runs detection/tracking, proposal generation, and all visual debug outputs.

## Notes

- `src/vlm/infer.py` is model-agnostic but may need minor adaptation for specific VLM families.
- `src/triggers/base.py` is the stable extension interface for future trigger methods.
