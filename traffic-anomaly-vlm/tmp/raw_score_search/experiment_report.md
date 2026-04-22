# Raw Score Sweep Report

## Goal

Evaluate the **initial anomaly score only** on the full TU-DAT abnormal split, without tree building, and answer two questions:

1. Does the raw score show meaningful change around the ground-truth abnormal interval?
2. Can a first-pass span proposal mostly cover the GT interval?

All experiments in this report were kept under `tmp/` to avoid touching source pipeline code.

## Experiment Files

- Raw-score extractor: `tmp/orchestrator_raw_score.py`
- Sweep + visualization + metrics: `tmp/run_raw_score_sweep.py`
- Cached per-video features: `tmp/raw_score_search/cache/*/feature_rows.json`
- Aggregate leaderboard: `tmp/raw_score_search/results/leaderboard.json`
- Best config summary: `tmp/raw_score_search/results/best/best_config.json`
- Per-video plots and summaries: `tmp/raw_score_search/results/best/videos/*`

## Dataset

- Frames root: `/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal`
- GT manifest: `/nvme2/VAD_yemao/data/TU-DAT-25/abnormal.txt`
- Videos evaluated: `47`

## Method

1. Reuse the offline detector + tracker + track refiner flow, but stop before tree building.
2. Cache window-level raw feature rows for every abnormal video.
3. Build initial score curves from object and scene features.
4. Convert window scores into dense frame-level scores.
5. Use `BoundaryDetector` to produce first-pass spans.
6. Evaluate against GT using:
   - `mean_best_cover`
   - `mean_best_iou`
   - `hit_ratio`
   - `full80_ratio`
   - `avg_num_pred_spans`

## Best Coverage-Oriented Setting

Best row in the full sweep:

- Score scheme: `baseline_current_raw`
- Boundary scheme: `peeks_hz0.55_exp14_24`

Metrics:

- `mean_hit_ratio = 1.0000`
- `mean_best_cover = 0.9519`
- `mean_best_iou = 0.5529`
- `full80_ratio = 0.9149`
- `avg_num_pred_spans = 1.2128`

Interpretation:

- The initial score **can usually hit the GT interval**.
- For most videos, the first-pass span is already good enough as a coarse proposal.
- The main remaining issue is **span width**, not total miss.

## Source-Like Ablation

To check whether the gain mainly came from weight changes or from boundary choice, I re-ran a few source-like variants on the cached features.

### Source-like, no smoothing, default boundary

- `mean_hit_ratio = 1.0000`
- `mean_best_cover = 0.9812`
- `mean_best_iou = 0.5408`
- `full80_ratio = 0.9787`
- `avg_num_pred_spans = 1.1064`
- `avg_span_ratio = 0.9319`

### Source-like, no smoothing, searched best boundary

- `mean_hit_ratio = 1.0000`
- `mean_best_cover = 0.9869`
- `mean_best_iou = 0.5306`
- `full80_ratio = 0.9787`
- `avg_num_pred_spans = 1.0851`
- `avg_span_ratio = 0.9552`

### Source-like, smooth=5, default boundary

- `mean_hit_ratio = 1.0000`
- `mean_best_cover = 0.8639`
- `mean_best_iou = 0.5515`
- `full80_ratio = 0.7447`
- `avg_num_pred_spans = 1.5319`
- `avg_span_ratio = 0.8521`

### Source-like, smooth=5, searched best boundary

- `mean_hit_ratio = 1.0000`
- `mean_best_cover = 0.9519`
- `mean_best_iou = 0.5529`
- `full80_ratio = 0.9149`
- `avg_num_pred_spans = 1.2128`
- `avg_span_ratio = 0.9001`

## Main Finding

The current raw feature composition is **not weak**. In this experiment, the stronger effect came from the **boundary strategy** and the **cover-vs-width tradeoff**, not from adding many new features.

Put differently:

- If the goal is **high GT inclusion**, the current feature mix is already workable.
- If the goal is **tighter initial spans**, simply increasing coverage is misleading, because many high-cover settings make the predicted interval too broad.

## Worst Cases Under the Best Sweep Setting

These are the videos that still need attention:

- `v3`
  - `pred_spans = 2`
  - `mean_best_cover = 0.5126`
  - `mean_best_iou = 0.2383`
- `v7`
  - `pred_spans = 4`
  - `mean_best_cover = 0.4586`
  - `mean_best_iou = 0.4586`
- `v11`
  - `pred_spans = 2`
  - `mean_best_cover = 0.6381`
  - `mean_best_iou = 0.4556`
- `v22`
  - `pred_spans = 1`
  - `mean_best_cover = 0.6176`
  - `mean_best_iou = 0.3642`

Representative strong cases:

- `v4`
  - `mean_best_cover = 1.0000`
  - `mean_best_iou = 0.6761`
- `v36`
  - `mean_best_cover = 0.9821`
  - `mean_best_iou = 0.6548`

## Practical Recommendation

For the next stage, the most defensible temporary choice is:

- Keep the raw score close to the current `baseline_current_raw` weighting.
- Use `by_peeks` instead of `by_thres`.
- Treat the initial span as a **high-recall coarse proposal**, not a precise final boundary.

This is a better fit for the later VLM refinement stage than chasing perfect initial IoU too early.

## Notes

- The `by_thres` family was noticeably worse than the best `by_peeks` settings in this sweep.
- There is a source-side bug in `src/proposals/boundary_detector.py` for the `by_thres` branch, so threshold-mode span extraction was handled locally in `tmp/run_raw_score_sweep.py` for this experiment instead of patching source code.
