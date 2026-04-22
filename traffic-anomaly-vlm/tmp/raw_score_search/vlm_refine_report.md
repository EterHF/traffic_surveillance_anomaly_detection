# VLM Refine Span Search Report

## Goal

Find a better **raw coarse-span preset** for the next-stage VLM refinement.

The selection target is not the same as simple GT coverage:

- The span must cover the true anomaly as much as possible.
- The span should not be unnecessarily wide.
- The score curve should still show useful GT-vs-background lift.

## Files

- Search script: `tmp/run_vlm_refine_span_search.py`
- Full leaderboard: `tmp/raw_score_search/vlm_refine_results/leaderboard.json`
- Best preset outputs: `tmp/raw_score_search/vlm_refine_results/best`
- Extra candidate outputs:
  - `tmp/raw_score_search/balanced_source`
  - `tmp/raw_score_search/balanced_object`

## What Changed Compared With The First Sweep

This search adds extra metrics for VLM-oriented coarse spans:

- `avg_span_ratio`
- `mean_best_pred_to_gt_ratio`
- `mean_gt_lift`

and ranks configurations by a balance of:

- cover
- IoU
- full-coverage rate
- GT lift
- span width
- predicted-span count

## Final Conclusion

For **later VLM refine**, the safest default is still a **high-recall coarse preset**, not the tightest preset.

### Recommended Default Coarse Preset

- Scheme: `source_like_smooth3`
- Boundary: `peeks_hz1.35_exp14_24_mg12_min6`

Metrics:

- `mean_hit_ratio = 1.0000`
- `mean_best_cover = 0.9850`
- `mean_best_iou = 0.5519`
- `full80_ratio = 0.9787`
- `avg_num_pred_spans = 1.1064`
- `avg_span_ratio = 0.9149`
- `mean_best_pred_to_gt_ratio = 2.0091`
- `mean_gt_lift = 0.6045`

Why this is still the default:

- VLM refine can tolerate extra context.
- VLM refine cannot recover anomalies that were missed by the coarse span.
- After explicitly penalizing wide spans, this preset still stayed at the top.

So for your current pipeline, **coverage is still the first-order constraint**.

## Tighter Alternatives

### Source-Compatible Balanced Preset

- Scheme: `source_like_smooth5`
- Boundary: `peeks_hz1.35_exp10_16_mg12_min6`

Metrics:

- `mean_hit_ratio = 1.0000`
- `mean_best_cover = 0.8905`
- `mean_best_iou = 0.5649`
- `full80_ratio = 0.7872`
- `avg_num_pred_spans = 1.4255`
- `avg_span_ratio = 0.8434`
- `mean_best_pred_to_gt_ratio = 1.6678`
- `mean_gt_lift = 0.6000`

This is a useful **ablation** if you want somewhat tighter spans while staying very close to the current source feature design.

Outputs:

- `tmp/raw_score_search/balanced_source/best/best_config.json`
- `tmp/raw_score_search/balanced_source/best/videos/*`

### Object-Dominant Balanced Preset

- Scheme: `object_only_smooth5`
- Boundary: `peeks_hz1.35_exp10_16_mg12_min6`

Metrics:

- `mean_hit_ratio = 1.0000`
- `mean_best_cover = 0.9034`
- `mean_best_iou = 0.5790`
- `full80_ratio = 0.7872`
- `avg_num_pred_spans = 1.4681`
- `avg_span_ratio = 0.8472`
- `mean_best_pred_to_gt_ratio = 1.6813`
- `mean_gt_lift = 0.5784`

This is slightly tighter than the default high-recall route, but it still loses hard cases and is less source-compatible.

Outputs:

- `tmp/raw_score_search/balanced_object/best/best_config.json`
- `tmp/raw_score_search/balanced_object/best/videos/*`

## Important Tradeoff

The tighter presets reduce span width, but they also hurt the hard videos first.

Representative difficult examples:

- `v3`
- `v7`
- `v22`
- `v11`

In particular, `v7` remains a difficult case for every tighter preset. That means:

- narrowing spans too early is risky
- this video should be handled by later VLM refinement, not by forcing the raw span to be tight

## Recommendation For The Next Stage

For the next VLM-refine experiment, I recommend:

1. Use `source_like_smooth3 + peeks_hz1.35_exp14_24_mg12_min6` as the default coarse preset.
2. Use `source_like_smooth5 + peeks_hz1.35_exp10_16_mg12_min6` as the tighter ablation preset.
3. Compare both under the same VLM refine logic instead of continuing to over-optimize raw spans.

That comparison will tell you whether VLM is strong enough to turn the tighter but less safe spans into better final intervals.

## Source Bug Fixes Applied

Minimal source patches were made for the two confirmed issues:

1. `src/proposals/boundary_detector.py`
   - fixed the `by_thres` branch bug where `expanded_spans` was never initialized correctly

2. `src/features/feature_components/scene.py`
   `src/features/feature_builder.py`
   `src/pipeline/orchestrator.py`
   - added a minimal precomputed pair-signal path so offline low-res/layout change cues can be reused across overlapping windows instead of being recomputed every window

These patches are small and source-local; the search logic itself still lives in `tmp/`.
