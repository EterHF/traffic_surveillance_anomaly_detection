# Object + Lowres + Layout Refined Search

## Goal

Run a more focused search after the component observation stage.

This round only searched:

- object features
- `lowres_eventness`
- `track_layout_eventness`
- boundary-detector parameters

and explicitly penalized:

- overly wide coarse spans
- low-value single-span solutions
- excessive predicted-span inflation

## Files

- Search script: `tmp/run_object_scene_refined_search.py`
- Leaderboard: `tmp/raw_score_search/object_scene_refined/leaderboard.json`
- Best overall preset:
  - `tmp/raw_score_search/object_scene_refined/best_overall/best_config.json`
- Best compact preset:
  - `tmp/raw_score_search/object_scene_refined/best_compact/best_config.json`

## Main Result

### Best Overall

- Scheme: `turn_focus__none__scene0.00__smooth3`
- Boundary: `peeks_hz1.35_exp10_16_mg12_min6`

Metrics:

- `hit_ratio = 1.0000`
- `full80_ratio = 0.9787`
- `mean_best_cover = 0.9739`
- `mean_best_iou = 0.5464`
- `avg_span_ratio = 0.9071`
- `mean_best_pred_to_gt_ratio = 2.0028`
- `num_pred_spans = 1.1915`
- `gt_lift = 0.5543`

Interpretation:

- after refining the search space, the strongest default coarse preset is still mostly **object-driven**
- the most useful object direction is a **turn-focused** mix:
  - `dir_turn_rates_sum`
  - `speed_turn_rates_sum`
  - `turn_active_ratio`
  - with smaller residual support

### Best Compact

- Scheme: `turn_focus__layout_only__scene0.15__smooth5`
- Boundary: `peeks_hz1.15_exp12_20_mg12_min6`

Metrics:

- `hit_ratio = 1.0000`
- `full80_ratio = 0.8936`
- `mean_best_cover = 0.9456`
- `mean_best_iou = 0.5456`
- `avg_span_ratio = 0.8892`
- `mean_best_pred_to_gt_ratio = 1.8962`
- `num_pred_spans = 1.2340`
- `gt_lift = 0.5685`

Interpretation:

- this is the cleaner “less bloated” preset
- the only scene cue that survived in this compact solution was:
  - `track_layout_eventness`
- `lowres_eventness` dropped out of the best compact preset

## What This Means

The current evidence is:

1. for the default coarse proposal, **object motion/turn cues remain the backbone**
2. `track_layout_eventness` is the only scene cue that still helps in a compact preset
3. `lowres_eventness` can still be useful, but it is not necessary in the best refined solution

This matches the earlier component observation:

- `track_layout_eventness` was stronger than `lowres_eventness`
- CLIP did not beat the cheaper scene cues
- RAFT was not worth its cost

## Recommended Presets For Next-Step VLM Refine

### Preset A: Default High-Recall Coarse Span

Use when the priority is:

- do not miss the anomaly
- let later VLM/tree handle refinement

Preset:

- `turn_focus__none__scene0.00__smooth3`
- `peeks_hz1.35_exp10_16_mg12_min6`

### Preset B: Compact Ablation

Use when the priority is:

- slightly tighter raw spans
- cleaner later VLM refinement input
- acceptable recall drop on hard videos

Preset:

- `turn_focus__layout_only__scene0.15__smooth5`
- `peeks_hz1.15_exp12_20_mg12_min6`

## Important Risk

The compact preset is cleaner overall, but it drops hard cases earlier.

Worst videos under `best_compact` still include:

- `v4`
- `v3`
- `v7`
- `v11`

So:

- if your next step is immediate VLM refine, use `best_overall` as the primary preset
- use `best_compact` as the contrastive ablation

## Practical Conclusion

At this stage, the most defensible next-step path is:

1. keep the coarse proposal mainly object-driven
2. only keep `track_layout_eventness` as the first scene add-on
3. postpone broader CLIP/RAFT grid search
4. compare `best_overall` vs `best_compact` under the same downstream VLM refine

That comparison will tell you whether a cleaner compact raw span is actually worth the recall drop once VLM is introduced.
