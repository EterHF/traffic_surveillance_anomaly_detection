# Component Observation Report

## Goal

Before doing a larger parameter/grid search, inspect which feature components are actually useful for the initial coarse anomaly interval.

This round focused on:

1. fixing repeated scene-feature computation in source code
2. observing the effect of each component
3. deciding which components are worth future large-scale search

## Source Fixes

Minimal source-side changes were applied:

- `src/features/feature_components/scene.py`
  - added `SceneFeatureExtractor.compute_pair_signals(...)`
  - unified lowres / layout / clip / raft pair-signal computation
- `src/pipeline/orchestrator.py`
  - offline flow now precomputes reusable scene pair signals once per sampled video
  - clip / raft are handled the same way as lowres / layout
- `src/features/feature_builder.py`
  - already supported pair-signal slices; no structural rewrite needed

This means:

- no repeated lowres/layout work across overlapping windows
- no repeated clip/raft work across overlapping windows once they are enabled
- style stays consistent across all scene cues

## Experiment Files

### Full abnormal set, lightweight components

- `tmp/raw_score_search/component_observation_base.json`

### Representative CLIP subset

Subset:

- `v1 v3 v4 v7 v11 v20 v22 v30 v31 v33 v36 v37`

Files:

- `tmp/raw_score_search/cache_clip_subset`
- `tmp/raw_score_search/component_observation_clip_subset.json`

### Representative RAFT subset

Subset:

- `v3 v7 v22 v36`

Files:

- `tmp/raw_score_search/cache_raft_subset`
- `tmp/raw_score_search/component_observation_raft_subset.json`

## Full-Set Observation: Low-Cost Components

On the full abnormal split, the most useful scene cue was:

- `track_layout_eventness`

It slightly outperformed:

- `lowres_eventness`

and both were clearly more useful than adding nothing.

### Best single scene features on the full set

From `component_observation_base.json`:

- `track_layout_eventness`
  - `mean_best_cover = 0.9405`
  - `mean_best_iou = 0.5461`
  - `avg_span_ratio = 0.9081`
  - `gt_lift = 0.5995`
- `lowres_eventness`
  - `mean_best_cover = 0.9639`
  - `mean_best_iou = 0.5238`
  - `avg_span_ratio = 0.9407`
  - `gt_lift = 0.5490`

Interpretation:

- `lowres` is a little better at broad coverage
- `track_layout` is a little better at keeping boundaries less inflated while preserving GT lift

### Best lightweight combined schemes on the full set

- `object_layout`
- `object_lowres_layout`
- `object_only`
- `object_lowres`

The ordering is close, but the signal is:

- scene cues help
- `track_layout` is the first scene component worth keeping
- `lowres` is still useful as a complementary cue

## CLIP Observation

CLIP was evaluated first on a representative subset rather than the full set, because full-set CLIP augmentation is noticeably more expensive.

### CLIP as a standalone scene feature

On the subset:

- `clip_eventness`
  - `mean_best_cover = 0.9299`
  - `mean_best_iou = 0.5624`
  - `avg_span_ratio = 0.9269`
  - `mean_best_pred_to_gt_ratio = 1.6896`
  - `gt_lift = 0.5078`

Compared with the same subset:

- `track_layout_eventness`
  - `mean_best_cover = 0.9467`
  - `mean_best_iou = 0.5689`
  - `gt_lift = 0.5612`
- `lowres_eventness`
  - `mean_best_cover = 0.9534`
  - `mean_best_iou = 0.5789`
  - `gt_lift = 0.5173`

Interpretation:

- CLIP is not useless
- but it is **not stronger than lowres/layout**
- and it does not justify immediate full-set sweep by itself

### CLIP as an add-on to object / lowres / layout

Subset schemes showed:

- `object_clip` was effectively no better than `object_only`
- `object_lowres_layout_clip` was worse than `object_lowres`
- `object_lowres_layout_clip` was also worse than `object_lowres_layout`

This is the most important CLIP result:

**In the current formulation, CLIP does not improve the coarse interval enough to justify making it a default component.**

## RAFT Observation

RAFT was evaluated on a smaller hard-case subset because its cost is much higher than CLIP.

### RAFT as a standalone scene feature

- `raft_eventness`
  - `mean_best_cover = 0.7136`
  - `mean_best_iou = 0.3653`
  - `hit_ratio = 0.75`
  - `gt_lift = 0.1722`

This is clearly weaker than:

- `lowres_eventness`
- `track_layout_eventness`

on the same subset.

### RAFT as an add-on

- `object_raft` was effectively no better than `object_only`
- `object_lowres_layout_raft` was worse than `object_lowres_layout`

Conclusion:

**RAFT is not a priority for the next search stage.**

At the moment it increases cost much more than it improves the coarse span.

## Practical Recommendation

For the next large-scale search and later VLM-refine stage:

### Keep

- object features
- `track_layout_eventness`
- `lowres_eventness`

### Delay

- CLIP

Reason:

- useful enough to keep in mind
- not strong enough yet to beat cheaper cues
- should only come back after the low-cost branch is better tuned

### Deprioritize

- RAFT

Reason:

- expensive
- weak standalone effect
- weak additive effect in the current design

## What To Search Next

The most promising next search space is still:

- object branch weights
- `track_layout_eventness` weight
- `lowres_eventness` weight
- boundary parameters

and the search objective should still prefer:

- high GT coverage
- reasonable span width
- enough span count for later refinement

instead of blindly minimizing the number of coarse intervals.

## Bottom Line

If the question is "which scene components are worth carrying into the next round?", the answer is:

1. `track_layout_eventness`
2. `lowres_eventness`
3. `clip_eventness` only as a later optional experiment
4. `raft_eventness` not worth prioritizing right now
