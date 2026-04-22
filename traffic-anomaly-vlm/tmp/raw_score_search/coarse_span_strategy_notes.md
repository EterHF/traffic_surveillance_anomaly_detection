# Coarse Span Strategy Notes

## Why direct coarse-span VLM refine is weak

- Our coverage-first preset keeps the anomaly body, but often collapses many local peaks into one wide span.
- In the current search, the coverage-best preset
  - `turn_focus__none__scene0.00__smooth3`
  - `peeks_hz1.35_pg2_exp10_16_mg12_min4`
  has:
  - `mean_best_cover = 0.9739`
  - `raw_span_count = 23.34`
  - `final_span_count = 1.19`
  - `merge_to_single_rate = 0.7872`
- This means a large fraction of videos start from many local cues, but end as one coarse span after expansion and merging.
- Asking a VLM to score that large span is close to asking a generic clip-level classifier, which weakens temporal localization.

## What the current experiments suggest

- If we optimize only for coverage, `by_peeks` remains strongest.
- If we optimize for later VLM usage, compact `by_thres` proposals become more attractive:
  - `source_like_smooth3`
  - `thres_hz1.2_lz0_exp8_12_mg8_min8`
  gives:
  - `mean_best_cover = 0.8687`
  - `mean_best_iou = 0.5706`
  - `num_pred_spans = 1.7872`
  - `avg_span_ratio = 0.8251`
  - `merge_to_single_rate = 0.0`
- A tree-friendly `by_peeks` compromise also exists:
  - `source_like_smooth3`
  - `peeks_hz1_pg2_exp4_8_mg12_min4`
  gives:
  - `mean_best_cover = 0.8195`
  - `mean_best_iou = 0.5726`
  - `num_pred_spans = 1.8298`
  - `avg_span_ratio = 0.7937`
  - `wide_single_flag = 0.0638`

## More practical VLM integration options

### 1. Dual-bank proposal design

- Bank A: recall envelope
  - Use a high-recall preset to guarantee the anomaly body is not missed.
- Bank B: micro-proposal bank
  - Use a compact preset to keep smaller spans around local peaks.
- The VLM only scores Bank B.
- Bank A is used only as a safety envelope or for later span merging.

Why this helps:
- It keeps coverage and explainability at the same time.
- It avoids feeding oversized coarse spans into the VLM.

### 2. Peak-bank VLM scoring

- Keep unmerged local peak clusters before final coarse merging.
- Turn each cluster into a short fixed-width evidence bag.
- Ask the VLM to score each bag independently.
- Merge only VLM-positive neighboring bags afterwards.

Why this helps:
- Each final decision can be traced back to concrete local peaks.
- It is more interpretable than asking one question over a long span.

### 3. Boundary-probe VLM

- Use the raw score to find a center proposal.
- Do not ask the VLM for the whole coarse span.
- Instead, ask two small questions:
  - whether the left boundary should move inward/outward
  - whether the right boundary should move inward/outward
- Only probe a short neighborhood around each boundary.

Why this helps:
- The VLM is used on small, localized evidence.
- The cost is lower and the refinement target is clearer.

## Recommended next method direction

The most practical and explainable next step is:

`high-recall envelope + compact micro-proposal bank + small-span VLM scoring`

Concretely:

1. Keep one high-recall span preset only to guarantee anomaly coverage.
2. Build a second compact proposal bank from smaller spans.
3. Score only the compact bank with the VLM.
4. Merge compact positives under the constraint of the high-recall envelope.
5. Build the tree only on compact positives, not on oversized coarse spans.

This is closer to the strengths of VADTree/EventVAD/SUVAD-style small-span reasoning, but better aligned with the traffic setting and the current raw-score behavior.

## Literature pointers

- GEBD defines generic event boundaries as meaningful temporal chunks rather than category-specific actions:
  - https://arxiv.org/abs/2101.10511
- EventVAD uses statistical event boundary detection to reduce MLLM complexity:
  - https://arxiv.org/abs/2504.13092
- VADTree builds a hierarchical tree from generic event nodes and injects priors for node-wise anomaly reasoning:
  - https://arxiv.org/abs/2510.22693
- LAVAD uses frame descriptions and LLM temporal aggregation:
  - https://arxiv.org/abs/2404.01014
- SUVAD emphasizes semantic video understanding with MLLM-generated descriptions:
  - https://colab.ws/articles/10.1109%2Ficassp49660.2025.10888431
