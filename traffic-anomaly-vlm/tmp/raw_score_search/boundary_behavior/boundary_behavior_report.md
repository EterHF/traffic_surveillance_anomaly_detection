# Boundary Behavior Search

## Coverage Best
- scheme: `turn_focus__none__scene0.00__smooth3`
- boundary: `peeks_hz1.35_pg2_exp10_16_mg12_min4`
- mean_best_cover: `0.9739`
- mean_best_iou: `0.5464`
- avg_num_pred_spans: `1.1915`
- avg_span_ratio: `0.9071`
- wide_single_flag: `0.5319`

## VLM-Ready Best
- scheme: `source_like_smooth3`
- boundary: `thres_hz1.2_lz0_exp8_12_mg8_min8`
- mean_best_cover: `0.8687`
- mean_best_iou: `0.5706`
- avg_num_pred_spans: `1.7872`
- avg_span_ratio: `0.8251`
- wide_single_flag: `0.2979`
- merge_to_single_rate: `0.0000`

## Best `by_thres`
- scheme: `source_like_smooth3`
- boundary: `thres_hz1.2_lz0_exp8_12_mg8_min8`
- mean_best_cover: `0.8687`
- mean_best_iou: `0.5706`
- avg_num_pred_spans: `1.7872`
- avg_span_ratio: `0.8251`
- wide_single_flag: `0.2979`

## Method Summary
- `{
  "by_peeks": {
    "mean_cover": 0.7358787368129476,
    "mean_iou": 0.5163561088845524,
    "mean_wide_single": 0.1276595744680851,
    "mean_merge_to_single": 0.2369739952718676
  },
  "by_thres": {
    "mean_cover": 0.6186044641059298,
    "mean_iou": 0.4769785448358294,
    "mean_wide_single": 0.07679521276595745,
    "mean_merge_to_single": 0.018284574468085103
  }
}`

## Top-100 VLM-Ready Frequency
- method: `{"by_peeks": 18, "by_thres": 82}`
- merge_gap: `{"0": 9, "12": 18, "2": 9, "4": 18, "8": 46}`
- peak_expand_sum: `{"12": 9, "16": 42, "20": 45, "6": 4}`

## Top-100 VLM-Ready Means
- merge_gap -> wide_single_flag: `{"0": 0.2978723404255319, "12": 0.17021276595744678, "2": 0.2978723404255319, "4": 0.2978723404255319, "8": 0.2876965772432932}`
- merge_gap -> merge_to_single_rate: `{"0": 0.0, "12": 0.425531914893617, "2": 0.0, "4": 0.0, "8": 0.004162812210915819}`
- peak_expand_sum -> avg_span_ratio: `{"12": 0.8003023374372467, "16": 0.820310595832178, "20": 0.8277902252028072, "6": 0.7774985267637382}`
- peak_expand_sum -> mean_best_cover: `{"12": 0.8560615872422321, "16": 0.8589848691084693, "20": 0.8566035737855442, "6": 0.8239150644605205}`

## Takeaways
- `merge_gap` and `peak_expand` are the main drivers of over-wide single spans; they matter more than `min_span_len` in the current grid.
- `by_thres` is worth re-checking after the source bug fix, but it still needs to beat `by_peeks` on both coverage and compactness before becoming a default coarse proposal.
- For later VLM usage, the better preset is not the one with the highest raw coverage alone, but the one that preserves multiple compact candidate spans without dropping the anomaly body.
