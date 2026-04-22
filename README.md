# traffic_surveillance_anomaly_detection

## Tree Pipeline Config (Recommended)

Use root-level `tree_pipeline` in `traffic-anomaly-vlm/configs/default.yaml` as the
primary config for the modular proposal pipeline.

Current flow:
`raw score -> VLM refined coarse span -> build tree -> final fine score + anomaly spans`

The previous `components.tree_pipeline` section is still supported for compatibility,
but new experiments should prefer root-level `tree_pipeline`.

## Runtime Config Usage

Build one runtime config object from loaded settings and pass it directly:

```python
from src.proposals.tree_pipeline import build_routed_event_tree, runtime_config_from_settings

runtime_cfg = runtime_config_from_settings(cfg)
result = build_routed_event_tree(
	frame_ids=frame_ids,
	images=images,
	tracks_per_frame=tracks_per_frame,
	track_scores=track_scores,
	object_scores=object_scores,
	seed_scores=seed_scores,
	output_dir=output_dir,
	runtime_cfg=runtime_cfg,
)
```

This avoids manually building separate coarse/tree/selector/router configs and keeps
experiments easier to manage.
