# Track Parameter Calibration

Videos: v1, v3, v10, v15, v20

Statistics are computed on sampled/refined tracks outside manifest anomaly intervals.

## Distributions

### area_ratio
count: 5628

| percentile | value |
|---:|---:|
| p5 | 1.000426 |
| p25 | 1.004319 |
| p50 | 1.017256 |
| p75 | 1.048144 |
| p85 | 1.074963 |
| p90 | 1.097868 |
| p95 | 1.153347 |
| p98 | 1.237758 |
| p99 | 1.326832 |

### disappear_edge_norm
count: 104

| percentile | value |
|---:|---:|
| p5 | 0.026744 |
| p25 | 0.069206 |
| p50 | 0.103557 |
| p75 | 0.138243 |
| p85 | 0.161198 |
| p90 | 0.179206 |
| p95 | 0.205508 |
| p98 | 0.216353 |
| p99 | 0.227123 |

### edge_distance_norm
count: 5750

| percentile | value |
|---:|---:|
| p5 | 0.061743 |
| p25 | 0.098240 |
| p50 | 0.128217 |
| p75 | 0.174146 |
| p85 | 0.199305 |
| p90 | 0.213214 |
| p95 | 0.226836 |
| p98 | 0.237848 |
| p99 | 0.241758 |

### low_disp_step_px
count: 1000

| percentile | value |
|---:|---:|
| p5 | 0.006397 |
| p25 | 0.020528 |
| p50 | 0.043887 |
| p75 | 0.102851 |
| p85 | 0.171641 |
| p90 | 0.282792 |
| p95 | 0.611582 |
| p98 | 1.545092 |
| p99 | 2.766454 |

### low_disp_step_ratio
count: 1000

| percentile | value |
|---:|---:|
| p5 | 0.000142 |
| p25 | 0.000485 |
| p50 | 0.001430 |
| p75 | 0.004257 |
| p85 | 0.007231 |
| p90 | 0.011048 |
| p95 | 0.027654 |
| p98 | 0.074294 |
| p99 | 0.148033 |

### step_px
count: 5628

| percentile | value |
|---:|---:|
| p5 | 0.015141 |
| p25 | 0.209738 |
| p50 | 0.948278 |
| p75 | 3.052677 |
| p85 | 5.206800 |
| p90 | 7.062299 |
| p95 | 11.824342 |
| p98 | 20.565528 |
| p99 | 26.261862 |

### step_ratio
count: 5628

| percentile | value |
|---:|---:|
| p5 | 0.000374 |
| p25 | 0.007086 |
| p50 | 0.039585 |
| p75 | 0.104763 |
| p85 | 0.153877 |
| p90 | 0.193931 |
| p95 | 0.243886 |
| p98 | 0.323517 |
| p99 | 0.385225 |

### track_disp_ratio
count: 94

| percentile | value |
|---:|---:|
| p5 | 0.007908 |
| p25 | 1.245848 |
| p50 | 2.774654 |
| p75 | 6.445287 |
| p85 | 8.973975 |
| p90 | 9.975641 |
| p95 | 12.447300 |
| p98 | 13.857081 |
| p99 | 14.683448 |

### triplet_min_step_px
count: 5515

| percentile | value |
|---:|---:|
| p5 | 0.009920 |
| p25 | 0.135697 |
| p50 | 0.715977 |
| p75 | 2.401105 |
| p85 | 4.298101 |
| p90 | 5.751588 |
| p95 | 8.663684 |
| p98 | 14.406013 |
| p99 | 20.642629 |

### triplet_min_step_ratio
count: 5515

| percentile | value |
|---:|---:|
| p5 | 0.000247 |
| p25 | 0.004643 |
| p50 | 0.031442 |
| p75 | 0.076363 |
| p85 | 0.129147 |
| p90 | 0.160105 |
| p95 | 0.219903 |
| p98 | 0.261675 |
| p99 | 0.304684 |

## Recommendation

| parameter | value |
|---|---:|
| STATIC_MOVE_RATIO | 0.300000 |
| DISAPPEAR_FAR_EDGE_RATIO | 0.179206 |
| DISAPPEAR_LOOKAHEAD | 2 |
| DISAPPEAR_MIN_TRACK_LEN | 3 |
| TURN_CHANGE_RATIO | 0.500000 |
| SPEED_SCALING_FACTOR | 15 |
| MIN_TURN_SPEED_RATIO | 0.030000 |
| MIN_DIR_MOTION_RATIO | 0.030000 |
| MIN_DIR_MOTION_PX | 2 |
| TOPK_AGG_COUNT | 3 |
| EDGE_UNSTABLE_RATIO | 0.030000 |
| MAX_MOTION_AREA_RATIO | 1.500000 |
| MAX_STEP_RATIO | 1 |
| REPLACEMENT_CENTER_RATIO | 0.060000 |
| REPLACEMENT_IOU | 0.050000 |
| TRACK_POINT_Y_RATIO | 0.600000 |
