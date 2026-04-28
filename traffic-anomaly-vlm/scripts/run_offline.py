from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.orchestrator import Orchestrator


def args_run_offline():
    import argparse
    parser = argparse.ArgumentParser(description="Run the full pipeline in offline mode on a single video.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    return parser.parse_args()

def main():
    args = args_run_offline()
    orchestrator = Orchestrator(args.cfg)
    result = orchestrator.run_offline(args.input_video)
    if isinstance(result, dict):
        print(
            "video_id={video_id} num_pred_spans={num_pred_spans} "
            "mean_best_cover={cover:.4f} mean_best_iou={iou:.4f} runtime_sec={runtime:.2f}".format(
                video_id=result.get("video_id", ""),
                num_pred_spans=int(result.get("num_pred_spans", 0)),
                cover=float(result.get("gt_metrics", {}).get("mean_best_cover", 0.0)),
                iou=float(result.get("gt_metrics", {}).get("mean_best_iou", 0.0)),
                runtime=float(result.get("runtime_sec", 0.0)),
            )
        )
        return
    if result is None:
        print("run_offline finished with no returned nodes")
        return
    print(f"Total detected nodes: {len(result)}")
    for i, node in enumerate(result):
        print(f"Node {i}: span=({node.start_frame}, {node.end_frame}), prior_score={node.span_prior_score}, vlm_score={node.vlm_score}")


if __name__ == "__main__":
    main()
