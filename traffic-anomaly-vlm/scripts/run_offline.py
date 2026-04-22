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
    all_nodes = orchestrator.run_offline(args.input_video)
    print(f"Total detected nodes: {len(all_nodes)}")
    for i, node in enumerate(all_nodes):
        print(f"Node {i}: span=({node.start_frame}, {node.end_frame}), prior_score={node.span_prior_score}, vlm_score={node.vlm_score}")


if __name__ == "__main__":
    main()