from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.orchestrator import Orchestrator


def args_run_offline():
    import argparse
    parser = argparse.ArgumentParser(description="Run the full pipeline in offline mode.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--input_video", type=str, default="", help="Path to one input video/frame directory.")
    return parser.parse_args()


def _print_single_result(result: dict):
    print(
        "video_id={video_id} num_pred_spans={num_pred_spans} "
        "num_final_spans={num_final_spans} runtime_sec={runtime:.2f}".format(
            video_id=result.get("video_id", ""),
            num_pred_spans=int(result.get("num_pred_spans", 0)),
            num_final_spans=int(result.get("num_final_spans", 0)),
            runtime=float(result.get("runtime_sec", 0.0)),
        )
    )


def _print_manifest_summary(summary: dict):
    print(
        "manifest={manifest} completed={completed}/{total} skipped={skipped} failed={failed} "
        "runtime_sec={runtime:.2f}".format(
            manifest=summary.get("manifest_path", ""),
            completed=int(summary.get("num_completed", 0)),
            total=int(summary.get("num_items", 0)),
            skipped=int(summary.get("num_skipped", 0)),
            failed=int(summary.get("num_failed", 0)),
            runtime=float(summary.get("runtime_sec", 0.0)),
        )
    )


def main():
    args = args_run_offline()
    orchestrator = Orchestrator(args.cfg)
    if args.input_video:
        result = orchestrator.run_offline(args.input_video)
    else:
        summary = orchestrator.run_manifest()
        _print_manifest_summary(summary)
        return

    if isinstance(result, dict):
        _print_single_result(result)


if __name__ == "__main__":
    main()
