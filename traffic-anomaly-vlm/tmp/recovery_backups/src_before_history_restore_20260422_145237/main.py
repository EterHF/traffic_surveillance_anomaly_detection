from __future__ import annotations

from src.pipeline.offline import OfflinePipeline
from src.settings import load_settings, parse_cli


def main() -> None:
    args = parse_cli()
    cfg = load_settings(args.config)
    if args.input_video:
        cfg.input_video = args.input_video

    pipeline = OfflinePipeline(cfg)
    results = pipeline.run(cfg.input_video)
    print(f"done, events={len(results)}")


if __name__ == "__main__":
    main()
