#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root from this script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Defaults (can be overridden by CLI args below).
CONFIG="configs/default.yaml"
FRAME_ROOT="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal"
OUTPUT_ROOT="outputs/test/track_vis_features_refined"
VIDEO_ID="v1"
WINDOW_SIZE=16
TRACK_FIT_DEGREE=0
TRACK_HISTORY_LEN=0
TRAIL_LEN=100
REFINE_MAX_GAP=""
REFINE_MAX_CENTER_DIST=""
REFINE_MAX_SIZE_RATIO=""
FEATURE_PANEL_WIDTH=500
FEATURE_HISTORY_POINTS=100
OUTPUT_FPS=25
CODEC="mp4v"
RESIZE_MAX_SIDE=640
RESIZE_INTERPOLATION="area"

usage() {
	cat <<'EOF'
Usage: scripts/sh/visualize_track.sh [options]

Options:
	--video-id <id>                 Video id under frame-root (default: v1)
	--frame-root <path>             Root directory for frame folders
	--output-root <path>            Output directory for visualization video
	--config <path>                 Config yaml path
	--window-size <int>             Feature window size
	--track-fit-degree <int>        Polynomial fit degree for track features
	--track-history-len <int>       History length for track features
	--trail-len <int>               Number of previous positions to draw
	--refine-max-gap <int>          Optional override for refiner max_gap
	--refine-max-center-dist <f>    Optional override for refiner max_center_dist
	--refine-max-size-ratio <f>     Optional override for refiner max_size_ratio
	--feature-panel-width <int>     Right panel width in pixels
	--feature-history-points <int>  Number of history points in mini curves
	--output-fps <int>              Output video fps
	--codec <str>                   Video codec (e.g. mp4v)
	--resize-max-side <int>         Max side for input resize
	--resize-interpolation <str>    Resize interpolation
	-h, --help                      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--video-id) VIDEO_ID="$2"; shift 2 ;;
		--frame-root) FRAME_ROOT="$2"; shift 2 ;;
		--output-root) OUTPUT_ROOT="$2"; shift 2 ;;
		--config) CONFIG="$2"; shift 2 ;;
		--window-size) WINDOW_SIZE="$2"; shift 2 ;;
		--track-fit-degree) TRACK_FIT_DEGREE="$2"; shift 2 ;;
		--track-history-len) TRACK_HISTORY_LEN="$2"; shift 2 ;;
		--trail-len) TRAIL_LEN="$2"; shift 2 ;;
		--refine-max-gap) REFINE_MAX_GAP="$2"; shift 2 ;;
		--refine-max-center-dist) REFINE_MAX_CENTER_DIST="$2"; shift 2 ;;
		--refine-max-size-ratio) REFINE_MAX_SIZE_RATIO="$2"; shift 2 ;;
		--feature-panel-width) FEATURE_PANEL_WIDTH="$2"; shift 2 ;;
		--feature-history-points) FEATURE_HISTORY_POINTS="$2"; shift 2 ;;
		--output-fps) OUTPUT_FPS="$2"; shift 2 ;;
		--codec) CODEC="$2"; shift 2 ;;
		--resize-max-side) RESIZE_MAX_SIDE="$2"; shift 2 ;;
		--resize-interpolation) RESIZE_INTERPOLATION="$2"; shift 2 ;;
		-h|--help) usage; exit 0 ;;
		*)
			echo "Unknown option: $1" >&2
			usage
			exit 1
			;;
	esac
done

CMD=(
	python src/eval/visualize_track.py
	--config="${CONFIG}"
	--frame-root="${FRAME_ROOT}"
	--output-root="${OUTPUT_ROOT}"
	--video-id="${VIDEO_ID}"
	--trail-len="${TRAIL_LEN}"
	--window-size="${WINDOW_SIZE}"
	--track-fit-degree="${TRACK_FIT_DEGREE}"
	--track-history-len="${TRACK_HISTORY_LEN}"
	--feature-panel-width="${FEATURE_PANEL_WIDTH}"
	--feature-history-points="${FEATURE_HISTORY_POINTS}"
	--output-fps="${OUTPUT_FPS}"
	--codec="${CODEC}"
	--resize-max-side="${RESIZE_MAX_SIDE}"
	--resize-interpolation="${RESIZE_INTERPOLATION}"
)

if [[ -n "${REFINE_MAX_GAP}" ]]; then
	CMD+=(--refine-max-gap="${REFINE_MAX_GAP}")
fi
if [[ -n "${REFINE_MAX_CENTER_DIST}" ]]; then
	CMD+=(--refine-max-center-dist="${REFINE_MAX_CENTER_DIST}")
fi
if [[ -n "${REFINE_MAX_SIZE_RATIO}" ]]; then
	CMD+=(--refine-max-size-ratio="${REFINE_MAX_SIZE_RATIO}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
