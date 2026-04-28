root="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal"

for video in $(ls $root); do
    echo "Processing $video"
    python scripts/run_offline.py \
        --input_video "$root/$video" \
        --cfg ./configs/default.yaml
done