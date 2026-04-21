video_root="/nvme2/VAD_yemao/data/TU-DAT-25/frames/abnormal"
output_root="/nvme2/VAD_yemao/traffic-anomaly-vlm/outputs/test/boundary"
# list all video ids in the video root
video_ids=$(ls $video_root)
# run tmp/compute_features.py for each video id
for video_id in $video_ids; do
    echo "Processing video: $video_id"
    python tmp/compute_features.py --video-id $video_id --out-dir $output_root
done