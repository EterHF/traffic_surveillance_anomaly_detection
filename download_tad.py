import os
import kagglehub

root_dir = "./data"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# Download latest version
path = kagglehub.dataset_download("nikanvasei/traffic-anomaly-dataset-tad",
                                  output_dir=root_dir,
                                  force_download=True)

print("Path to dataset files:", path)