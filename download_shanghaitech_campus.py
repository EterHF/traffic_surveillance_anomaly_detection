import kagglehub

# Download latest version
path = kagglehub.dataset_download("nikanvasei/shanghaitech-campus-dataset",
                                  output_dir="./data/shanghaitech_campus/train")

print("Path to dataset files:", path)

path = kagglehub.dataset_download("nikanvasei/shanghaitech-campus-dataset-test",
                                  output_dir="./data/shanghaitech_campus/test")

print("Path to dataset files:", path)