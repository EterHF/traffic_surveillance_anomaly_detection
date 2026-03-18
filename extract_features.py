# Extract features using CLIP
# Assume video data is stored in a directory structure like:
# root/
#   train/
#     video1/   
#       frame1.jpg
#       frame2.jpg
#       ...  
#     video2/
#   test/
#     video1/
#     video2/   

import os
import glob
import torch
from transformers import CLIPModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Set device to GPU if available for faster feature extraction
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

# Use torchvision transforms instead of AutoProcessor. This is much faster
# and avoids multiprocessing issues during DataLoader batching.
clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), 
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

class FrameDataset(Dataset):
    def __init__(self, frame_paths):
        self.frame_paths = frame_paths

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path = self.frame_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            pixel_values = clip_transform(image)
            return pixel_values
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, 224, 224)

def extract_features(root_dir, output_dir, batch_size=256, num_workers=8):
    """
    Extracts features for all videos in `root_dir` and saves them to `output_dir`.
    Expects `root_dir` to contain subdirectories for each video.
    """
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} not found. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    video_dirs = glob.glob(os.path.join(root_dir, "*/"))
    
    for video_dir in tqdm(video_dirs, desc=f"Processing videos in {os.path.basename(root_dir)}"):
        video_name = os.path.basename(os.path.normpath(video_dir))
        out_path = os.path.join(output_dir, f"{video_name}.pt")
        
        # Skip if already extracted
        if os.path.exists(out_path):
            continue
            
        # Get sorted frames
        frames = sorted(glob.glob(os.path.join(video_dir, "*.jpg")) + glob.glob(os.path.join(video_dir, "*.png"))) 
        if not frames:
            continue
            
        dataset = FrameDataset(frames)
        # Using DataLoader with multiple workers to speed up image reading and preprocessing
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        features_list = []
        with torch.no_grad():
            for batch_pixels in dataloader:
                batch_pixels = batch_pixels.to(device)
                # Compute image features and normalize
                features = model.get_image_features(pixel_values=batch_pixels)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                features_list.append(features.cpu())
                
        if features_list:
            video_features = torch.cat(features_list, dim=0)
            torch.save(video_features, out_path)

if __name__ == "__main__":
    # Base directory according to your actual data structure
    data_root = "root"
    
    # Input directories containing the video folders
    train_frames_root = os.path.join(data_root, "train")
    test_frames_root = os.path.join(data_root, "test")
    
    # Output directories for the extracted features (.pt files)
    train_output_root = os.path.join(data_root, "train_features")
    test_output_root = os.path.join(data_root, "test_features")
    
    print("Starting feature extraction...")
    
    # Process test data (if exists)
    print("\n--- Test Data ---")
    extract_features(test_frames_root, test_output_root)
    
    # Process train data (if exists)
    print("\n--- Train Data ---")
    extract_features(train_frames_root, train_output_root)
    
    print("\nExtraction complete!")