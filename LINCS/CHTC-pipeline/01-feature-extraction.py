import numpy as np
import torch
import os
import skimage
import pandas as pd
from torchvision.transforms import v2
from tqdm import tqdm

def fold_channels(image, channel_width, mode="ignore"):
    output = np.reshape(image, (image.shape[0], channel_width, -1), order="F")
    if mode == "ignore":
        pass
    elif mode == "drop":
        output = output[:, :, 0:-1]
    elif mode == "apply":
        mask = output["image"][:, :, -1:]
        output = output[:, :, 0:-1] * mask
    return output

def channel_to_rgb(channel):
    px = np.concatenate(
        (channel[np.newaxis, :, :], channel[np.newaxis, :, :], channel[np.newaxis, :, :]),
        axis=0)
    tensor = torch.Tensor(px)[None, ...]
    normalized_tensor = v2.functional.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalized_tensor

def process_batch(image_batch):
    image_tensor = torch.cat(image_batch, axis=0)
    output = dinov2_vits14_reg.forward_features(image_tensor.to(device))
    features = output["x_norm_clstoken"].cpu().detach().numpy()
    return features

# Load model
gpu = 1
device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'

dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
dinov2_vits14_reg.eval()
dinov2_vits14_reg.to(device)

base_path = "/scr/data/LINCS-DINO/max_concentration_set"
csv_path = os.path.join(base_path, "sc-metadata.csv")

df = pd.read_csv(csv_path)

batch_size = 128  # Set your batch size here
all_features = []
batch_images = []
batch_names = []
max_images = 1000  # Maximum number of images to process
processed_images = 0  # Counter for processed images

for i, img_rel_path in tqdm(enumerate(df['Image_Name']), total=df.shape[0]):
    if processed_images >= max_images:
        break  # Stop processing after reaching max_images
    
    img_path = os.path.join(base_path, img_rel_path)
    try:
        img = skimage.io.imread(img_path)
        fold = fold_channels(img, img.shape[0])
        
        # Process each channel and append to the batch
        for j in range(5):
            rgb_tensor = channel_to_rgb(fold[17:-17, 17:-17, j])
            batch_images.append(rgb_tensor)
            batch_names.append(img_path)
        
        processed_images += 1  # Increment the counter for each image processed

        # If batch size is reached, process the batch
        if len(batch_images) >= batch_size:
            features = process_batch(batch_images)
            
            # Reshape and concatenate features for each image
            num_images = len(batch_names) // 5
            reshaped_features = features.reshape(num_images, 5, -1)
            concatenated_features = reshaped_features.reshape(num_images, -1)
            all_features.append(concatenated_features)
            
            batch_images = []
            batch_names = []

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Process any remaining images in the last batch
if batch_images:
    features = process_batch(batch_images)
    
    num_images = len(batch_names) // 5
    reshaped_features = features.reshape(num_images, 5, -1)
    concatenated_features = reshaped_features.reshape(num_images, -1)
    all_features.append(concatenated_features)

if all_features:
    all_features = np.concatenate(all_features, axis=0)
    np.savez_compressed("features.npz", all_features)
else:
    print("No features were extracted. Please check the paths and image processing steps.")