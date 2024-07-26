import os
import numpy as np
import pandas as pd
import skimage.io
import torch
from torchvision.transforms import v2
from tqdm import tqdm

def fold_channels(image, channel_width, mode="ignore"):
    output = np.reshape(image, (image.shape[0], channel_width, -1), order="F")
    if mode == "ignore":
        pass
    elif mode == "drop":
        output = output[:, :, 0:-1]
    elif mode == "apply":
        mask = output[:, :, -1:]
        output = output[:, :, 0:-1] * mask
    return output

def channel_to_rgb(channel):
    px = np.concatenate((channel[np.newaxis, :, :], channel[np.newaxis, :, :], channel[np.newaxis, :, :]), axis=0)
    tensor = torch.Tensor(px)[None, ...]
    normalized_tensor = v2.functional.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalized_tensor

gpu = 1
device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
dinov2_vits14_reg.eval()
dinov2_vits14_reg.to(device)

base_path = "/scr/data/LINCS-DINO/max_concentration_set/"
csv_path = os.path.join(base_path, "sc-metadata.csv")

df = pd.read_csv(csv_path)

all_features = []

for i, img_rel_path in tqdm(enumerate(df['Image_Name']), total=df.shape[0]):
    img_path = os.path.join(base_path, img_rel_path)
    try:
        img = skimage.io.imread(img_path)
        fold = fold_channels(img, img.shape[0])
        image_batch = torch.cat([channel_to_rgb(fold[17:-17, 17:-17, j]) for j in range(5)])
        output = dinov2_vits14_reg.forward_features(image_batch.to(device))
        features = output["x_norm_clstoken"].cpu().detach().numpy()
        all_features.append(features)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

if all_features:
    all_features = np.concatenate(all_features, axis=0)
    np.savez_compressed("features.npz", all_features)
else:
    print("No features were extracted. Please check the paths and image processing steps.")