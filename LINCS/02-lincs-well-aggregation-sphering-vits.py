import argparse
import json
import torch
import os
import sys
import sklearn.metrics

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import profiling

# Load configuration values

parser = argparse.ArgumentParser('Feature aggregation')
parser.add_argument('--config', type=str, required=True, help='path to config file')
args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)

os.system(f'mkdir -p {config["output_folder"]}')
os.system(f'cp {config["baseline_results_folder"]}/cp_CNN_final.csv {config["output_folder"]}/cp_CNN.csv')

# Load metadata
meta = pd.read_csv(config["lincs_single_cell_metadata"])

# Define the feature file path
if config["feature_path"].endswith("npz"):
    # Load features in numpy format
    print("Loading features")
    features = np.load(config["feature_path"])
    features = features['features']
    print("Standardize features")
    features = StandardScaler().fit_transform(features)
elif config["feature_path"].endswith("pth"):
    print("Loading features")
    f = torch.load(config["feature_path"])
    print("Normalize features")
    f = nn.functional.normalize(f, dim=1, p=2)
    features = f.numpy()


print("Array shape:", features.shape)


# Get the total number of images and the number of features per image
total_images = features.shape[0]
features_per_image = features.shape[1]

print("Total images:", total_images)
print("Number of features per image:", features_per_image)

group_dict = meta.groupby('Key').groups
print("Grouping finished.")


site_level_data = []
site_level_features = []

for site_name in tqdm(list(group_dict.keys())):
    metadata = site_name.split('/')
    indices = group_dict[site_name]
    # mean_profile = np.median(features[indices], axis=0)
    mean_profile = np.median(features[indices], axis=0)

    site_level_data.append(
        {
            "Plate": metadata[0],
            "Well": metadata[1],
            "Treatment": meta["Treatment"][indices].unique()[0]
        }

    )
    site_level_features.append(mean_profile)

num_features = features_per_image
columns1 = ["Plate", "Well", "Treatment"] # dataset
columns2 = [i for i in range(num_features)]

sites1 = pd.DataFrame(columns=columns1, data=site_level_data)
sites2 = pd.DataFrame(columns=columns2, data=site_level_features)
sites = pd.concat([sites1, sites2], axis=1)

sites["Treatment_Clean"] = sites["Treatment"].apply(lambda x: "-".join([str(i) for i in x.split("-")[:2]]))

# Collapse well data
wells = sites.groupby(["Plate", "Well", "Treatment", "Treatment_Clean"]).mean().reset_index()
wells.to_csv(f"{config['output_folder']}/{config['raw_features_file']}")

print("Control wells:",sum(wells["Treatment"].isin(["DMSO@NA"])))

shN = profiling.SpheringNormalizer(wells.loc[wells["Treatment"].isin(["DMSO@NA"]), columns2], config["sphering_regularization"])
shD = shN.normalize(wells[columns2])

# Save whitened profiles
wells[columns2] = shD
wells.to_csv(f'{config["output_folder"]}/{config["output_file"]}', index=False)

