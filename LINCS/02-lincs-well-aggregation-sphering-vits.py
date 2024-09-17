# import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import sklearn.metrics
import os
import sys

from tqdm import tqdm

#sys.path.append("../profiling/")
#import profiling

sys.path.append("../profiling/")
import norm

output_folder = './celldino_ps8_ViTs'
output_file = "well_level_profiles_vits_celldino_ps8.csv"
REG_PARAM = 1e-5

# Load metadata
meta = pd.read_csv("../../data/sc-metadata.csv")

import torch
from sklearn.preprocessing import StandardScaler

# Define the feature file path
#feature_path = '/scr/data/LINCS-DINO/meta_features/ViTs/LINCS_celldino_ps8_features.pth'
feature_path = '../../data/nikita_features_all_lincs.npz'
# Load the .pth file into a tensor
features = np.load(feature_path)

features = features['features']
print("Array shape:", features.shape)


# Get the total number of images and the number of features per image
total_images = features.shape[0]
features_per_image = features.shape[1]

print("Total images:", total_images)
print("Number of features per image:", features_per_image)

#COMMENTING OUT STANDARD SCALER
#scaled_features = StandardScaler().fit_transform(features)

# cell_names = np.concatenate(([f[1] for f in open_files]))
# order, ordered_features = (np.array(t) for t in zip(*sorted(zip(cell_names, scaled_features))))

meta

group_dict = meta.groupby('Key').groups
print("Grouping finished.")

#all_data = pd.concat([meta, pd.DataFrame(features)], axis=1)

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
wells[:10]

wells.to_csv(f"./{output_folder}/Wells_Prewhitened_ViT_large_LINCS.csv")

sum(wells["Treatment"].isin(["DMSO@NA"]))

whN = norm.WhiteningNormalizer(wells.loc[wells["Treatment"].isin(["DMSO@NA"]), columns2], REG_PARAM)
whD = whN.normalize(wells[columns2])

# Save whitened profiles
wells[columns2] = whD
wells.to_csv(f'{output_folder}/{output_file}', index=False)
