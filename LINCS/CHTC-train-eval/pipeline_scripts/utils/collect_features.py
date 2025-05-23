import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

## ARGUMENTS
parser = argparse.ArgumentParser('Feature collection')
parser.add_argument('--feature_dir', type=str, required=True, help='path to feature files')
parser.add_argument('--output_file', type=str, required=True, help='path where output is written')
parser.add_argument('--feature_dim', type=int, required=True, help='number of features')
args = parser.parse_args()

## FUNCTIONS
def load(f):
    try:
        with open(args.feature_dir + f, "rb") as source:
            data = torch.load(source)
        return (data[0].numpy(), data[1])
    except:
        print("Error with",f)
        return None

## MAIN PROCEDURE
all_files = [x for x in os.listdir(args.feature_dir) if x.endswith(".pth")]

print("Loading feature files")
with Pool(50) as p:
    arrays = list(tqdm(p.imap(load, all_files), total=len(all_files)))

print("All files loaded?", len(arrays) == len(all_files))

total = np.sum([len(x[1]) for x in arrays])
features = np.zeros((total, args.feature_dim))
print("Organizing features")
for data in tqdm(arrays):
    tensor, pos = data
    features[pos,:] = tensor

print("Saving feature matrix")
np.savez(args.output_file, features=features)