import os
import requests
import pickle
import argparse
import json
import random
import shutil

import pandas as pd
import numpy as np

from os import walk
from collections import Counter

# Load configuration values

parser = argparse.ArgumentParser('Cross validation splits')
parser.add_argument('--config', type=str, required=True, help='path to config file')
args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)

## FUNCTIONS

def create_targets(df, cols="moa", drop_dummy=True):
    """Create the binary multi-label targets for each compound"""
    df['val'] = 1
    df_targets = pd.pivot_table(
        df,
        values=['val'],
        index='pert_iname',
        columns=[cols],
        fill_value=0
    )

    df_targets.columns.names = (None,None)
    df_targets.columns = df_targets.columns.droplevel(0)

    df_targets = df_targets.reset_index().rename({'index':'pert_iname'}, axis = 1)

    if drop_dummy:
        df_targets = df_targets.drop(columns=["dummy"])

    return df_targets


def train_test_split(train_cpds, test_cpds, df):
    df_trn = df.loc[df['pert_iname'].isin(train_cpds)].reset_index(drop=True)
    df_tst = df.loc[df['pert_iname'].isin(test_cpds)].reset_index(drop=True)
    return df_trn, df_tst


def create_shuffle_data(df_trn, target_cols):
    """Create shuffled train data where the replicates of each compound are given wrong target labels"""
    df_trn_cpy = df_trn.copy()
    df_trn_tgts = df_trn_cpy[target_cols].copy()
    rand_df = pd.DataFrame(np.random.permutation(df_trn_tgts), columns=df_trn_tgts.columns.tolist())
    df_trn_cpy.drop(target_cols, axis = 1, inplace = True)
    df_trn_cpy = pd.concat([df_trn_cpy, rand_df], axis = 1)
    return df_trn_cpy

def save_to_csv(df, path, file_name, compress=None):
    """saves dataframes to csv"""

    if not os.path.exists(path):
        os.mkdir(path)

    df.to_csv(os.path.join(path, file_name), index=False, compression=compress)

## MAIN PROCEDURE

# file name of features
file_cp = "_cellprofiler"
file_cnn = "_CNN"
file_dino = "_dino"

df_cellprofiler = pd.read_csv(
    os.path.join(config["output_folder"], f'cp{file_cp}.csv'),
    low_memory = False
)

df_cnn = pd.read_csv(
    os.path.join(config["output_folder"], f'cp{file_cnn}.csv'),
    low_memory = False
)

df_dino = pd.read_csv(
    os.path.join(config["output_folder"], f'cp{file_dino}.csv'),
    low_memory = False
)

print("Dataframe shapes")
print("CellProfiler:", df_cellprofiler.shape, "CP-CNN:", df_cnn.shape, "Input features:", df_dino.shape)

df_cpds_moas_lincs = pd.read_csv(os.path.join(config["output_folder"], config["split_moas_output"]))

print("MOA dataframe shape:",df_cpds_moas_lincs.shape)
print("Compounds:", len(df_cpds_moas_lincs.pert_iname.unique()))

all_cpds = df_cpds_moas_lincs['pert_iname'].unique()

df_cellprofiler = df_cellprofiler.loc[df_cellprofiler['pert_iname'].isin(all_cpds)].reset_index(drop=True)
df_cnn = df_cnn.loc[df_cnn['pert_iname'].isin(all_cpds)].reset_index(drop=True)
df_dino = df_dino.loc[df_dino['pert_iname'].isin(all_cpds)].reset_index(drop=True)

print("Dataframe shapes")
print("CellProfiler:",df_cellprofiler.shape, "CP-CNN:",df_cnn.shape, "Input features:",df_dino.shape)

df_cpds_moas = df_cpds_moas_lincs.copy()
df_cpds_moas.loc[:, 'moa'] = df_cpds_moas.loc[:,'moa'].fillna("dummy")

print("Unique MOAs", len(df_cpds_moas['moa'].unique()))

df_moa_targets = create_targets(df_cpds_moas, cols='moa', drop_dummy=False)

df_cellprofiler = df_cellprofiler.merge(df_moa_targets, on='pert_iname')
df_cnn = df_cnn.merge(df_moa_targets, on='pert_iname')
df_dino = df_dino.merge(df_moa_targets, on='pert_iname')

train_cpds = df_cpds_moas_lincs[df_cpds_moas_lincs['train']]['pert_iname'].unique()
test_cpds = df_cpds_moas_lincs[df_cpds_moas_lincs['test']]['pert_iname'].unique()

print("Training compounds:", len(train_cpds), "Testing compounds:",len(test_cpds))

df_cellprofiler_trn, df_cellprofiler_tst = train_test_split(train_cpds, test_cpds, df_cellprofiler)
df_dino_trn, df_dino_tst = train_test_split(train_cpds, test_cpds, df_dino)
df_cnn_trn, df_cnn_tst = train_test_split(train_cpds, test_cpds, df_cnn)

print("CellProfiler training:",df_cellprofiler_trn.shape, "CellProfiler test:",df_cellprofiler_tst.shape)
print("Input features training:",df_dino_trn.shape, "Input features test:",df_dino_tst.shape)
print("CP-CNN training:", df_cnn_trn.shape, "CP-CNN test:",df_cnn_tst.shape)

print("Aligning and saving")
target_cols = df_moa_targets.columns[1:]

df_cellprofiler_trn_shuf = create_shuffle_data(df_cellprofiler_trn, target_cols)
df_cellprofiler_tst_shuf = create_shuffle_data(df_cellprofiler_tst, target_cols)

df_dino_trn_shuf = create_shuffle_data(df_dino_trn, target_cols)
df_dino_tst_shuf = create_shuffle_data(df_dino_tst, target_cols)

df_cnn_trn_shuf = create_shuffle_data(df_cnn_trn, target_cols)
df_cnn_tst_shuf = create_shuffle_data(df_cnn_tst, target_cols)


save_to_csv(df_cellprofiler_trn, f"{config['output_folder']}/model_data/", f'train_data{file_cp}.csv.gz', compress="gzip")
save_to_csv(df_cellprofiler_tst, f"{config['output_folder']}/model_data/", f'test_data{file_cp}.csv.gz', compress="gzip")

save_to_csv(df_cnn_trn, f"{config['output_folder']}/model_data/", f'train_data{file_cnn}.csv.gz', compress="gzip")
save_to_csv(df_cnn_tst, f"{config['output_folder']}/model_data/", f'test_data{file_cnn}.csv.gz', compress="gzip")

save_to_csv(df_dino_trn, f"{config['output_folder']}/model_data/", f'train_data{file_dino}.csv.gz', compress="gzip")
save_to_csv(df_dino_tst, f"{config['output_folder']}/model_data/", f'test_data{file_dino}.csv.gz', compress="gzip")

save_to_csv(df_cellprofiler_trn_shuf, f"{config['output_folder']}/model_data/",
            f'train_shuffle_data{file_cp}.csv.gz', compress="gzip")
save_to_csv(df_cellprofiler_tst_shuf, f"{config['output_folder']}/model_data/",
            f'test_shuffle_data{file_cp}.csv.gz', compress="gzip")

save_to_csv(df_cnn_trn_shuf,  f"{config['output_folder']}/model_data/",
            f'train_shuffle_data{file_cnn}.csv.gz', compress="gzip")
save_to_csv(df_cnn_tst_shuf,  f"{config['output_folder']}/model_data/",
            f'test_shuffle_data{file_cnn}.csv.gz', compress="gzip")

save_to_csv(df_dino_trn_shuf,  f"{config['output_folder']}/model_data/",
            f'train_shuffle_data{file_dino}.csv.gz', compress="gzip")
save_to_csv(df_dino_tst_shuf,  f"{config['output_folder']}/model_data/",
            f'test_shuffle_data{file_dino}.csv.gz', compress="gzip")
save_to_csv(df_moa_targets, f"{config['output_folder']}/model_data/", f'target_labels.csv')

print("All files ready")
