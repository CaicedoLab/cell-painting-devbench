import os
import requests
import pickle
import argparse
import pandas as pd
import numpy as np
import re
from os import walk
from collections import Counter
import random
import shutil

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

#cp /scr/data/LINCS-DINO/data/cp_CNN_final.csv ./celldino_ps8_ViTs

cp_data_path = 'celldino_ps8_ViTs/'
cpd_split_path = 'celldino_ps8_ViTs/'

# file name of features
file_cp = "_cellprofiler_final"
file_cnn = "_CNN_final"
file_dino = "_dino_final"

df_cellprofiler = pd.read_csv(
    os.path.join(cp_data_path, f'cp{file_cp}.csv'),
    low_memory = False
)

df_cnn = pd.read_csv(
    os.path.join(cp_data_path, f'cp{file_cnn}.csv'),
    low_memory = False
)

df_dino = pd.read_csv(
    os.path.join(cp_data_path, f'cp{file_dino}.csv'),
    low_memory = False
)

# print(df_cellprofiler.shape, df_cnn.shape, df_dino.shape)

df_cpds_moas_lincs = pd.read_csv(os.path.join(cpd_split_path, f'split_moas_cpds_celldino_ps8_ViTs_final.csv'))

print(df_cpds_moas_lincs.shape)
print(len(df_cpds_moas_lincs.pert_iname.unique()))
df_cpds_moas_lincs.head()

all_cpds = df_cpds_moas_lincs['pert_iname'].unique()

df_cellprofiler = df_cellprofiler.loc[df_cellprofiler['pert_iname'].isin(all_cpds)].reset_index(drop=True)
df_cnn = df_cnn.loc[df_cnn['pert_iname'].isin(all_cpds)].reset_index(drop=True)
df_dino = df_dino.loc[df_dino['pert_iname'].isin(all_cpds)].reset_index(drop=True)

# print(df_cellprofiler.shape, df_cnn.shape, df_dino.shape)

df_cpds_moas = df_cpds_moas_lincs.copy()
df_cpds_moas.loc[:, 'moa'] = df_cpds_moas.loc[:,'moa'].fillna("dummy")

print(len(df_cpds_moas['moa'].unique()))

df_moa_targets = create_targets(df_cpds_moas, cols='moa', drop_dummy=False)
df_moa_targets

df_cellprofiler = df_cellprofiler.merge(df_moa_targets, on='pert_iname')
df_cnn = df_cnn.merge(df_moa_targets, on='pert_iname')
df_dino = df_dino.merge(df_moa_targets, on='pert_iname')

# print(df_cellprofiler.shape, df_cnn.shape, df_dino.shape)

train_cpds = df_cpds_moas_lincs[df_cpds_moas_lincs['train']]['pert_iname'].unique()
test_cpds = df_cpds_moas_lincs[df_cpds_moas_lincs['test']]['pert_iname'].unique()

print(len(train_cpds), len(test_cpds))

df_cellprofiler_trn, df_cellprofiler_tst = train_test_split(train_cpds, test_cpds, df_cellprofiler)
df_dino_trn, df_dino_tst = train_test_split(train_cpds, test_cpds, df_dino)
df_cnn_trn, df_cnn_tst = train_test_split(train_cpds, test_cpds, df_cnn)

print(df_cellprofiler_trn.shape, df_cellprofiler_tst.shape)
print(df_dino_trn.shape, df_dino_tst.shape)
# print(df_cnn_trn.shape, df_cnn_tst.shape)

target_cols = df_moa_targets.columns[1:]

df_cellprofiler_trn_shuf = create_shuffle_data(df_cellprofiler_trn, target_cols)
df_cellprofiler_tst_shuf = create_shuffle_data(df_cellprofiler_tst, target_cols)

df_dino_trn_shuf = create_shuffle_data(df_dino_trn, target_cols)
df_dino_tst_shuf = create_shuffle_data(df_dino_tst, target_cols)

df_cnn_trn_shuf = create_shuffle_data(df_cnn_trn, target_cols)
df_cnn_tst_shuf = create_shuffle_data(df_cnn_tst, target_cols)

df_dino_trn_shuf

save_to_csv(df_cellprofiler_trn, "celldino_ps8_ViTs/model_data/", f'train_data{file_cp}.csv.gz', compress="gzip")
save_to_csv(df_cellprofiler_tst, "celldino_ps8_ViTs/model_data/", f'test_data{file_cp}.csv.gz', compress="gzip")

save_to_csv(df_cnn_trn, "celldino_ps8_ViTs/model_data/", f'train_data{file_cnn}.csv.gz', compress="gzip")
save_to_csv(df_cnn_tst, "celldino_ps8_ViTs/model_data/", f'test_data{file_cnn}.csv.gz', compress="gzip")

save_to_csv(df_dino_trn, "celldino_ps8_ViTs/model_data/", f'train_data{file_dino}.csv.gz', compress="gzip")
save_to_csv(df_dino_tst, "celldino_ps8_ViTs/model_data/", f'test_data{file_dino}.csv.gz', compress="gzip")

save_to_csv(df_cellprofiler_trn_shuf, "celldino_ps8_ViTs/model_data/",
            f'train_shuffle_data{file_cp}.csv.gz', compress="gzip")
save_to_csv(df_cellprofiler_tst_shuf, "celldino_ps8_ViTs/model_data/",
            f'test_shuffle_data{file_cp}.csv.gz', compress="gzip")

save_to_csv(df_cnn_trn_shuf,  "celldino_ps8_ViTs/model_data/",
            f'train_shuffle_data{file_cnn}.csv.gz', compress="gzip")
save_to_csv(df_cnn_tst_shuf,  "celldino_ps8_ViTs/model_data/",
            f'test_shuffle_data{file_cnn}.csv.gz', compress="gzip")

save_to_csv(df_dino_trn_shuf,  "celldino_ps8_ViTs/model_data/",
            f'train_shuffle_data{file_dino}.csv.gz', compress="gzip")
save_to_csv(df_dino_tst_shuf,  "celldino_ps8_ViTs/model_data/",
            f'test_shuffle_data{file_dino}.csv.gz', compress="gzip")
save_to_csv(df_moa_targets, "celldino_ps8_ViTs/model_data/", f'target_labels_final.csv')
