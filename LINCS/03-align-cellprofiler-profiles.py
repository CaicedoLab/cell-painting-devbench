import os
import pathlib
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
#import split_compounds 
##import split_cpds_moas

# Copy well level profiles to '../source_data'
# Load Data file for Our Experiment
df_dino = pd.read_csv("celldino_ps8_ViTs/well_level_profiles_vits_celldino_ps8.csv")
df_CNN = pd.read_csv('data/well_level_profiles_cpcnn_LINCS_1e-5_final.csv')
print(df_dino.shape)

os.system("wget https://github.com/broadinstitute/lincs-profiling-complementarity/raw/master/1.Data-exploration/Profiles_level4/cell_painting/cellpainting_lvl4_cpd_replicate_datasets/cp_level4_cpd_replicates.csv.gz")

# Load CellProfiler's datafile

# Download cp_level4_cpd_replicates.csv.gz from https://github.com/broadinstitute/lincs-profiling-complementarity/blob/master/1.Data-exploration/Profiles_level4/cell_painting/cellpainting_lvl4_cpd_replicate_datasets/cp_level4_cpd_replicates.csv.gz
df_cellprofiler = pd.read_csv('cp_level4_cpd_replicates.csv.gz',
    compression='gzip',
    low_memory = False
)
print(df_cellprofiler.shape)

# Exclude DMSO
df_dino = df_dino[df_dino['Treatment'] != 'DMSO@NA'].reset_index(drop=True)

#df_CNN = df_CNN[df_CNN['Treatment'] != 'DMSO@NA'].reset_index(drop=True)
#df_CNN["Treatment_Clean"] = df_CNN["broad_sample"].apply(lambda x: '-'.join(x.split('-')[:2]))

df_cellprofiler = df_cellprofiler[df_cellprofiler['broad_id'] != 'DMSO'].reset_index(drop=True)

print(df_dino.shape, df_cellprofiler.shape)

common_treatment = list(set(df_dino["Treatment_Clean"].unique())
                        & set(df_cellprofiler["broad_id"].unique()))

len(common_treatment)

# Select rows with common treatments only
df_cellprofiler = df_cellprofiler.loc[df_cellprofiler['broad_id'].isin(common_treatment)]
df_dino = df_dino.loc[df_dino['Treatment_Clean'].isin(common_treatment)]
# df_CNN = df_CNN.loc[df_CNN['Treatment_Clean'].isin(common_treatment)]

print(len(df_cellprofiler["broad_id"].unique()))
print(len(df_dino['Treatment_Clean'].unique()))
#print(len(df_CNN["Treatment_Clean"].unique()))

# Filter for only max dose
idx = df_cellprofiler.groupby(['broad_id'])['Metadata_dose_recode'].transform(max) == \
        df_cellprofiler['Metadata_dose_recode']
df_cellprofiler = df_cellprofiler[idx]

print(df_cellprofiler.shape)
print(df_dino.shape)
# print(df_CNN.shape)

# Convert moa annotation to lower case
df_cellprofiler['moa'] = df_cellprofiler['moa'].apply(lambda x: x.lower())

# Create moa-compound dictionary
df_cpds_moas = df_cellprofiler.drop_duplicates(['broad_id','moa'])[['broad_id','moa']]
cpds_moa = dict(zip(df_cpds_moas['broad_id'], df_cpds_moas['moa']))
len(cpds_moa)

df_cpds_moas.to_csv('celldino_ps8_ViTs/moa_annotation.csv', index=False)

# Concatenate moa for three datasets
df_dino["moa"]= df_dino["Treatment_Clean"].map(cpds_moa)
# df_CNN['moa'] = df_CNN['Treatment_Clean'].map(cpds_moa)

print(len(df_cellprofiler["moa"].unique()),
      len(df_dino['moa'].unique()))
# Add compound name 'pert_iname' for dino and cpcnn features
pertname = df_cellprofiler.drop_duplicates(['pert_iname','broad_id'])[['pert_iname','broad_id']]
pertname_dict = dict(zip(pertname['broad_id'], pertname['pert_iname']))

df_dino['pert_iname'] = df_dino['Treatment_Clean'].map(pertname_dict)
# df_CNN['pert_iname'] = df_CNN['Treatment_Clean'].map(pertname_dict)

#Save file to csv
out_dir = 'celldino_ps8_ViTs'

df_cellprofiler.to_csv(f"{out_dir}/cp_cellprofiler_final.csv",index=False)
df_dino.to_csv(f"{out_dir}/cp_dino_final.csv",index=False)
#df_CNN.to_csv(f"{out_dir}/cp_CNN_final.csv",index=False)

# create cpd name - moa dictionary
df_cpds_moas = df_cellprofiler.drop_duplicates(['pert_iname','moa'])[['pert_iname','moa']]
cpds_moa = dict(zip(df_cpds_moas['pert_iname'], df_cpds_moas['moa']))
len(cpds_moa)

def sort_moas(cpds_moa):
    """
    Sort MOAs based on the number of compounds that are attributed to them in ASCENDING order.
    This is HIGHLY Required before performing the compounds split into train & test.
    """
    cpds_moa_split = {cpd:cpds_moa[cpd].split('|') for cpd in cpds_moa}
    moa_listts = [moa for moa_lt in cpds_moa_split.values() for moa in moa_lt]
    moa_count_dict = {ky:val for ky,val in sorted(Counter(moa_listts).items(),key=lambda item: item[1])}
    moa_lists = list(moa_count_dict.keys())
    return moa_lists
def create_cpd_moa_df(cpds_moa):
    """
    Create a dataframe that comprises of compounds with their corresponding MOAs, including three additional
    columns: "test", "train" & "marked" which are needed for the compounds split.
    """
    cpds_moa_split = {cpd:cpds_moa[cpd].split('|') for cpd in cpds_moa}
    df_pert_cpds_moas = pd.DataFrame([(key, moa) for key,moa_list in cpds_moa_split.items() for moa in moa_list],
                                     columns = ['pert_iname', 'moa'])
    df_pert_cpds_moas['train'] = False
    df_pert_cpds_moas['test'] = False
    df_pert_cpds_moas['marked'] = df_pert_cpds_moas['train'] | df_pert_cpds_moas['test']
    return df_pert_cpds_moas

def split_cpds_moas(cpd_moas_dict, train_ratio=0.8, test_ratio=0.2):
    """
    This function splits compounds into test & train data based on the number of MOAs that are attributed to them,
    i.e. if the MOAs are present in just one compound, the compounds for those specific MOAs are given to only the
    train data, but if present in more than one compound, the compounds for that MOA are divided into Train/Test
    split based on the test/train ratio.

    - This function was extracted from https://rpubs.com/shantanu/lincs_split_moa
    and then refactored to Python

    Args:
         cpd_moas_dict: Dictionary comprises of compounds as the keys and their respective MOAs (Mechanism of actions)
         as the values
         train_ratio: A decimal value that represent what percent of the data should be given to the train set
         test_ratio: A decimal value that represent what percent of the data should be given to the test set

    Returns:
            df: pandas dataframe containing compounds, MOAs and three new boolean columns (Train, Test, Marked)
            indicating whether a compound is in Train or Test dataset.
    """
    ##preliminary funcs
    moa_list = sort_moas(cpd_moas_dict)
    df = create_cpd_moa_df(cpd_moas_dict)

    random.seed(333)
    for moa in moa_list:
        df_moa = df[df['moa'] == moa].reset_index(drop=True)
        no_cpd = df_moa.shape[0]

        if no_cpd == 1:
            n_trn, n_tst = 1, 0
        else:
            n_trn, n_tst = np.floor(no_cpd*train_ratio), np.ceil(no_cpd*test_ratio),

        n_tst_mk = sum(df_moa.test)
        n_trn_mk = sum(df_moa.train)
        moa_mk = df_moa[df_moa['marked']].copy()
        moa_not_mk = df_moa[~df_moa['marked']].copy()
        trn_needed = int(n_trn - n_trn_mk)
        tst_needed = int(n_tst - n_tst_mk)
        n_cpds_needed = trn_needed + tst_needed
        ##print(moa, df_moa.shape[0], moa_not_mk.shape[0], n_cpds_needed, trn_needed, tst_needed)

        trn_needed = max(trn_needed, 0)
        tst_needed = max(tst_needed, 0)
        trn_flg = list(np.concatenate((np.tile(True, trn_needed), np.tile(False, tst_needed))))
        trn_flg = random.sample(trn_flg, n_cpds_needed)
        tst_flg = [not boolean for boolean in trn_flg]
        moa_not_mk.train = trn_flg
        moa_not_mk.test = tst_flg
        if moa_not_mk.shape[0] > 0:
            moa_not_mk.marked = True
        df_moa = pd.concat([moa_not_mk, moa_mk], axis=0, ignore_index=True)
        df_other_moa = df[df['moa'] != moa].reset_index(drop=True)
        df_otrs_mk = df_other_moa[df_other_moa['marked']].reset_index(drop=True)
        df_otrs_not_mk= df_other_moa[~df_other_moa['marked']].reset_index(drop=True)
        df_otrs_not_mk = df_otrs_not_mk[['pert_iname', 'moa']].merge(moa_not_mk.drop(['moa'], axis=1),
                                                                     on=['pert_iname'], how='left').fillna(False)

        df = pd.concat([df_moa, df_otrs_mk, df_otrs_not_mk], axis=0, ignore_index=True)
        df[['train', 'test']] = df[['train', 'test']].apply(lambda x: x.astype(bool))

    return df

df_pert_cpds_moas = split_cpds_moas(cpds_moa)
df_pert_cpds_moas

len(df_pert_cpds_moas[df_pert_cpds_moas['test']]['pert_iname'].unique()) ##moas in the test data

def get_moa_count(df):
    """
    Get the number of compounds MOAs are present in, for both train and test data
    """
    df_moa_ct = df.drop(['pert_iname'], axis=1).groupby(['moa']).agg(['sum'])
    df_moa_ct.columns = df_moa_ct.columns.droplevel(1)
    df_moa_ct.reset_index(inplace=True)
    return df_moa_ct

def get_test_ratio(df):
    if df['test'] > 0:
        return df["train"] / df["test"]
    return 0

df_moa_count = get_moa_count(df_pert_cpds_moas)

df_moa_count['test_ratio'] = df_moa_count.apply(get_test_ratio, axis=1)

## All MOAs found in test should be found in train data, so this should output nothing...GOOD!
df_moa_count[(df_moa_count['train'] == 0) & (df_moa_count['test'] >= 1)]

## moas that are represented in more than one compounds (> 1),
## present in train set but not present in test set
df_moa_count[(df_moa_count['train'] > 1) & (df_moa_count['test'] == 0)]

len(df_pert_cpds_moas[df_pert_cpds_moas['train']]['pert_iname'].unique()) ##no of compounds in train data

len(df_pert_cpds_moas[df_pert_cpds_moas['test']]['pert_iname'].unique()) ##no of compounds in test data

def save_to_csv(df, path, file_name, compress=None):
    """saves dataframes to csv"""

    if not os.path.exists(path):
        os.mkdir(path)
    df.to_csv(os.path.join(path, file_name), index=False, compression=compress)

save_to_csv(df_pert_cpds_moas, "celldino_ps8_ViTs", 'split_moas_cpds_celldino_ps8_ViTs_final.csv')
                                                                                                                             
