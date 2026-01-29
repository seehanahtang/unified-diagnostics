import pandas as pd 
import numpy as np 
import os 
import sys
import warnings
import random
import json
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, f1_score
from sklearn.utils import class_weight
import xgboost as xgb
import glob
import gc

data_path = "/orcd/pool/003/dbertsim_shared/ukb"
COHORT = sys.argv[1]

def get_sample_map():
    sample_map = pd.read_csv(f"{data_path}/bgen/ch18/c18_b0_v1_samples.csv")
    assert(len(sample_map.loc[sample_map['ID_1'] != sample_map['ID_2']]) == 0)
    cols_to_read = ['eid', 'breast_cancer', 'breast_time_to_diagnosis']
    df_diag = pd.read_csv(f"{data_path}/ukb_cancer_{COHORT}.csv", usecols=cols_to_read) # TRAINING to change
    sample_map = sample_map.rename(columns = {"ID_1": "eid"})
    sample_map = pd.merge(sample_map, df_diag, on = 'eid', how = 'right')
    return sample_map
    
def clean_chrom(path):
    df = pd.read_parquet(path)
    df = df.fillna(-1)
    df = df.loc[df['sample_idx'].isin(sample_map['sample_idx'])]
    df_pivot = df.pivot(index='sample_idx', columns='variant_idx', values='dosage')
    df_pivot = df_pivot.fillna(0)
    df_pivot = df_pivot.add_prefix("c18_").reset_index()
    df_pivot = pd.merge(sample_map, df_pivot, on = 'sample_idx', how = 'inner')
    return df_pivot


df_sis = pd.read_parquet(f"{data_path}/bgen/ch18/c18_sis.parquet")

# Select top columns only 
cols_to_keep = list(df_sis.loc[df_sis['score']>0.03]['feature'])
df_selected = pd.DataFrame(columns=["eid"])
files = sorted(glob.glob(f"{data_path}/bgen/ch18/c18_*parquet"))

for path in files:
    if "sis" in path:
        continue
    df = clean_chrom(path)
    valid_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[valid_cols + ['eid']]
    df_selected = pd.merge(df_selected, df, how = 'outer', on = 'eid')
    del df
    gc.collect()
    
df_selected.to_csv(f'/orcd/pool/003/dbertsim_shared/ukb/bgen/ch18/{COHORT}_c18_selected_features_score0.03.csv', index =False)