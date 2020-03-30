import yaml
import pandas as pd
from functools import reduce


config = yaml.safe_load(open("../config.yml"))

hydrophobic_proteins = config['hydrophobic']

def get_data():
    df_DSSP = pd.read_csv(config['path']['processed_data']+'dssp_data.csv')
    df_TMHMM = pd.read_csv(config['path']['processed_data']+'tmhmm_data.csv')
    df_fasta = pd.read_csv(config['path']['processed_data']+'fasta_data.csv')
    df_nsp2 = pd.read_csv(config['path']['processed_data']+'netsurp2_data.csv')
    df_monomer = pd.read_csv(config['path']['processed_data']+'monomer_data.csv')
    df_global_seq_features = pd.read_csv(config['path']['processed_data']+'global_seq_features_data.csv')

    data_frames = [df_DSSP, df_TMHMM, df_global_seq_features, df_fasta, df_nsp2]

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['id'],
                                                how='inner'), data_frames)

    df_no_tmp = df_merged[~df_merged['tmp']]
    return df_no_tmp

print(get_data())
