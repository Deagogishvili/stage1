import pandas as pd
from functools import reduce
import numpy as np
import os
import glob
import yaml

config = yaml.safe_load(open("../config.yml"))

# hydrophobic_proteins =  ['F','W','I','L','M','V','C','A']
hydrophobic_proteins = config['hydrophobic']
def get_working_data():
    df_DSSP = pd.read_csv(config['path']['processed_data']+'dssp_data.csv')
    df_TMHMM = pd.read_csv(config['path']['processed_data']+'tmhmm_data.csv')
    df_fasta = pd.read_csv(config['path']['processed_data']+'fasta_data.csv')
    df_global_seq_features = pd.read_csv(config['path']['processed_data']+'global_seq_features_data.csv')

    data_frames = [df_DSSP, df_TMHMM, df_global_seq_features, df_fasta]

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['id'],
                                                how='inner'), data_frames)
    # df_merged = df_merged[df_merged['perc_undefined_count'] < .03]
    return df_merged

def get_working_filtered_data():
    df = get_working_data()
    df_no_tmp = df[~df['tmp']]
    return df_no_tmp

def split_data():
    train_file = '/home/jan/Documents/BioInformatics/stage/HSAtool/files/CSV/train/train.csv'
    test_file = '/home/jan/Documents/BioInformatics/stage/HSAtool/files/CSV/test/test.csv'

    train_old = pd.read_csv(train_file)
    test_old = pd.read_csv(test_file)
    df = get_working_filtered_data()
    df = df.merge(get_netsurfp2(config['path']['netsurfp2']), on='id', how='inner')

    train = df[df['id'].isin(train_old['id'])]
    test = df[df['id'].isin(test_old['id'])]

    return train, test

def get_training_columns():
    df_global_seq_features = pd.read_csv(config['path']['processed_data']+'global_seq_features_data.csv')
    return df_global_seq_features.columns[:-1]

def get_netsurfp2(path):
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    df = pd.concat([pd.read_csv(x) for x in result])
    df['seq_hydr'] = df['seq'].isin(hydrophobic_proteins)
    df2 = df[df['seq_hydr'] == True].groupby(['id']).sum()['asa'].reset_index()

    df3 = df2.merge(df.groupby(['id']).sum()['asa'].reset_index(), on='id', how='inner')
    df4 = df3.merge(df.groupby(['id']).mean()[['p[q3_H]','p[q3_E]','p[q3_C]']].reset_index(), on='id', how='inner')
    df4 = df4.rename(columns={
                            'asa_x':'thsa_netsurfp2',
                            'asa_y':'tasa_netsurfp2',
                            'p[q3_H]':'q3_H',
                            'p[q3_E]':'q3_E',
                            'p[q3_C]':'q3_C'
                            })
    df4['rhsa_netsurfp2'] = df4['thsa_netsurfp2']/df4['tasa_netsurfp2']
    return df4 [['id','thsa_netsurfp2', 'rhsa_netsurfp2', 'tasa_netsurfp2','q3_H','q3_E','q3_C']]
