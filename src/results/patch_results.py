import prepare_data
import pickle
import yaml
from PlotFactory import PlotFactory
import os
import matplotlib.pyplot as plt
import pandas as pd

config = yaml.safe_load(open("../config.yml"))
plotFactory = PlotFactory()
train_file = '/home/jan/Documents/BioInformatics/stage/HSAtool/files/CSV/train/train.csv'
test_file = '/home/jan/Documents/BioInformatics/stage/HSAtool/files/CSV/test/test.csv'

train_old = pd.read_csv(train_file)
test_old = pd.read_csv(test_file)

df = pd.read_csv('../../data/patches/lp_residue_all.csv')
df2 = pd.read_csv('../../data/processed_data/csv/global_seq_features_data.csv')
df3 = pd.read_csv('../../data/processed_data/csv/netsurp2_data.csv')

df = df3.merge(df[df['rank'] == 1]).merge(df2)

df_train = df[df['id'].isin(train_old['id'])]
df_test = df[df['id'].isin(test_old['id'])]

testing_columns = [ 'length', 'entropy',
       'hydr_count', 'polar_count', 'burried',
       'gravy', 'molecular_weight', 'aromaticity', 'instability_index',
       'isoelectric_point', 'A', 'C', 'D', 'E', 'F',
       'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
       'Y']
length = ['length']
three_features = ['length','polar_count','hydr_count']
nsp2 = ['rhsa_netsurfp2','thsa_netsurfp2','tasa_netsurfp2']
testing_columns_nsp2 = testing_columns + nsp2

X_test_tfm = df_test[three_features]
X_test_own = df_test[testing_columns]
X_test_nsp2 = df_test[testing_columns_nsp2]
X_test_nsp2m = df_test[nsp2]
X_test_naive = df_test[length]
y_test = df_test['size']

modeltypes = {'GFM':X_test_own, 'NetSurfP2':X_test_nsp2, 'nsp2m':X_test_nsp2m, 'Naive':X_test_naive, 'TFM':X_test_tfm}
names =  {'GFM':'GFM', 'NetSurfP2':'Combination Model', 'nsp2m':'NetSurfP2', 'Naive':'Naive', 'TFM':'TFM'}
pred_dict = {}
for modeltype in modeltypes.keys():
    file_name = 'patch_'+modeltype+'_model.model'
    filename = os.path.join(config['path']['model'], file_name)
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(modeltypes[modeltype])

    # best_features = loaded_model.best_estimator_.feature_importances_
    # plt.figure(figsize=(14,10))
    # plt.barh([x for _,x in sorted(zip(best_features,modeltypes[modeltype].columns))][-10:], sorted(best_features)[-10:])
    # plt.title('LHPSA feature importance ('+names[modeltype]+')')
    # plt.xticks(fontsize=24)
    # plt.yticks(fontsize=24)
    # plt.tight_layout()
    # plt.show()
    # exit()
    pred_dict[names[modeltype]] = result

xlab = 'Error threshold (%)'
ylab = 'Correctly predicted (%)'
title = 'Relative error threshold curve for predicting LHPSA'
plotFactory.plot_curve(y_test, pred_dict, 100, xlab, ylab, title)
