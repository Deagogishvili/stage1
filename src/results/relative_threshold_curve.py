import yaml
from PlotFactory import PlotFactory
import matplotlib.pyplot as plt
import pandas as pd

config = yaml.safe_load(open("../config.yml"))
plotFactory = PlotFactory()

df_test = pd.read_csv(config['path']['processed_data']+'ready_to_use_data_test.csv')

names =  {'gfm':'GFM', 'nsp2_gfm':'Combination Model', 'tfm':'TFM', 'length':'Naive Model'}
pred_dict = {'NetSurfP2':df_test['thsa_netsurfp2']}
for name in names:
    if name == 'tfm':
        result = pd.read_csv('../../data/predictions/thsa_tfm_prediction.csv')['x']
    else:
        result = pd.read_csv('../../data/predictions/thsa_'+name+'_prediction.csv')['prediction']

    pred_dict[names[name]] = result

xlab = 'Error threshold (%)'
ylab = 'Correctly predicted (%)'
title = 'Relative error threshold curve for predicting THSA'
plotFactory.plot_curve(df_test['thsa'], pred_dict, 100, xlab, ylab, title)


names =  {'gfm':'GFM', 'nsp2_gfm':'Combination Model', 'tfm':'TFM', 'length':'Naive Model'}
pred_dict = {'NetSurfP2':df_test['rhsa_netsurfp2']}
for name in names.keys():
    if name == 'tfm':
        result = pd.read_csv('../../data/predictions/rhsa_tfm_prediction.csv')['x']
    else:
        result = pd.read_csv('../../data/predictions/rhsa_'+name+'_prediction.csv')['prediction']
    pred_dict[names[name]] = result

xlab = 'Error threshold (%)'
ylab = 'Correctly predicted (%)'
title = 'Relative error threshold curve for predicting RHSA'
plotFactory.plot_perc_curve(df_test['rhsa'], pred_dict, 100, xlab, ylab, title)

names =  {'nsp2':'NetSurfP2 trained model', 'gfm':'GFM', 'nsp2_gfm':'Combination Model', 'tfm':'TFM', 'length':'Naive Model'}
pred_dict = {}
for name in names:
    if name == 'tfm':
        result = pd.read_csv('../../data/predictions/lhpsa_tfm_prediction.csv')['x']
    else:
        result = pd.read_csv('../../data/predictions/lhpsa_'+name+'_prediction.csv')['prediction']

    pred_dict[names[name]] = result

xlab = 'Error threshold (%)'
ylab = 'Correctly predicted (%)'
title = 'Relative error threshold curve for predicting LHPSA'
plotFactory.plot_curve(df_test['size'], pred_dict, 100, xlab, ylab, title)
