import prepare_data
import pickle
import yaml
from PlotFactory import PlotFactory
import os
import matplotlib.pyplot as plt

config = yaml.safe_load(open("../config.yml"))
plotFactory = PlotFactory()
df_train, df_test = prepare_data.split_data()

three_features = ['length','polar_count','hydr_count']
testing_columns = ['length', 'entropy',
       'hydr_count', 'polar_count', 'burried',
       'gravy', 'molecular_weight', 'aromaticity', 'instability_index',
       'isoelectric_point', 'A', 'C', 'D', 'E', 'F',
       'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
       'Y']
length = ['length']
testing_columns_nsp2 = testing_columns + ['rhsa_netsurfp2','thsa_netsurfp2','tasa_netsurfp2','q3_H','q3_E','q3_C']

X_test_tfm = df_test[three_features]
X_test_own = df_test[testing_columns]
X_test_nsp2 = df_test[testing_columns_nsp2]
X_test_length = df_test[length]
y_test = df_test['thsa']

modeltypes = {'GFM':X_test_own, 'NetSurfP2':X_test_nsp2, 'TFM':X_test_tfm, 'Naive':X_test_length}
names =  {'GFM':'GFM', 'NetSurfP2':'Combination Model', 'TFM':'TFM', 'Naive':'Naive Model'}
pred_dict = {'NetSurfP2':df_test['thsa_netsurfp2']}
for modeltype in modeltypes.keys():
    file_name = 'thsa_'+modeltype+'_model.model'
    filename = os.path.join(config['path']['model'], file_name)
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(modeltypes[modeltype])

    # best_features = loaded_model.best_estimator_.feature_importances_
    # plt.figure(figsize=(14,10))
    # plt.barh([x for _,x in sorted(zip(best_features,modeltypes[modeltype].columns))][-10:], sorted(best_features)[-10:])
    # plt.title('THSA feature importance ('+names[modeltype]+')')
    # plt.xticks(fontsize=24)
    # plt.yticks(fontsize=24)
    # plt.tight_layout()
    # plt.show()
    # exit()
    df_new = df_test[['id']]
    df_new['predict'] = result
    df_new.to_csv('../../data/results/'+modeltype+'_result.csv', index=False)
    pred_dict[names[modeltype]] = result

xlab = 'Error threshold (%)'
ylab = 'Correctly predicted (%)'
title = 'Relative error threshold curve for predicting THSA'
plotFactory.plot_curve(y_test, pred_dict, 100, xlab, ylab, title)
