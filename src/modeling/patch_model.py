from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import prepare_data
from sklearn.svm import SVR
import numpy as np
import xgboost
import pickle
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

def StandardScale(X_train, X_test):
    scaler = StandardScaler()
    X_train_stand = scaler.fit_transform(X_train)
    X_test_stand = scaler.transform(X_test)
    return X_train_stand, X_test_stand

def get_best_model(X, y, methods):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # X_train,X_test = StandardScale(X_train, X_test)

    highest_r = 0
    best_model = None

    for method in methods:
        clf = GridSearchCV(methods[method][0], methods[method][1])
        clf.fit(X_train, y_train)
        rsquared = clf.score(X_test, y_test)

        if rsquared > highest_r:
            highest_r = rsquared
            best_model = clf

    best_features = best_model.best_estimator_.feature_importances_
    plt.bar(X.columns, best_features)
    plt.show()
    print(highest_r)
    return best_model.fit(X,y)

config = yaml.safe_load(open("../config.yml"))

svr = SVR(kernel='rbf')
svr_params = {"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}

xgbbooster = xgboost.XGBRegressor()
xgbbooster_params = {
        'min_child_weight': [1, 5, 10],
        # 'gamma': [0.5, 1, 1.5, 2, 5],
        # 'subsample': [0.6, 0.8, 1.0],
        # 'colsample_bytree': [0.6, 0.8, 1.0],
        # 'max_depth': [2, 3, 4, 5]
        }

methods = {
'xgboost':[xgbbooster, xgbbooster_params],
# 'svr':[svr, svr_params]
}

methods_naive = {
'linreg':[LinearRegression(), {}],
# 'svr':[svr, svr_params]
}

train_file = '/home/jan/Documents/BioInformatics/stage/HSAtool/files/CSV/train/train.csv'
test_file = '/home/jan/Documents/BioInformatics/stage/HSAtool/files/CSV/test/test.csv'

train_old = pd.read_csv(train_file)
test_old = pd.read_csv(test_file)

df = pd.read_csv('../../data/patches/lp_residue_all.csv')
df2 = pd.read_csv('../../data/processed_data/csv/global_seq_features_data.csv')
df3 = pd.read_csv('../../data/processed_data/csv/netsurp2_data.csv')
df4 = pd.read_csv('../../data/processed_data/csv/tmhmm_data.csv')

df = df3.merge(df[df['rank'] == 1]).merge(df2).merge(df4)
df = df[~   df['tmp']]
df_train = df[df['id'].isin(train_old['id'])]
df_test = df[df['id'].isin(test_old['id'])]

training_columns = [ 'length', 'entropy',
       'hydr_count', 'polar_count', 'burried',
       'gravy', 'molecular_weight', 'aromaticity', 'instability_index',
       'isoelectric_point', 'A', 'C', 'D', 'E', 'F',
       'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
       'Y']
three_features = ['length','polar_count','hydr_count']
length = ['length']
nsp2 = ['rhsa_netsurfp2','thsa_netsurfp2','tasa_netsurfp2']
training_columns_nsp2 = training_columns + nsp2

X_train_tfm = df_train[three_features]
X_train_nsp2m = df_train[nsp2]
X_train_own = df_train[training_columns]
X_train_nsp2 = df_train[training_columns_nsp2]
X_train_length = df_train[length]
y_patch = df_train['size']

get_best_model(X_train_own, y_patch, methods), open(config['path']['model']+'patch_GFM_model.model', 'wb')

# pickle.dump(get_best_model(X_train_length, y_patch, methods_naive), open(config['path']['model']+'patch_Naive_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_own, y_patch, methods), open(config['path']['model']+'patch_GFM_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_nsp2, y_patch, methods), open(config['path']['model']+'patch_NetSurfP2_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_nsp2m, y_patch, methods), open(config['path']['model']+'patch_nsp2m_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_tfm, y_patch, methods), open(config['path']['model']+'patch_TFM_model.model', 'wb'))
