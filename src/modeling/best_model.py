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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd

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
        clf = GridSearchCV(methods[method][0], methods[method][1], cv=5)
        clf.fit(X_train, y_train)
        rsquared = clf.score(X_test, y_test)

        if rsquared > highest_r:
            highest_r = rsquared
            best_model = clf

    # best_features = best_model.best_estimator_.feature_importances_
    # plt.bar(X.columns, best_features)
    # plt.show()
    # print('*'*10)
    # print(best_model)
    # print(highest_r)
    return best_model.fit(X,y)

config = yaml.safe_load(open("../config.yml"))

svr = SVR(kernel='rbf')
svr_params = {"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}

xgbbooster = xgboost.XGBRegressor(objective='reg:squarederror')
xgbbooster_params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.5, 1.5, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [2, 3, 4]
        }

randfor = RandomForestRegressor()
randfor_params = {'max_depth':[2,3,4]}

destree = DecisionTreeRegressor()
destree_params = {'max_depth':[2,3,4]}

linreg = LinearRegression()
linreg_params = {}

methods = {
'xgboost':[xgbbooster, xgbbooster_params],
'svr':[svr, svr_params],
'destree':[destree, destree_params],
'randfor':[randfor,randfor_params],
}

df_train = pd.read_csv(config['path']['processed_data']+'ready_to_use_data_train.csv')
df_test = pd.read_csv(config['path']['processed_data']+'ready_to_use_data_test.csv')

three_features = ['length','polar_count','hydr_count']
training_columns = [ 'length', 'entropy',
       'hydr_count', 'polar_count', 'burried',
       'gravy', 'molecular_weight', 'aromaticity', 'instability_index',
       'isoelectric_point', 'A', 'C', 'D', 'E', 'F',
       'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
       'Y']
length = ['length']
nsp2_columns = ['rhsa_netsurfp2','thsa_netsurfp2','tasa_netsurfp2']
training_columns_nsp2 = training_columns + nsp2_columns

X_train_tfm = df_train[three_features]
X_train_own = df_train[training_columns]
X_train_nsp2 = df_train[training_columns_nsp2]
X_train_nsp2m = df_train[nsp2_columns]
X_train_length = df_train[length]
y_thsa, y_rhsa, y_lhpsa = df_train['thsa'], df_train['rhsa'], df_train['size']

pickle.dump(get_best_model(X_train_own, y_thsa, methods), open(config['path']['model']+'thsa_GFM_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_nsp2, y_thsa, methods), open(config['path']['model']+'thsa_NetSurfP2_combination_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_tfm, y_thsa, methods), open(config['path']['model']+'thsa_TFM_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_length, y_thsa, {}), open(config['path']['model']+'thsa_Naive_model.model', 'wb'))

# pickle.dump(get_best_model(X_train_own, y_rhsa, methods), open(config['path']['model']+'rhsa_GFM_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_nsp2, y_rhsa, methods), open(config['path']['model']+'rhsa_NetSurfP2_combination_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_tfm, y_rhsa, methods), open(config['path']['model']+'rhsa_TFM_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_length, y_rhsa, methods), open(config['path']['model']+'rhsa_Naive_model.model', 'wb'))

# pickle.dump(get_best_model(X_train_own, y_lhpsa, methods), open(config['path']['model']+'patch_GFM_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_nsp2, y_lhpsa, methods), open(config['path']['model']+'patch_NetSurfP2_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_nsp2m, y_lhpsa, methods), open(config['path']['model']+'patch_nsp2m_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_tfm, y_lhpsa, methods), open(config['path']['model']+'patch_TFM_model.model', 'wb'))
# pickle.dump(get_best_model(X_train_length, y_lhpsa, methods_naive), open(config['path']['model']+'patch_Naive_model.model', 'wb'))
