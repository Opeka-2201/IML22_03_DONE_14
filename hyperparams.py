import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV



def split_train_test(X, Y):
    X_train = X[Y['TARGETVAR'] != -1]
    Y_train = Y[Y['TARGETVAR'] != -1]
    X_test = X[Y['TARGETVAR'] == -1]
    return X_train, X_test, Y_train

def handle_outliers(df):
    return df
      
if __name__ == '__main__':
    flatten = True
    N_ZONES = 10
    X_format = 'data/X_Zone_{i}.csv'
    Y_format = 'data/Y_Zone_{i}.csv'

    os.makedirs('best_params', exist_ok=True)
    for i in range(1, N_ZONES+1):
        os.makedirs('best_params/zone_{}'.format(i), exist_ok=True)
        
    Xs, Ys = [], []
    for i in range(1, N_ZONES+1):
        print('Reading zone {}...'.format(i))
        Xs.append(pd.read_csv(X_format.format(i=i)))
        Ys.append(pd.read_csv(Y_format.format(i=i)))

        # Flatten temporal dimension (NOTE: this step is not compulsory)
        if flatten:
            X_train, X_test, Y_train = split_train_test(Xs[i-1], Ys[i-1])
            Xs[i-1] = (X_train, X_test)
            Ys[i-1] = Y_train

    print("\033[92m \nData loaded successfully\033[00m")

    methods = ['lregression', 'ridge', 'dtr', 'knn']

    for i in range(1, N_ZONES+1):
      print('\nFitting hyperparameters for zone', i)
      X_train, Y_train = Xs[i-1][0], Ys[i-1]
      for method in methods:
        print('    Method :', method)

        if method == 'lregression':
            f = open('best_params/zone_{i}/lregression.txt'.format(i=i), 'w')
            f.write('None')
            f.close()

        elif method == 'ridge':
            parameters = {'alpha': [0.1, 1, 10, 100, 1000]}
            clf = GridSearchCV(Ridge(), parameters, cv=5)
            clf.fit(X_train, Y_train)
        
            f = open('best_params/zone_{i}/ridge.txt'.format(i=i), 'w')
            f.write(str(clf.best_params_['alpha']))
            f.close()

        # elif method == 'rforest':
        #     parameters = {'n_estimators': [10, 100, 1000], 'max_depth': [10, 100, 1000], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        #     clf = GridSearchCV(RandomForestRegressor(), parameters, cv=5)
        #     clf.fit(X_train, Y_train)
            
        #     f = open('best_params/zone_{i}/rforest.txt'.format(i=i), 'w')
        #     f.write(str(clf.best_params_['n_estimators']) + ' ' + str(clf.best_params_['max_depth']) + ' ' + str(clf.best_params_['min_samples_split']) + ' ' + str(clf.best_params_['min_samples_leaf']))
        #     f.close()

        elif method == 'dtr':
            parameters = {'max_depth': [10, 100, 1000], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
            clf = GridSearchCV(DecisionTreeRegressor(), parameters, cv=5)
            clf.fit(X_train, Y_train)
            
            f = open('best_params/zone_{i}/dtr.txt'.format(i=i), 'w')
            f.write(str(clf.best_params_['max_depth']) + ' ' + str(clf.best_params_['min_samples_split']) + ' ' + str(clf.best_params_['min_samples_leaf']))
            f.close()

        elif method == 'knn':
            parameters = {'n_neighbors': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
            clf = GridSearchCV(KNeighborsRegressor(), parameters, cv=5)
            clf.fit(X_train, Y_train)
            
            f = open('best_params/zone_{i}/knn.txt'.format(i=i), 'w')
            f.write(str(clf.best_params_['n_neighbors']) + ' ' + str(clf.best_params_['weights']) + ' ' + str(clf.best_params_['algorithm']))
            f.close()
