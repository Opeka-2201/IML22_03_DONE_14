import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


def split_train_test(X, Y):
    """
    Split the input dataset (X, Y) into training and testing sets, where
    training data corresponds to true (x, y) pairs and testing data corresponds
    to (x, -1) pairs.

    Arguments
    ---------
    - X: df of input features
        The input samples.
    - Y: df of output values
        The corresponding output values.

    Return
    ------
    - X_train: df of input features
        The training set of input samples.
    - Y_train: df of output values
        The training set of corresponding output values.
    - X_test: df of input features
        The testing set of input samples. The corresponding output values are
        the ones you should predict.
    """

    X_train = X[Y['TARGETVAR'] != -1]
    Y_train = Y[Y['TARGETVAR'] != -1]
    X_test = X[Y['TARGETVAR'] == -1]
    return X_train, X_test, Y_train


def method_selection(method, hyperparams):
    """
    Select the best model for the given method.

    Arguments
    ---------
    - method: str
        The method to use for the prediction.
    - hyperparams: list of str
        The hyperparameters of the method.

    Return
    ------
    - model: sklearn model
        The best model for the given method.
    """

    if method == 'lregression':
        model = LinearRegression()
    elif method == 'rforest':
        model = RandomForestRegressor(n_estimators=int(
            hyperparams[0]), max_depth=int(hyperparams[1]), min_samples_split=int(hyperparams[2]), min_samples_leaf=int(hyperparams[3]), random_state=0)
    elif method == 'dtr':
        model = DecisionTreeRegressor(max_depth=int(hyperparams[0]))
    elif method == 'ridge':
        model = Ridge(alpha=float(hyperparams[0]))
    elif method == 'knn':
        model = KNeighborsRegressor(n_neighbors=int(hyperparams[0]))
    else:
        print('Unknonw method.')
        sys.exit(1)

    return model


if __name__ == '__main__':

    flatten = True
    N_ZONES = 10
    X_format = 'data/X_Zone_{i}.csv'
    Y_format = 'data/Y_Zone_{i}.csv'

    os.makedirs('submissions', exist_ok=True)

    # Read input and output files (1 per zone)
    Xs, Ys = [], []
    for i in range(N_ZONES):
        print('Reading zone {}...'.format(i+1))
        Xs.append(pd.read_csv(X_format.format(i=i+1)))
        Ys.append(pd.read_csv(Y_format.format(i=i+1)))

        # Flatten temporal dimension (NOTE: this step is not compulsory)
        if flatten:
            X_train, X_test, Y_train = split_train_test(Xs[i], Ys[i])
            Xs[i] = (X_train, X_test)
            Ys[i] = Y_train

    print("\033[92m \nData loaded successfully\033[00m")

    method = sys.argv[1]
    print('\nUsing method {} :'.format(method))

    os.makedirs('submissions/{}'.format(method), exist_ok=True)

    models = []

    for i in range(N_ZONES):
        print('   Fitting model for zone {}...'.format(i+1))
        
        filename = 'best_params/zone_{}/{}.txt'.format(i+1, method)
        file = open(filename, 'r')
        hyperparams = file.read().split()
        file.close()
        
        model = method_selection(method, hyperparams)
        model.fit(Xs[i][0], Ys[i]['TARGETVAR'])
        models.append(model)

    # predict test series
    Y_pred = []
    for i in range(N_ZONES):
        print('   Predicting test series for zone {}...'.format(i+1))
        Y_i_pred = pd.DataFrame()
        Y_i_pred['TARGETVAR'] = models[i].predict(Xs[i][1])
        Y_pred.append(Y_i_pred)

    # Write submission files (1 per zone). The predicted test series must
    # follow the order of X_test.
    for i in range(N_ZONES):
        print('   Writing submission file for zone {}...'.format(i+1))
        Y_pred[i].to_csv('submissions/{}/Y_pred_Zone_{}.csv'.format(method, i+1), index=False)
