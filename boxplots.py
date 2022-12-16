import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


if __name__ == '__main__':

    flatten = True
    N_ZONES = 10
    X_format = 'data/X_Zone_{i}.csv'
    Y_format = 'data/Y_Zone_{i}.csv'

    os.makedirs('submissions', exist_ok=True)
    os.makedirs('figs', exist_ok=True)

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

    # plot all boxplots for all zones X only for the wind u10
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(N_ZONES):
        Xs[i][0]['U10'].plot.box(ax=axes[i//5, i % 5])
        axes[i//5, i % 5].set_title('Zone {}'.format(i+1))
    fig.savefig('figs/boxplots_X_u10.png')

    # plot all boxplots for all zones X only for the wind v10
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(N_ZONES):
        Xs[i][0]['V10'].plot.box(ax=axes[i//5, i % 5])
        axes[i//5, i % 5].set_title('Zone {}'.format(i+1))
    fig.savefig('figs/boxplots_X_v10.png')

    # plot all boxplots for all zones X only for the wind u100
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(N_ZONES):
        Xs[i][0]['U100'].plot.box(ax=axes[i//5, i % 5])
        axes[i//5, i % 5].set_title('Zone {}'.format(i+1))
    fig.savefig('figs/boxplots_X_u100.png')

    # plot all boxplots for all zones X only for the wind v100
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(N_ZONES):
        Xs[i][0]['V100'].plot.box(ax=axes[i//5, i % 5])
        axes[i//5, i % 5].set_title('Zone {}'.format(i+1))
    fig.savefig('figs/boxplots_X_v100.png')

    # plot all boxplots for all zones Y (only for training data) only for targetvar
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(N_ZONES):
        Ys[i]['TARGETVAR'].plot.box(ax=axes[i//5, i % 5])
        axes[i//5, i % 5].set_title('Zone {}'.format(i+1))
    fig.savefig('figs/boxplots_Y.png')