import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def handle_outliers(X):
    """
    Handle outliers in the data X.

    Arguments
    ---------
    - X: df of input features

    Return
    ------
    - X: df of input features
        The data X with outliers handled.
    """

    # Handle outliers in the data X by replacing them with the nearest non outlier value

    columns_handle = ['U10', 'V10', 'U100', 'V100']

    for column in columns_handle:
        data = X[column].values
        # threshold = outside of boxplots 
        threshold = 1.5 * (np.percentile(data, 75) - np.percentile(data, 25))
        # replace outliers with nearest non outlier value
        for i in range(len(data)):
            if data[i] > np.percentile(data, 75) + threshold:
                data[i] = data[i-1]
            elif data[i] < np.percentile(data, 25) - threshold:
                data[i] = data[i-1]
        X[column] = data

    return X

if __name__ == '__main__':

    flatten = True
    N_ZONES = 10
    X_format = 'data/X_Zone_{i}.csv'

    # Read input and output files (1 per zone)
    Xs = []
    for i in range(N_ZONES):
        print('Reading zone {}...'.format(i+1))
        Xs.append(pd.read_csv(X_format.format(i=i+1)))

    Xs_handled = []
    for i in range(N_ZONES):
        print('Handling outliers for zone {}...'.format(i+1))
        Xs_handled.append(handle_outliers(Xs[i]))

    # Save the handled data
    for i in range(N_ZONES):
        print('Saving handled data for zone {}...'.format(i+1))
        Xs_handled[i].to_csv('data/X_Zone_{i}.csv'.format(i=i+1), index=False)

  
