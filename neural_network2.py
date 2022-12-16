import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

import pandas as pd


def split_train_test(X, Y):
    X_train = X[Y['TARGETVAR'] != -1]
    Y_train = Y[Y['TARGETVAR'] != -1]
    X_test = X[Y['TARGETVAR'] == -1]
    return X_train, X_test, Y_train


if __name__ == '__main__':
    flatten = True
    N_ZONES = 10
    X_format = 'data/X_Zone_{i}.csv'
    Y_format = 'data/Y_Zone_{i}.csv'

    Xs, Ys = [], []
    for i in range(1, N_ZONES+1):
        print('Reading zone {}...'.format(i))

        Xsi = pd.read_csv(X_format.format(i=i))
        Ysi = pd.read_csv(Y_format.format(i=i))

        Xs.append(Xsi[['U10', 'V10', 'U100', 'V100',
                  'Day', 'Month', 'Year', 'Hour']])
        Ys.append(Ysi[['TARGETVAR']])

        if flatten:
            X_train, X_test, Y_train = split_train_test(Xs[i-1], Ys[i-1])
            Xs[i-1] = (X_train, X_test)
            Ys[i-1] = Y_train

    print("\033[92m \nData loaded successfully\n \033[00m")

    model = nn.Sequential(
        nn.Linear(8, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
    )

    for i in range(1, N_ZONES+1):
        print('Training zone {}...'.format(i))

        tensor_x_train = torch.Tensor(Xs[i-1][0].values)
        tensor_y_train = torch.Tensor(Ys[i-1].values)

        tensor_x_test = torch.Tensor(Xs[i-1][1].values)

        # Define the loss function
        loss_fn = torch.nn.MSELoss(reduction='sum')

        # Define the optimizer
        learning_rate = 1e-2
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        for t in range(1000):
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(tensor_x_train)

            loss = loss_fn(y_pred, tensor_y_train)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save the model
        os.makedirs('models2', exist_ok=True)
        torch.save(model, 'models2/model_{}.pt'.format(i))

        # Test the model
        y_pred = model(tensor_x_test)

        # Save the predictions
        os.makedirs('submissions/nn2', exist_ok=True)

        # save results in a csv file with name of column TARGETVAR
        pd.DataFrame(y_pred.detach().numpy(), columns=['TARGETVAR']).to_csv(
            'submissions/nn2/Y_pred_Zone_{}.csv'.format(i), index=False)

    print("\033[92m \nAll zones trained successfully\033[00m")
