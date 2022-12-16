import os
import pandas as pd
import numpy as np
import pprint

# load ytrueall in a list
ytrueall = []
for i in range(1, 11):
    temp = pd.read_csv('ytrueall/Y_true_Zone_{}.csv'.format(i))
    ytrueall.append(temp['TARGETVAR'].values)
    

# list of subfolder of folder submission
subfolders = [f.path for f in os.scandir('submissions') if f.is_dir()]

# list of methods
methods = [subfolder.split('/')[-1] for subfolder in subfolders]

print('methods : ' + str(methods) + '\n')

ypredall = {}
for method in methods:
    ypredall[method] = []
    for i in range(1, 11):
        temp = pd.read_csv('submissions/{}/Y_pred_Zone_{}.csv'.format(method, i))
        ypredall[method].append(temp['TARGETVAR'].values)

def MSE(ytrue, ypred):
    return np.mean((ytrue - ypred) ** 2)

def MAE(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))

results = {}
for method in methods:
    results[method] = {}
    results[method]['MSE'] = []
    results[method]['MAE'] = []
    for i in range(10):
        results[method]['MSE'].append(MSE(ytrueall[i], ypredall[method][i]))
        results[method]['MAE'].append(MAE(ytrueall[i], ypredall[method][i]))
    results[method]['MSE'] = np.mean(results[method]['MSE'])
    results[method]['MAE'] = np.mean(results[method]['MAE'])

results = pd.DataFrame(results)
results.to_csv('results.csv', index=1)
pprint.pprint(results)

# print best MAE and MSE methods
print('\nBest MAE method : ' + str(results.idxmin(axis=1)['MAE']))
print('\nBest MSE method : ' + str(results.idxmin(axis=1)['MSE']))