import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


data = pd.read_csv('runs/err/predictions.csv')
train_data = pd.read_csv('./data/train.csv')

X, X_validation, y, y_validation = train_test_split(range(len(data)), range(len(data)),
                                                    train_size=0.875, random_state=13)

data = data.iloc[X]

y_true = data['target'].values
y_pred = data['prediction'].values
names = data['ID_code']

pos = y_pred[y_true == 1]
neg = y_pred[y_true == 0]

pos_names = names[y_true == 1]
neg_names = names[y_true == 0]

pos_errors = np.zeros_like(pos)
for i in tqdm(range(len(pos))):
    pos_errors[i] = float(len(neg[neg >= pos[i]])) / len(neg)

neg_errors = np.zeros_like(neg)
for i in tqdm(range(len(neg))):
    neg_errors[i] = float(len(pos[pos <= neg[i]])) / len(pos)

pos_errors = np.asarray(pos_errors)
neg_errors = np.asarray(neg_errors)
print np.mean(pos_errors), np.std(pos_errors)

for i in range(0, 101, 1):
    print i, np.percentile(pos_errors, i), np.percentile(neg_errors, i)

pos_higher_percentile = pos_errors < np.percentile(pos_errors, 98)
neg_higher_percentile = neg_errors < np.percentile(neg_errors, 98)

pos_names = pos_names[pos_higher_percentile]
neg_names = neg_names[neg_higher_percentile]

print pos_names.shape, neg_names.shape

target_names = pd.concat([pos_names, neg_names])

print target_names.shape
cleared = train_data.loc[train_data['ID_code'].isin(target_names)]
print cleared.shape

cleared.to_csv('./data/cleared_bce.csv', index=False)