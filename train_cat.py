from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier,Pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
from make_data import make_data

from net import make_net
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def history_to_predictions_mean(history):
    if len(history) != 1:
        history = np.concatenate(history, axis=1)
    else:
        history = history[0]
    return np.mean(history, axis=1).reshape(-1)


data = pd.read_csv('data_train.csv')
X = data.drop(columns=['ID_code', 'target']).values
y = data['target'].values

data_val = pd.read_csv('data_test.csv')
X_validation = data_val.drop(columns=['ID_code', 'target']).values
y_validation = data_val['target'].values

test_df = pd.read_csv('test.csv')
X_test = test_df.drop(columns=['ID_code']).values

scaler = StandardScaler()

scaler.fit(np.concatenate([X, X_test], axis=0))

X = scaler.transform(X)

X_test = scaler.transform(X_test)

X_validation = scaler.transform(X_validation)

n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

model = CatBoostClassifier(loss_function="Logloss",
                           eval_metric="AUC",
                           task_type="GPU",
                           learning_rate=0.01,
                           iterations=10000,
                           random_seed=42,
                           od_type="Iter",
                           depth=10,
                           early_stopping_rounds=500)

prediction = np.zeros(len(X_test))
predictions_history = []
train_prediction = np.zeros(len(X_validation))
train_predictions_history = []

for idx, (train_index, valid_index) in enumerate(folds.split(X, y)):
    y_train, y_valid = X[train_index], y[valid_index]
    X_train, X_valid = X[train_index], y[valid_index]

    _train = Pool(X_train, label=y_train)
    _valid = Pool(X_valid, label=y_valid)

    print "\nFold ", idx + 1
    fit_model = model.fit(_train, eval_set=_valid, use_best_model=True, verbose=200, plot=True)

    pred = fit_model.predict_proba(X_valid)[:, 1]
    print "  auc = ", roc_auc_score(y_valid, pred)

    pred = model.predict_proba(X_validation)

    print 'ROC AUC CURR', roc_auc_score(y_valid, model.predict(X_valid).reshape(-1))
    print 'ROC AUC FULL', roc_auc_score(y_validation, pred.reshape(-1))
    print 'ROC AUC MEAN', roc_auc_score(y_validation, history_to_predictions_mean(train_predictions_history))

    predictions_history.append(model.predict_proba(X_test))

print 'Submission'
sub1 = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub1["target"] = history_to_predictions_mean(predictions_history)
sub1.to_csv("submission_cat.csv", index=False)