import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import time
from make_data import make_data

from net import make_net
from sklearn.utils import class_weight

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data = pd.read_csv('train.csv')
X = data.drop(columns=['ID_code', 'target']).values
y = data['target'].values

#data_val = pd.read_csv('data_test.csv')
#X_validation = data_val.drop(columns=['ID_code', 'target']).values
#y_validation = data_val['target'].values

test_df = pd.read_csv('test.csv')
X_test = test_df.drop(columns=['ID_code']).values

scaler = StandardScaler()

scaler.fit(np.concatenate([X, X_test], axis=0))

X = scaler.transform(X)

X_test = scaler.transform(X_test)

#X_validation = scaler.transform(X_validation)

class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

scale_pos_weigh = class_weights[1] / class_weights[0]

params = {'num_leaves': 20,
          'min_data_in_leaf': 20,
          'objective': 'binary',
          'max_depth': 35,
          'learning_rate': 0.01,
          'boosting': 'gbdt',
          'reg_alpha': 1.,
          'reg_lambda': 1.,
          'random_state': 42,
          'metric': 'auc',
          'verbosity': -10,
          'num_threads': 16}

prediction = np.zeros(len(X_test))
prediction_val = np.zeros(len(X))

for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
    print 'Fold', fold_n + 1, 'started at', time.ctime()

    ld = 200
    net_model, local_model = make_net(ld, 5e-4)

    net_model.load_weights('model_' + str(fold_n))

    lX = X#make_data(local_model, X)

    lX_test = X_test#make_data(local_model, X_test)

    X_train, X_valid = lX[train_index], lX[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    model = lgb.train(params, train_data, num_boost_round=100000,
                      valid_sets=[train_data, valid_data], verbose_eval=300, early_stopping_rounds=600)

    pred = model.predict(lX, num_iteration=model.best_iteration)
    prediction_val += pred

    print 'ROC AUC CURR', roc_auc_score(y, pred)
    print 'ROC AUC MEAN', roc_auc_score(y, prediction_val / (1 + fold_n))

    # y_pred_valid = model.predict(X_valid)
    prediction += model.predict(lX_test, num_iteration=model.best_iteration) / float(n_fold)


print 'Submission'
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub["target"] = prediction
sub.to_csv("submission_xgb.csv", index=False)