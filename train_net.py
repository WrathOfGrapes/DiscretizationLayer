import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from net import make_net, IntervalEvaluation, make_net2, make_net3
import experiment_utils
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import class_weight
import visualization
import json

from keras import backend as K
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def history_to_predictions(history):
    result = []
    if len(history) != 1:
        history = np.concatenate(history, axis=1)
    else:
        history = history[0]
    for i in range(history.shape[0]):
        mean = np.mean(history[i])
        if mean < 0.5:
            result.append(min(history[i]))
        else:
            result.append(max(history[i]))
    return np.asarray(result).reshape(-1)


def history_to_predictions_mean(history):
    if len(history) != 1:
        history = np.concatenate(history, axis=1)
    else:
        history = history[0]
    return np.mean(history, axis=1).reshape(-1)


args = experiment_utils.parse_args()

DEFAULT_DICT = {'disc_layer': {'bins_init': 'uniform'}}

configs = json.load(open(args.configs, 'r')) if args.configs is not None else DEFAULT_DICT


folder_name = experiment_utils.create_experiment_folder(args.name)
folder_path = os.path.join('runs', folder_name)

n_fold = 5 if args.prod else 1

if n_fold == 1:
    data = pd.read_csv('./data/data_train.csv')
else:
    data = pd.read_csv('./data/train.csv')
X = data.drop(columns=['ID_code', 'target']).values
y = data['target'].values

data_val = pd.read_csv('./data/data_test.csv')
X_validation = data_val.drop(columns=['ID_code', 'target']).values
y_validation = data_val['target'].values

test_df = pd.read_csv('./data/test.csv')
test = test_df.drop(columns=['ID_code']).values

scaler = StandardScaler()
scaler.fit(np.concatenate([X, test], axis=0))

test = scaler.transform(test)
X = scaler.transform(X)
X_validation = scaler.transform(X_validation)

ld = 100

if n_fold > 1:
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

prediction = np.zeros(len(test))
predictions_history = []
train_prediction = np.zeros(len(X_validation))
train_predictions_history = []

batch_size = 1024


def train_model(X_train, y_train, X_valid, y_valid, fold_number):

    print('Fold', fold_number + 1, 'started at', time.ctime())
    K.clear_session()

    model, local_model = make_net(ld, 1e-3, configs=configs)

    visualization.plot_all_bins_model(model, X_train, feature_list=list(range(10)), target_path_prefix=os.path.join(folder_path, 'pics/start_%d' % fold_number))
    print(model.summary())
    print('Train mean', np.mean(y_train), 'Test mean', np.mean(y_valid))


    checkpoint_path = os.path.join(folder_path, 'model_' + str(fold_number))

    callbacks = callbacks = [
        ReduceLROnPlateau(monitor='val_auroc', factor=0.5, patience=5, min_lr=1e-6, verbose=1, mode='max'),
        ModelCheckpoint(checkpoint_path, monitor='val_auroc', verbose=1, save_best_only=True, mode='max'),
        EarlyStopping(patience=10, monitor='val_auroc', mode='max')]

    json.dump(configs, open(os.path.join(folder_path, 'configs.json'), 'w'))

    model.fit(X_train, y_train, batch_size=batch_size, epochs=100, shuffle=True, validation_data=(X_valid, y_valid),
              callbacks=callbacks)

    model.load_weights(checkpoint_path)
    visualization.plot_all_bins_model(model, X_train, feature_list=list(range(10)), target_path_prefix=os.path.join(folder_path, 'pics/trained_%d' % fold_number))
    

    pred = model.predict(X_validation)
    train_predictions_history.append(pred)

    print('ROC AUC CURR', roc_auc_score(y_valid, model.predict(X_valid).reshape(-1)))
    print('ROC AUC FULL', roc_auc_score(y_validation, pred.reshape(-1)))
    print('ROC AUC MAX ', roc_auc_score(y_validation, history_to_predictions(train_predictions_history)))
    print('ROC AUC MEAN', roc_auc_score(y_validation, history_to_predictions_mean(train_predictions_history)))

    predictions_history.append(model.predict(test))


if n_fold > 1:
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        train_model(X_train, y_train, X_valid, y_valid, fold_n)
else:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=27)
    train_model(X_train, y_train, X_valid, y_valid, 0)

print('Submission')
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub["target"] = history_to_predictions_mean(predictions_history)
sub.to_csv(os.path.join(folder_path, "submission_net.csv"), index=False)

with open('./results.tsv', 'a') as f:
    f.write('%s\t%f\n' % (folder_name, roc_auc_score(y_validation, history_to_predictions_mean(train_predictions_history))))
