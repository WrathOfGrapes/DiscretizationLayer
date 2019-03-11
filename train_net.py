from __future__ import print_function


def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import  StandardScaler
from net import make_net, IntervalEvaluation
import experiment_utils
import visualization
import json
from deep_dict import DeepDict

from keras import backend as K
import time
from pprint import pprint


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


def plot_everything(model, configs, experiment_folder, prefix=''):
    visualization.plot_bin_vertical(model, [0, 30, 60, 90],
                                    target_path=os.path.join(experiment_folder, 'pics', 'bins_vertical.png'))

    path_prefixs = [os.path.join(experiment_folder, 'pics', prefix + name, prefix + name) for name in
                    "horizontal_bins horizontal_fn vertical_fn".split()]
    [os.makedirs(os.path.split(path)[0]) for path in path_prefixs]
    visualization.plot_all_bins_model(model, X_train, feature_list=list(range(10)),
                                      target_path_prefix=path_prefixs[0])
    visualization.plot_presoftmax(model, X, in_feature_inds=range(0, 100, 10), out_feature_inds=range(0, 100, 10),
                                  disc_layer_configs=configs['disc_layer'],
                                  axis='horizontal',
                                  target_path_prefix=path_prefixs[1])
    visualization.plot_presoftmax(model, X, in_feature_inds=range(0, 100, 10), out_feature_inds=range(0, 100, 10),
                                  disc_layer_configs=configs['disc_layer'],
                                  axis='vertical',
                                  target_path_prefix=path_prefixs[2])


args = experiment_utils.parse_args()
n_fold = args.fold
config = args.config
name = args.name
submission = args.submission
validation = args.validation
lr = args.lr
dimension = args.dimension
verbose = not args.silent

configs = DeepDict({'name': None,
                    'disc_layer': {
                        'bins_init': 'linspace',
                        'bins_init_range': 3,
                        'pre_sm_dropout': 0.0,
                        'softmax': 'softmax',
                        'bias_init': 0.0,
                        'pre_sm_activation': 'lelu',
                    }})

configs_update = json.load(open(config, 'r')) if config is not None else {}

configs.merge(configs_update)

if verbose:
    print('Final config file:')
    pprint(configs)

# Choosing name. Priorities: name from command line > name in configs > name of the config file
experiment_name = name or configs['name'] or (
    os.path.split(config)[-1].split('.')[0] if config else None)

folder_name = experiment_utils.create_experiment_folder(experiment_name)
folder_path = os.path.join('runs', folder_name)

if verbose:
    print('Loading data')
data = pd.read_csv('./data/train.csv')
X = data.drop(columns=['ID_code', 'target']).values
y = data['target'].values

test_df = pd.read_csv('./data/test.csv')
test = test_df.drop(columns=['ID_code']).values

scaler = StandardScaler()
scaler.fit(np.concatenate([X, test], axis=0))

test = scaler.transform(test)
X = scaler.transform(X)

if validation:
    X, X_validation, y, y_validation = train_test_split(X, y, train_size=0.875, random_state=13)
else:
    X_validation = None
    y_validation = None

batch_size = 1024


def train_model(train, test, lr, ld, validation=None, fold_number=0, plot=False, verbose=False):
    print('Fold', fold_number + 1, 'started at', time.ctime())
    K.clear_session()

    X_train, y_train = train
    X_test, y_test = test

    X_validation, y_validation = validation
    should_validate = y_validation is not None and X_validation is not None

    model, local_model = make_net(ld, lr, configs=configs)

    if verbose:
        print_list = ['TR {:6.3f}%'.format(100 * np.mean(y_train)),
                      'TS {:6.3f}%'.format(100 * np.mean(y_test))]
        if should_validate:
            print_list.append('VAL {:6.3f}%'.format(100 * np.mean(y_validation)))
        print(' '.join(print_list))

    if plot:
        plot_everything(model, configs=configs, experiment_folder=folder_path, prefix='init_')

    checkpoint_path = os.path.join(folder_path, 'model_' + str(fold_number))
    if verbose:
        print('Checkpoint at', checkpoint_path)

    callbacks = [
        ReduceLROnPlateau(monitor='val_auroc', factor=0.5, patience=4, min_lr=5e-5, verbose=1 if verbose else 0,
                          mode='max'),
        ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True,
                        mode='max'),
        EarlyStopping(patience=15, monitor='val_auroc', mode='max')]

    if verbose:
        if should_validate:
            callbacks.append(IntervalEvaluation(validation_data=(X_validation, y_validation), interval=1))
        else:
            callbacks.append(IntervalEvaluation(validation_data=(X_test, y_test), interval=1))

    try:
        model.fit(X_train, y_train, batch_size=batch_size, epochs=200, shuffle=True, validation_data=(X_test, y_test),
                  callbacks=callbacks, verbose=1 if verbose else 2)
    except KeyboardInterrupt:
        print('')

    model.load_weights(checkpoint_path)

    if plot:
        plot_everything(model, configs=configs, experiment_folder=folder_path, prefix='')

    return model


train_predictions_history = []
test_predictions_history = []
validation_predictions_history = []

if verbose:
    print('Training')

if n_fold > 1:
    iterator = enumerate(StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42).split(X, y))
else:
    ind_train, ind_test = train_test_split(list(range(len(X))), test_size=25000, random_state=23)
    iterator = [(0, (ind_train, ind_test))]

for fold_n, (train_index, test_index) in iterator:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = train_model((X_train, y_train),
                        (X_test, y_test),
                        lr,
                        dimension,
                        validation=(X_validation, y_validation),
                        fold_number=fold_n,
                        verbose=verbose)
    train_predictions_history.append(model.predict(X))
    if verbose:
        print('TOTAL ROC AUC', roc_auc_score(y, history_to_predictions_mean(train_predictions_history)))
        print('TRAIN ROC AUC', roc_auc_score(y_train, model.predict(X_train).reshape(-1)))
        print('TEST  ROC AUC', roc_auc_score(y_test, model.predict(X_test).reshape(-1)))
    if validation:
        validation_predictions_history.append(model.predict(X_validation))
        print('VAL ROC AUC', roc_auc_score(y_validation, history_to_predictions_mean(validation_predictions_history)))
    if submission:
        test_predictions_history.append(model.predict(test))

if submission:
    print('Submission')
    sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
    sub["target"] = history_to_predictions_mean(test_predictions_history)
    sub.to_csv(os.path.join(folder_path, "submission_net.csv"), index=False)
