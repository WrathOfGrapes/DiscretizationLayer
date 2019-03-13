from __future__ import print_function


def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import experiment_utils
args = experiment_utils.parse_args()

experiment = args.experiment
submission = args.submission
validation = args.validation
verbose = not args.silent
save = args.save

config_path = os.path.join('./experiments', experiment)
default_path = os.path.split(config_path)
default_path = default_path[:len(default_path) - 1]
default_path = os.path.join(*default_path)

default_configs_path = default_path

try:
    loaded = True
    configuration = experiment_utils.load_configuration(config_path, './experiments')
    print('Global config at', './experiments/', 'applied')
except IOError:
    loaded = False
    configuration = {}
    print('No global default config found, local will be used as main config')

try:
    configuration_local = experiment_utils.load_configuration(config_path, default_configs_path)
    if not loaded:
        configuration = configuration_local
    else:
        configuration = experiment_utils.merge_two_dicts(configuration, configuration_local)

    print('Local config at', default_configs_path, 'applied')
except IOError:
    if loaded:
        print('No local config found')
    else:
        print('Neither local nor global configs are available, make sure you specified all parameters in main config')
        configuration = experiment_utils.load_json(os.path.join(config_path, "config.json"))

import pandas as pd
import numpy as np
import tensorflow as tf
import random

np.random.seed(configuration['np seed'])
tf.random.set_random_seed(configuration['tf seed'])
random.seed(configuration['np seed'] + configuration['tf seed'] + 1)

from sklearn.model_selection import train_test_split

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import visualization

import time


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

from keras import backend as K

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import  StandardScaler
from net import make_net, IntervalEvaluation

folder_name = config_path

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

X_original = X
y_original = y

test_size = configuration['test size']

if validation:
    X, X_validation, y, y_validation = train_test_split(X, y, train_size=float(len(X) - test_size)/len(X),
                                                        shuffle=True, random_state=configuration['val split seed'])
else:
    X_validation = None
    y_validation = None

batch_size = configuration['batch size']


def train_model(train, test, validation=None, fold_number=0, plot=False, verbose=False):
    if verbose:
        print('Fold', fold_number + 1, 'started at', time.ctime())
    K.clear_session()

    def data_generator(X, y, batch_size):
        counter = 1
        steps = len(X) / batch_size
        while True:
            if steps == 1:
                yield X, y
                continue

            for train_index, test_index in StratifiedKFold(n_splits=steps, shuffle=True,
                                                           random_state=counter).split(X, y):
                yield X[test_index], y[test_index]

    X_train, y_train = train
    X_test, y_test = test

    X_validation, y_validation = validation
    should_validate = y_validation is not None and X_validation is not None

    model, local_model = make_net(configuration)

    if verbose:
        print_list = ['TR {:10d} {:6.3f}%'.format(len(y_train), 100 * np.mean(y_train)),
                      'TS {:10d} {:6.3f}%'.format(len(y_test), 100 * np.mean(y_test))]
        if should_validate:
            print_list.append('VAL {:10d} {:6.3f}%'.format(len(y_validation), 100 * np.mean(y_validation)))
        print(' '.join(print_list))

    if plot:
        plot_everything(model, configs=configuration['discritezation'], experiment_folder=config_path, prefix='init_')

    checkpoint_path = os.path.join(config_path, 'model_' + str(fold_number))
    if verbose:
        print('Checkpoint at', checkpoint_path)

    callbacks = [
        ReduceLROnPlateau(monitor='val_' + configuration['selector'],
                          factor=configuration['lr reduce'],
                          patience=configuration['reduce patience'],
                          min_lr=configuration['min lr'],
                          mode=configuration['mode']),
        ModelCheckpoint(checkpoint_path,
                        monitor='val_' + configuration['selector'],
                        save_best_only=True,
                        mode=configuration['mode']),
        EarlyStopping(patience=configuration['stop patience'],
                      monitor='val_' + configuration['selector'],
                      mode=configuration['mode'])]

    if verbose:
        if should_validate:
            callbacks.append(IntervalEvaluation(validation_data=(X_validation, y_validation), interval=1))
        else:
            callbacks.append(IntervalEvaluation(validation_data=(X_test, y_test), interval=1))

    try:
        if configuration['stratify']:
            model.fit_generator(data_generator(X_train, y_train, batch_size),
                                steps_per_epoch=len(X_train) / batch_size,
                                validation_data=data_generator(X_test, y_test,
                                                               len(X_test) / configuration['validation steps']),
                                validation_steps=configuration['validation steps'],
                                epochs=configuration['epochs'],
                                callbacks=callbacks,
                                verbose=1 if verbose else 0)
        else:
            model.fit(X_train, y_train, batch_size=batch_size,
                      validation_data=(X_test, y_test),
                      epochs=configuration['epochs'],
                      callbacks=callbacks,
                      verbose=1 if verbose else 0)

    except KeyboardInterrupt:
        print('')

    model.load_weights(checkpoint_path)

    if plot:
        plot_everything(model, configs=configuration['discritezation'], experiment_folder=config_path, prefix='')

    return model


train_predictions_history = []
test_predictions_history = []
validation_predictions_history = []

if verbose:
    print('Training')

if configuration['folds'] > 1:
    iterator = enumerate(StratifiedKFold(n_splits=configuration['folds'], shuffle=True,
                                         random_state=configuration['test split seed']).split(X, y))
else:
    ind_train, ind_test = train_test_split(list(range(len(X))), test_size=test_size, shuffle=True,
                                           random_state=configuration['test split seed'])
    iterator = [(0, (ind_train, ind_test))]

log_file = open(os.path.join(config_path, "log"), 'w')

for fold_n, (train_index, test_index) in iterator:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = train_model((X_train, y_train),
                        (X_test, y_test),
                        validation=(X_validation, y_validation),
                        verbose=verbose,
                        fold_number=fold_n)

    total_prediction = model.predict(X_original)
    train_predictions_history.append(total_prediction)
    inst_total_str = 'INST TOTAL ROC AUC {:.10f}'.format(
        roc_auc_score(y_original, total_prediction))

    mean_total_str = 'MEAN TOTAL ROC AUC {:.10f}'.format(
        roc_auc_score(y_original, history_to_predictions_mean(train_predictions_history)))

    train_str = 'INST TRAIN ROC AUC {:.10f}'.format(roc_auc_score(y_train, model.predict(X_train).reshape(-1)))

    test_str = 'INST TEST ROC AUC {:.10f}'.format(roc_auc_score(y_test, model.predict(X_test).reshape(-1)))
    if verbose:
        print('------------------------------')
        print(mean_total_str)
        print(inst_total_str)
        print('------------------------------')
        print(train_str)
        print('------------------------------')
        print(test_str)

    if validation:
        validation_prediction = model.predict(X_validation)
        validation_predictions_history.append(validation_prediction)
        inst_val_str = 'INST VAL ROC AUC {:.10f}'.format(
            roc_auc_score(y_validation, validation_prediction))
        mean_val_str = 'MEAN VAL ROC AUC {:.10f}'.format(
            roc_auc_score(y_validation, history_to_predictions_mean(validation_predictions_history)))

        log_data_string = [fold_n,
                           roc_auc_score(y_validation, validation_prediction),
                           roc_auc_score(y_validation, history_to_predictions_mean(validation_predictions_history))]
        if verbose:
            print('------------------------------')
            print(mean_val_str)
            print(inst_val_str)
    else:
        log_data_string = [fold_n,
                           roc_auc_score(y_original, total_prediction),
                           roc_auc_score(y_original, history_to_predictions_mean(train_predictions_history))]

    if verbose:
        print('------------------------------')

    if submission:
        test_predictions_history.append(model.predict(test))

    log_data_string = [str(item) for item in log_data_string]
    log_file.write(' '.join(log_data_string) + '\n')

log_file.close()

if save:
    print('Saving predictions')
    sub = pd.DataFrame({"ID_code": data.ID_code.values, "target": data.target.values})
    sub["prediction"] = history_to_predictions_mean(train_predictions_history)
    sub.to_csv(os.path.join(config_path, "predictions.csv"), index=False)

if submission:
    print('Submission')
    sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
    sub["target"] = history_to_predictions_mean(test_predictions_history)
    sub.to_csv(os.path.join(config_path, "submission.csv"), index=False)
