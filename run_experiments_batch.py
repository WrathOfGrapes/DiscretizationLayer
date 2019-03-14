from __future__ import print_function
import pandas as pd
from train_net import *


def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("experiment_folder", type=str, help="Path to experiment or set of experiments")
parser.add_argument("--silent", action='store_true', help="Work in silent mode")
parser.add_argument("--save", action='store_true', help="Save predictions for outliers detection")

args = parser.parse_args()

experiment_folder = args.experiment_folder
experiment_folders = experiment_folder.split(' ')
verbose = not args.silent
save = args.save

data = pd.read_csv('./data/train.csv')
X = data.drop(columns=['ID_code', 'target']).values
y = data['target'].values

test_df = pd.read_csv('./data/test.csv')
test = test_df.drop(columns=['ID_code']).values

for folder in experiment_folders:
    config_path = os.path.join('experiments', folder)

    experiments = [os.path.join(folder, directory) for directory in os.listdir(config_path)
                   if os.path.isdir(os.path.join(config_path, directory))]

    for experiment in sorted(experiments):
        print('Running', experiment)

        configuration = load_config(experiment)

        train_network(X, y, test, configuration, verbose, False, True, save)

