from __future__ import print_function


def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("experiment_folder", type=str, help="Path to experiment")
parser.add_argument("--silent", action='store_true', help="Work in silent mode")
parser.add_argument("--save", action='store_true', help="Save predictions for outliers detection")

args = parser.parse_args()

experiment_folder = args.experiment_folder
verbose = not args.silent
save = args.save

config_path = os.path.join('experiments', experiment_folder)

assert os.path.isdir(config_path)

experiments = [os.path.join(experiment_folder, directory) for directory in os.listdir(config_path)
               if os.path.isdir(os.path.join(config_path, directory))]

command_template = 'python train_net.py {:s} -v'
if not verbose:
    command_template += ' --silent'
if save:
    command_template += ' --save'

for experiment_folder in experiments:
    print('Running', experiment_folder)
    print('Command :', command_template.format(experiment_folder))
    os.system(command_template.format(experiment_folder))

