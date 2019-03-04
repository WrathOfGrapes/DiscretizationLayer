import argparse
import time
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prod", type=bool, default=False, help="Train net on 5 folds without validation set (for kagle submission)")
    parser.add_argument('-c', dest='configs', type=str, default=None, help="Path to config file")
    parser.add_argument('-name', type=str, default=None, help='Experiment name. Default: time in {%H:%M:%S} format')
    return parser.parse_args()


def create_experiment_folder(folder_name=None):
    name = folder_name or time.strftime('{%H_%M_%S}')
    os.makedirs('./runs/' + name + '/train')
    os.makedirs('./runs/' + name + '/test')


