import argparse
import time
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fold", type=int, default=1, help="Train net on 5 folds without validation set (for kagle submission)")
    parser.add_argument("-s", "--submission", action='store_true', help="Create kaggle submission")
    parser.add_argument("-v", "--validation", action='store_true', help="Use validation dataset")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    parser.add_argument("-n", "--name", type=str, default=None, help='Experiment name. Default: time in {HH:MM:SS} format')
    parser.add_argument("-lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("-d", "--dimension", type=int, default=100, help="Internal dimension")
    return parser.parse_args()


def create_experiment_folder(folder_name=None):
    name = folder_name or time.strftime('%H_%M_%S')
    i = 0
    while i < 100:
        try:
            postfix = '' if i == 0 else "_%d_" % i
            os.makedirs('./runs/' + name + postfix)
            i = None
            break
        except:
            i += 1
    if i is not None:
        raise Exception("There are more then 100 experiment folders with name `%s`!" % folder_name)
    os.makedirs('./runs/' + name + postfix + '/train')
    os.makedirs('./runs/' + name + postfix + '/test')
    os.makedirs('./runs/' + name + postfix + '/pics')
    return name + postfix



