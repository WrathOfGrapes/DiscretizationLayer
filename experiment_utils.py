import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str,
                        help="Path to experiment")
    parser.add_argument("-s", "--submission", action='store_true', help="Create kaggle submission")
    parser.add_argument("-v", "--validation", action='store_true', help="Use validation dataset")
    parser.add_argument("--silent", action='store_true', help="Work in silent mode")
    parser.add_argument("--save", action='store_true', help="Save predictions for outliers detection")
    return parser.parse_args()


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def load_json(path):
    with open(path, 'r') as experiment_file:
        configuration = json.load(experiment_file)
    return configuration


def load_configuration(config_path, experiment_name):
    configuration = load_json(os.path.join(config_path, "config.json"))

    configuration_defaults = load_json(os.path.join(experiment_name, "config_defaults.json"))
    configuration = merge_two_dicts(configuration_defaults, configuration)

    losses_defaults = load_json(os.path.join(experiment_name, "losses_defaults.json"))
    configuration = merge_two_dicts(configuration_defaults, configuration)

    configuration['loss']['parameters'] = merge_two_dicts(losses_defaults[configuration['loss']['type']],
                                                          configuration['loss']['parameters'])

    discretization_defaults = load_json(os.path.join(experiment_name, "discretization_defaults.json"))
    configuration['discritezation'] = merge_two_dicts(discretization_defaults, configuration['discritezation'])

    return configuration


