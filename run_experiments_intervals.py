from __future__ import print_function
import experiment_utils
import argparse
import os
import itertools
import json
import numpy as np


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_gitignore(directory):
    with open(os.path.join(directory, '.gitignore'), 'w') as f:
        f.write('*\n')
        f.write('!*.json\n')


def merge_dicts(d1, d2):
    result = {}
    total_keys = set(d1.keys() + d2.keys())
    for key in total_keys:
        if key in d1.keys() and key in d2.keys():
            if isinstance(d1[key], dict) and isinstance(d1[key], dict):
                result[key] = merge_dicts(d1[key], d2[key])
            else:
                result[key] = d2[key]
        else:
            result[key] = d1[key] if key in d1.keys() else d2[key]
    return result


def get_dict_from_keys(keys_list, value):
    result = {}
    cres = result
    for key in keys_list[:len(keys_list) - 1]:
        cres[key] = {}
        cres = cres[key]
    cres[keys_list[len(keys_list) - 1]] = value
    return result


def extract_dict_keys(d):
    result = []
    while isinstance(d, dict):
        result.append(d.keys()[0])
        d = d[d.keys()[0]]
    return result


def extract_dict_value(d):
    while isinstance(d, dict):
        d = d[d.keys()[0]]
    return d


def unroll_intervals(intervals_config, previous_keys=[]):
    result = []
    for key in intervals_config.keys():
        current_keys = previous_keys + [key]

        if isinstance(intervals_config[key], dict):
            result += unroll_intervals(intervals_config[key], current_keys)
        if isinstance(intervals_config[key], list):
            result.append(list([get_dict_from_keys(current_keys, item) for item in intervals_config[key]]))

    return result


parser = argparse.ArgumentParser()
parser.add_argument("experiment_folder", type=str, help="Path to experiment")
parser.add_argument("-s", "--samples", type=int, default=60,
                    help="Corresponding to number of random samples, if 0 all possible samples will be generated")
parser.add_argument("--seed", type=int, default=None, help="Random seed")

args = parser.parse_args()

experiment_folder = args.experiment_folder
samples = args.samples

config_path = os.path.join('experiments', experiment_folder)
assert os.path.isdir(config_path)

config = experiment_utils.load_json(os.path.join(config_path, 'config_template.json'))
intervals = experiment_utils.load_json(os.path.join(config_path, 'config_intervals.json'))

unrolled = unroll_intervals(intervals)
unrolled = sum(unrolled, [])

mapping = {}
for item in unrolled:
    keys = extract_dict_keys(item)
    identifier = '#'.join(keys)
    if identifier not in mapping.keys():
        mapping[identifier] = []
    mapping[identifier].append(item)


configuration_list = []

if samples > 0:
    np.random.seed(args.seed)

    value = 1
    limits = {}
    for key in mapping.keys():
        value *= len(mapping[key])
        limits[key] = len(mapping[key])

    points = []
    sample_collection = []
    while len(points) != samples:
        sample = list([(key, np.random.randint(0, limits[key])) for key in limits.keys()])
        sample_to_string = ''.join(list([str(item[1]) for item in sample]))
        if sample_to_string not in sample_collection:
            points.append(sample)
            sample_collection.append(sample_to_string)

    for point in points:
        configuration_list.append(list([mapping[key][number] for key, number in point]))
else:
    configuration_list = mapping[mapping.keys()[0]]
    for key in mapping.keys()[1:]:
        configuration_list = list(itertools.product(mapping[key], configuration_list))
        for i in range(len(configuration_list)):
            item1, item2 = configuration_list[i]
            if not isinstance(item2, tuple):
                item = (item1, item2)
            else:
                item = (item1,) + item2
            configuration_list[i] = item

print(len(configuration_list), 'samples will be generated')

configurations = []
for item in configuration_list:
    init_dict = item[0]
    for litem in item[1:]:
        init_dict = merge_dicts(init_dict, litem)
    configurations.append(init_dict)

for i in range(len(configurations)):
    merged_with_config = merge_dicts(config, configurations[i])
    path = os.path.join(config_path, experiment_folder + '_' + str(i))
    create_dir(path)
    create_gitignore(path)
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(merged_with_config, f)