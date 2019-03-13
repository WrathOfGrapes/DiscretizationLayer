from __future__ import print_function
import experiment_utils
import argparse
import os
import itertools
import json
import numpy as np
from loss_config_sanity import check_loss


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_gitignore(directory):
    with open(os.path.join(directory, '.gitignore'), 'w') as f:
        f.write('*\n')


def binary_random(a, b, size=None):
    binary_data = np.random.randint(0, 2, size=size)
    return binary_data * a + (1 - binary_data) * b


def is_distribution(value):
    if not (isinstance(value, unicode) or (isinstance(value, str))):
        return False
    value = value.split(' ')
    if len(value) != 3:
        return False
    if value[0] not in ['U', 'N', 'B']:
        return False
    try:
        float(value[1])
        float(value[2])
    except ValueError:
        return False
    return True


def run_distribution(distribution, size=None):
    distribution = distribution.split(' ')
    assert len(distribution) == 3
    dtype = distribution[0]
    fval = float(distribution[1])
    mval = float(distribution[2])
    distr = None
    if dtype == 'U':
        distr = np.random.uniform(fval, mval, size=size)
    if dtype == 'N':
        distr = np.random.normal(fval, mval, size=size)
    if dtype == 'B':
        distr = binary_random(fval, mval, size=size)

    return distr.tolist()


def check_randomness(d):
    result = False
    for key in d.keys():
        if isinstance(d[key], dict):
            result = result or check_randomness(d[key])
        else:
            result = result or is_distribution(d[key])
    return result


def random_instance(d, size):
    result = d
    for key in d.keys():
        if isinstance(d[key], dict):
            d[key] = random_instance(d[key], size)
        else:
            if is_distribution(d[key]):
                result[key] = run_distribution(d[key], size)
            else:
                result[key] = d[key]
    return result


def disassemble_instance(d):
    one = {}
    rest = {}
    final = False
    for key in d.keys():
        if isinstance(d[key], dict):
            one_p, rest_p = disassemble_instance(d[key])
            one[key] = one_p
            rest[key] = rest_p
            final = rest_p == {}
        else:
            if isinstance(d[key], list):
                one[key] = d[key][0]
                if len(d[key]) == 1:
                    final = True
                rest[key] = d[key][1:]
            else:
                one[key] = d[key]
                rest[key] = d[key]
    if final:
        rest = {}
    return one, rest


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


def configuration_generator(r, d):
    for item in d:
        yield item
    while True:
        insances = []
        for rng in r:
            assembled_instance = random_instance(rng, 100)
            while assembled_instance != {}:
                instance, assembled_instance = disassemble_instance(assembled_instance)
                insances.append(instance)
        np.random.shuffle(insances)
        for item in insances:
            yield item


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

config_template = experiment_utils.load_json(os.path.join(config_path, 'config_template.json'))
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

configuration_list = mapping[mapping.keys()[0]]
if len(mapping.keys()) == 1:
    configuration_list = list([[item] for item in configuration_list])
for key in mapping.keys()[1:]:
    configuration_list = list(itertools.product(mapping[key], configuration_list))
    for i in range(len(configuration_list)):
        item1, item2 = configuration_list[i]
        if not isinstance(item2, tuple):
            item = (item1, item2)
        else:
            item = (item1,) + item2
        configuration_list[i] = item

value = len(configuration_list)

configurations = []

for item in configuration_list:
    init_dict = item[0]
    for litem in item[1:]:
        init_dict = merge_dicts(init_dict, litem)
    configurations.append(init_dict)

generated = 0

if samples > 0:
    np.random.seed(args.seed)

rng_configurations = []
det_configurations = []
for c in configurations:
    if check_randomness(c):
        rng_configurations.append(c)
    else:
        det_configurations.append(c)

if len(rng_configurations) != 0:
    assert samples != 0
    configurations = configuration_generator(rng_configurations, det_configurations)
else:
    if samples > 0:
        np.random.shuffle(configurations)

for item in configurations:
    merged_with_config = merge_dicts(config_template, item)

    if 'loss' in merged_with_config.keys() and not check_loss(merged_with_config['loss']):
        continue

    generated += 1

    if generated == samples and samples != 0:
        break

    path = os.path.join(config_path, experiment_folder + '_' + str(generated))
    create_dir(path)
    create_gitignore(path)
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(merged_with_config, f)


print('{:5d} of samples generated'.format(generated))