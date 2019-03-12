from __future__ import print_function


def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import experiment_utils


def dicts_collect_equal_keys(dicts):
    if len(dicts) < 1:
        return []
    if len(dicts) == 1:
        return [(dicts.keys()[0], None)]
    total_keys = list(set(sum([dicts[key].keys() for key in dicts.keys()], [])))
    keys_to_consider = []
    for key in total_keys:
        dict_type = set(list([type(dicts[dkey].get(key, {})) == dict for dkey in dicts.keys()]))
        assert len(dict_type) == 1
        if list(dict_type)[0]:
            if key != 'loss':
                dicts_of_dicts = dict([(dkey, dicts[dkey].get(key, {})) for dkey in dicts.keys()])
                result = dicts_collect_equal_keys(dicts_of_dicts)
                if len(result) > 0:
                    keys_to_consider.append([(key, result)])
            else:
                # If no key here, something gone wrong
                unique_losses = set([dicts[dkey][key]['type'] for dkey in dicts.keys()])
                for loss in unique_losses:
                    loss_dict_of_dicts = dict(list([(dkey, dicts[dkey][key]['parameters']) for dkey in dicts.keys()
                                               if dicts[dkey][key]['type'] == loss]))
                    result = dicts_collect_equal_keys(loss_dict_of_dicts)
                    if len(result) > 0 or len(unique_losses) > 1:
                        keys_to_consider.append(('loss:' + loss, result))
        else:
            values = list([dicts[dkey].get(key, None) for dkey in dicts.keys()])
            if len(set(values)) == 1:
                continue
            collected_keys = list((dkey, dicts[dkey].get(key, None)) for dkey in dicts.keys())
            keys_to_consider.append((key, collected_keys))

    return keys_to_consider


def reccurent_collect(keys_data, additional_key=[]):
    result = []
    for key, local_data in keys_data:
        current_key = additional_key + [key]
        if type(local_data) == list:
            result += reccurent_collect(local_data, additional_key=current_key)
        else:
            result.append((current_key, local_data if local_data is not None else ''))
    return result


def dict_to_string(dictionary):
    c_string = []
    for key in dictionary.keys():
        if isinstance(dictionary[key], dict):
            c_string += [key] + dict_to_string(dictionary[key])
        else:
            c_string += [key, str(dictionary[key])]
    return c_string


def collect_valuable_parameters(dicts, path):
    equal_keys = dicts_collect_equal_keys(dicts)
    flat_keys = reccurent_collect(equal_keys)
    res = {}
    for names_list, value in flat_keys:
        c_dict = res
        name = names_list[len(names_list) - 1]
        if name not in c_dict.keys():
            c_dict[name] = {}
        c_dict = c_dict[name]
        for i in range(len(names_list) - 1):
            kname = names_list[i]
            if i != len(names_list) - 2:
                if kname not in c_dict.keys():
                    c_dict[kname] = {}
                c_dict = c_dict[kname]
            else:
                c_dict[kname] = value

    result = {}
    for key in res.keys():
        result[key] = ' '.join(dict_to_string(res[key]))

    return result


def get_log_results(log_file_path):
    try:
        with open(os.path.join(log_file_path, 'log'), 'r') as log:
            data = log.readlines()
        data_numerical = {}
        for string in data:
            splitted = string.split(' ')
            data_numerical[int(splitted[0])] = {"I": float(splitted[1]), "M": float(splitted[2])}
        return data_numerical
    except:
        return {}


parser = argparse.ArgumentParser()
parser.add_argument("experiment_folder", type=str, help="Path to experiment")

args = parser.parse_args()
experiment_folder = args.experiment_folder

config_path = os.path.join('experiments', experiment_folder)

assert os.path.isdir(config_path)

experiments = [os.path.join(experiment_folder, directory) for directory in os.listdir(config_path)
               if os.path.isdir(os.path.join(config_path, directory))]

logs = {}
configs = {}
for experiment in experiments:
    splitted = os.path.split(experiment)
    key = splitted[len(splitted) - 1]
    logs[key] = get_log_results(os.path.join('experiments', experiment))
    try:
        configs[key] = experiment_utils.load_configuration(os.path.join('experiments', experiment), config_path)
    except IOError:
        configs[key] = experiment_utils.load_configuration(os.path.join('experiments', experiment), 'experiments')

max_log_length = max([len(item.keys()) for item in logs.values()])
non_finished_keys = [key for key in logs.keys() if len(logs[key].keys()) != max_log_length]

if non_finished_keys:
    for key in non_finished_keys:
        del logs[key]
        del configs[key]

configs_compact = collect_valuable_parameters(configs, experiment_folder)
header = '| {:^5s} | {:60s} | {:^60s} | {:^10s} ------ {:^10s} |'.format('Fold', 'Experiment', 'Setup', 'Instant', 'Mean')
filler = ''.join(['-'] * len(header))
print()
print(filler)
print(header)
print(filler)
for fold in range(max_log_length):
    values = list([logs[key][fold]['M'] for key in configs_compact.keys()])
    values = zip(values, list(range(len(values))))
    values = sorted(values, key=lambda x: x[0], reverse=True)
    indicies = zip(*values)[1]
    for ind in indicies:
        key = configs_compact.keys()[ind]
        pstr = '| {:5d} | {:^60s} | {:^60s} | {:10.8f} ------ {:10.8f} |'
        print(pstr.format(1 + fold, key, configs_compact[key], logs[key][fold]['I'], logs[key][fold]['M']))

    print(filler)

print()
