import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import keras


def draw_function(func, true_values=None, bounds=None, label=None):
    """
    Plots the `func`(x) values on a [-3, 3] grid
    """
    # plt.clf()
    ax = plt.subplot(111)
    if true_values is not None:
        # plot train data distribution
        sns.distplot(true_values, ax=ax, hist=False, label=label)
        [line.set_linestyle("--") for line in ax.lines]

    if bounds is None:
        bounds = [-3, 3]
    X = np.linspace(bounds[0], bounds[1], 100)
    y = [func(x) for x in X]
    [line.set_linestyle("-") for line in ax.lines]
    sns.lineplot(x=X, y=y, ax=ax)
    return ax


def plot_weights(bin, width, bias, disc_layer_configs, true_values=None, label=None):
    """
    Plots pre-softmax function of i-th input feature and j-th output feature
    """
    activation_fn = disc_layer_configs['pre_sm_activation']

    def activation(x):
        if activation_fn == 'elu':
            return np.exp(x) - 1 if x < 0 else x
        elif activation_fn == 'lelu':
            return x * 0.2 if x < 0 else x
        elif activation_fn == 'iden':
            return x
        else:
            raise Exception
    y_func = lambda x: activation(bias - abs(x - bin) * width)
    return draw_function(y_func, true_values, label=label)


def get_weight_by_name(model, name):
    ind = None
    for i, weight in enumerate(model.trainable_weights):
        if weight.name == name:
            ind = i
    if ind is None:
        raise Exception("Name %s not found in model weights" % name)
    return model.get_weights()[ind]


def get_disc_layer_weights(model, model_type='DiscretizationLayerWide'):
    if model_type == 'DiscretizationLayerWide':
        n2n = {'bins': 'discretization_layer_wide_1/bins:0',
               "widths": 'discretization_layer_wide_1/widths:0',
               'biases': 'discretization_layer_wide_1/biases:0',
               'wc_w': 'discretization_layer_wide_1/w:0',
               'wc_b': 'discretization_layer_wide_1/b:0'}
        return {k: get_weight_by_name(model, v) for k, v in n2n.items()}
    else:
        raise Exception("No such model type %s" % model_type)


def plot_all_bins(bins, true_values, bounds=None):
    ax = plt.subplot(111)

    if true_values is not None:
        # plot train data distribution
        sns.distplot(true_values, ax=ax, hist=False, label='feature')
        [line.set_linestyle("--") for line in ax.lines]

    if bounds is None:
        bounds = [-3, 3]
    sns.distplot(bins, kde=True, hist=False, rug=True, label='bins')

    return ax


def plot_all_bins_model(model, data, feature_list, target_path_prefix, model_type='DiscretizationLayerWide', ):
    weights = get_disc_layer_weights(model, model_type=model_type)
    for i in feature_list:
        plt.clf()
        plot_all_bins(weights['bins'][i], data[i])
        plt.savefig(target_path_prefix + ('_%d.png' % i))


def plot_dist(input_feature_ind, output_feature_ind, train_data, bins, widths, biases, disc_layer_configs, **kwargs):
    bin = bins[input_feature_ind, output_feature_ind]
    width = widths[input_feature_ind, output_feature_ind]
    bias = biases[input_feature_ind, output_feature_ind]
    true_values = train_data[:, input_feature_ind] if train_data is not None else None
    return plot_weights(bin, width, bias, disc_layer_configs=disc_layer_configs, true_values=true_values)


def plot_bin_vertical(model, out_feature_inds, target_path, model_type='DiscretizationLayerWide'):
    plt.clf()
    weights = get_disc_layer_weights(model, model_type=model_type)
    ax = plt.subplot(111)
    for j in out_feature_inds:
        bins = weights['bins'][:, j]
        sns.distplot(bins, hist=False, label='ind: %d' % j, ax=ax)
    plt.savefig(target_path)


def plot_presoftmax(model, data, in_feature_inds, out_feature_inds, target_path_prefix, disc_layer_configs, axis='horizontal',
                    model_type='DiscretizationLayerWide'):
    weights = get_disc_layer_weights(model, model_type=model_type)
    assert axis in ['vertical', 'horizontal']

    in_ind = 0
    out_ind = 0

    while in_ind < len(in_feature_inds) and out_ind < len(out_feature_inds):
        plt.clf()
        while in_ind < len(in_feature_inds) and out_ind < len(out_feature_inds):
            in_feature = in_feature_inds[in_ind]
            out_feature = out_feature_inds[out_ind]
            plot_weights(weights['bins'][in_feature, out_feature],
                         weights['widths'][in_feature, out_feature],
                         weights['biases'][in_feature, out_feature],
                         disc_layer_configs=disc_layer_configs,
                         true_values=data[:, in_feature] if axis == 'horizontal' else None,
                         label='%s feature=%d' % (('in', in_feature) if axis == 'vertical' else ('out', out_feature)))
            if axis == 'vertical':
                in_ind +=1
            else:
                out_ind += 1
        plt.savefig(target_path_prefix + ('_%d.png' % (in_feature if axis == 'horizontal' else out_feature)))
        if axis == 'vertical':
            out_ind += 1
            in_ind = 0
        else:
            out_ind = 0
            in_ind += 1

    # for i in iter_1:
    #     plt.clf()
    #     for j in iter_2:




# def plot_model(model, plot_fn, target_path_prefix, model_type='DiscretizationLayerWide', **kwargs):
#     weights = get_disc_layer_weights(model, model_type=model_type)
#     plt.clf()
#     ax = plt.subplot(111)
#     plot_fn(weights=weights, ax=ax, **kwargs)

# import pandas as pd
# from net import make_net
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# model, local_model = make_net(100, 1e-3, configs={'disc_layer': {'bins_init': 'uniform'}})
#
# data = pd.read_csv('./data/data_train.csv')
# X = data.drop(columns=['ID_code', 'target']).values
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
#
# weights = get_disc_layer_weights(model)
# plot_all_bins(weights['bins'][0], X[0])
# plt.savefig('all_bins_%d_uniform_10_start.png' % 0)
# plt.clf()
#
# model.load_weights('runs/uniform_10/model_0')
# weights = get_disc_layer_weights(model)
# for i in range(10):
#     in_feature = i
#     plot_all_bins(weights['bins'][in_feature], X[in_feature])
#     plt.savefig('all_bins_%d_uniform_10.png' % i)
#     plt.clf()

    # import pandas as pd
    # from net import make_net
    # from sklearn.preprocessing import MinMaxScaler, StandardScaler
    # from deep_dict import DeepDict
    #
    # configs = DeepDict({'disc_layer': {'bins_init': 'uniform',
    #                                    'bins_init_range': 3}})
    # model, local_model = make_net(100, 1e-3, configs=configs)
    # model.load_weights('runs/linspace/model_0')
    # weights = get_disc_layer_weights(model)
    # data = pd.read_csv('./data/data_train.csv')
    # X = data.drop(columns=['ID_code', 'target']).values
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    # plot_presoftmax(model, X, in_feature_inds=range(0, 100, 10), out_feature_inds=range(0, 100, 10), axis='horizontal',
    #                 target_path_prefix='./pics/horizontal_function_vis_')
    # plot_presoftmax(model, X, in_feature_inds=range(0, 100, 10), out_feature_inds=range(0, 100, 10), axis='vertical',
    #                 target_path_prefix='./pics/vertical_function_vis_')
#
#
#
# for i in range(10):
#     in_feature = i
#     gt = X
#     # plot_all_bins(weights['bins'][in_feature], X[in_feature])
#     for j in range(0, 150, 10):
#         plot_dist(i, j, gt, **weights)
#         gt = None # plot true distribution only once
#     plt.savefig('all_bins_dists_%d.png' % i)
#     plt.clf()

