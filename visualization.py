import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import keras


def draw_function(func, true_values=None, bounds=None):
    ax = plt.subplot(111)

    if true_values is not None:
        # plot train data distribution
        sns.distplot(true_values, ax=ax, hist=False)
        [line.set_linestyle("--") for line in ax.lines]

    if bounds is None:
        bounds = [-3, 3]
    X = np.linspace(bounds[0], bounds[1], 100)
    y = [func(x) for x in X]
    [line.set_linestyle("-") for line in ax.lines]
    sns.lineplot(x=X, y=y, ax=ax)
    return ax


def plot_weights(bin, width, bias, true_values=None):
    y_func = lambda x: bias - abs(x - bin) * width
    return draw_function(y_func, true_values)


def get_weight_by_name(model: keras.models.Model, name):
    ind = None
    for i, weight in enumerate(model.trainable_weights):
        if weight.name == name:
            ind = i
    if ind is None:
        raise Exception("Name %s not found in model weights" % name)
    return model.get_weights()[ind]


def get_disc_layer_weights(model: keras.models.Model, model_type: str = 'DiscretizationLayerWide'):
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

def plot_dist(input_feature_ind, output_feature_ind, train_data, bins, widths, biases,  **kwargs, ):
    bin = bins[input_feature_ind, output_feature_ind]
    width = widths[input_feature_ind, output_feature_ind]
    bias = biases[input_feature_ind, output_feature_ind]
    return  plot_weights(bin, width, bias, true_values=train_data[:, input_feature_ind])



# import pandas as pd
# from net import make_net
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
#
#
# model, local_model = make_net(100, 1e-3, configs=None)
# model.load_weights('model_0')
# weights = get_disc_layer_weights(model)
# data = pd.read_csv('./data/data_train.csv')
# X = data.drop(columns=['ID_code', 'target']).values
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
#
# for i in range(10):
#     in_feature = i
#     plot_all_bins(weights['bins'][in_feature], X[in_feature])
#     plt.savefig('all_bins_%d.png' % i)
#     plt.clf()