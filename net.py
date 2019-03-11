import keras.layers as l
from keras.models import Model

from layers import LogarithmLayer, DiscretizationLayer, DiscretizationLayerWide, LaplaceLayerWide

import logging

from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import roc_auc_score, mean_squared_error
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras_metrics import f1_score


def deep_cancer_loss(y_true, y_pred):

    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
    pos = tf.cond(tf.is_nan(tf.reduce_mean(pos)), lambda: tf.constant(1, dtype=pos.dtype, shape=(1,)), lambda: pos)
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))
    neg = tf.cond(tf.is_nan(tf.reduce_mean(neg)), lambda: tf.constant(0, dtype=neg.dtype, shape=(1,)), lambda: neg)

    pos = tf.expand_dims(pos, 0)
    neg = tf.expand_dims(neg, 1)

    difference = tf.zeros_like(tf.matmul(neg, pos))
    difference = difference + neg
    difference = difference - pos

    difference = tf.log(1 + tf.exp(difference))

    pos_means = tf.reduce_mean(difference, axis=0)
    neg_means = tf.reduce_mean(difference, axis=1)

    reduced = tf.concat([pos_means, neg_means], axis=0)
    res = tf.reduce_mean(reduced)
    loss = tf.cond(tf.is_nan(res), lambda: tf.reduce_mean(y_true), lambda: res)
    loss = 1 * loss# + 0.9 * K.binary_crossentropy(y_true, y_pred)
    return loss


def deep_cancer_loss_offline(y_true, y_pred):

    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

    pos = tf.expand_dims(pos, 0)
    neg = tf.expand_dims(neg, 1)

    difference = tf.zeros_like(tf.matmul(neg, pos)) + neg - pos

    difference = tf.log(1 + tf.exp(difference))

    res = tf.reduce_mean(difference)
    return tf.cond(tf.is_nan(res), lambda: tf.reduce_mean(y_true), lambda: res)


def auc_loss(y_true, y_pred):
    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

    pos = tf.expand_dims(pos, 0)
    neg = tf.expand_dims(neg, 1)

    beta = 0.9

    difference = tf.sigmoid(beta * (tf.zeros_like(tf.matmul(neg, pos)) + pos - neg))

    res = tf.reduce_mean(difference)

    return 1 - tf.cond(tf.is_nan(res), lambda: tf.reduce_mean(y_true), lambda: res)

def metric_check(y_true, y_pred):
    ''' Implementation from the paper
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic'''
    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

    pos = tf.expand_dims(pos, 0)
    neg = tf.expand_dims(neg, 1)

    gamma = 0.0
    p = 2

    difference = tf.zeros_like(pos * neg) + pos - neg - gamma
    masked = tf.cast(difference < 0, dtype=difference.dtype)

    return tf.reduce_sum(masked)


def wmw_loss_canonical(y_true, y_pred):
    ''' Implementation from the paper
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic'''
    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

    pos = tf.expand_dims(pos, 0)
    neg = tf.expand_dims(neg, 1)

    gamma = 0.5
    p = 1.5

    difference = tf.zeros_like(tf.matmul(neg, pos)) + pos - neg - gamma
    masked = difference * tf.cast(difference < 0, dtype=difference.dtype)
    res = tf.reduce_mean(tf.pow(-masked, p))
    return tf.cond(tf.is_nan(res), lambda: tf.reduce_mean(y_true), lambda: res)


def wmw_loss(y_true, y_pred):
    '''tflearn implementation'''
    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

    pos = tf.expand_dims(pos, 0)
    neg = tf.expand_dims(neg, 1)

    gamma = 0.5
    p = 2

    difference = tf.zeros_like(pos * neg) + pos - neg - gamma
    masked = tf.boolean_mask(difference, difference < 0.0)
    res = tf.reduce_sum(tf.pow(-masked, p))

    loss = tf.cond(tf.is_nan(res), lambda: tf.reduce_mean(tf.zeros_like(y_true)), lambda: res)

    loss = loss#tf.losses.mean_squared_error(y_true, y_pred)

    return loss


def some_loss(y_true, y_pred):
    y_true = tf.reduce_sum(y_true, axis=1)
    y_pred = tf.reduce_sum(y_pred, axis=1)
    #alpha = 1 - tf.maximum(tf.reduce_mean(y_true), K.epsilon() * tf.ones_like(tf.reduce_mean(y_true)))

    # Rebalance classes, so now equal probability is 1 - alpha
    alpha = tf.reduce_mean(y_true)
    threshold_pos = 1 - alpha#1 - alpha + gap / 2
    threshold_neg = alpha#K.epsilon() + alpha#alpha - gap / 2
    target_balance = 0.5
    neg_neg_normed = y_pred * target_balance / threshold_neg
    pos_neg_normed = y_pred * target_balance / threshold_pos
    neg_pos_normed = target_balance + (1 - target_balance) * (y_pred - threshold_neg) / (1. - threshold_neg)
    pos_pos_normed = target_balance + (1 - target_balance) * (y_pred - threshold_pos) / (1. - threshold_pos)
    neg_neg = (1 - y_true) * tf.cast(y_pred < threshold_neg, dtype=y_true.dtype) * neg_neg_normed
    pos_neg = y_true * tf.cast(y_pred < threshold_pos, dtype=y_true.dtype) * pos_neg_normed
    neg_pos = (1 - y_true) * tf.cast(y_pred >= threshold_neg, dtype=y_true.dtype) * neg_pos_normed
    pos_pos = y_true * tf.cast(y_pred >= threshold_pos, dtype=y_true.dtype) * pos_pos_normed

    neg = neg_neg + neg_pos
    pos = pos_pos + pos_neg

    predictions = pos + neg

    y_true_pos = tf.expand_dims(y_true, 0)
    y_true_neg = tf.expand_dims(1 - y_true, 1)

    true_diff_matrix = tf.zeros_like(tf.matmul(y_true_pos, y_true_neg)) + y_true_pos - y_true_neg
    true_diff_matrix = 1 - tf.abs(true_diff_matrix)

    y_pred_pos = tf.expand_dims(y_true * y_pred, 0)
    y_pred_neg = tf.expand_dims((1 - y_true) * y_pred, 1)

    gap = np.abs(threshold_pos - threshold_neg)

    pred_diff_matrix = tf.zeros_like(y_pred_pos * y_pred_neg) + y_pred_pos - y_pred_neg - gap
    pred_diff_matrix = tf.cast(pred_diff_matrix < 0, dtype=y_true.dtype)
    pred_diff_matrix *= true_diff_matrix

    roc_auc_value = 1 - 2 * tf.reduce_sum(pred_diff_matrix) / tf.reduce_sum(true_diff_matrix)
    dim = K.epsilon() + tf.reduce_sum(y_true) * tf.reduce_sum(1 - y_true) * (K.epsilon() + 1 - roc_auc_value)

    #s = 2.
    #pred_diff_matrix += true_diff_matrix / s
    #dim += tf.reduce_sum(true_diff_matrix) / s

    pos_vector = y_true * tf.reduce_sum(pred_diff_matrix, axis=0) / dim
    neg_vector = (1 - y_true) * tf.reduce_sum(pred_diff_matrix, axis=1) / dim

    pos_weights = pos_vector * tf.reduce_sum(y_true)
    neg_weights = neg_vector * tf.reduce_sum(1 - y_true)

    weight_vector = (pos_weights + neg_weights)

    weight_vector /= (K.epsilon() + tf.reduce_mean(weight_vector))

    bce = -y_true * tf.log(K.epsilon() + predictions) - (1 - y_true) * tf.log(K.epsilon() + 1 - predictions)

    loss = tf.reduce_mean(bce * weight_vector)

    return loss


def sliding_wmw_loss(y_true, y_pred):
    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

    pos = tf.expand_dims(pos, 0)
    neg = tf.expand_dims(neg, 1)

    p = 3

    steps = 10
    lb = 0
    ub = 1

    segment = float(ub - lb) / steps

    def approx_function_canonical(x):
        masked = x * tf.cast(tf.less(x, 0.0), dtype=x.dtype)
        res = tf.reduce_sum(tf.pow(-masked, p))
        return tf.cond(tf.is_nan(res), lambda: tf.reduce_mean(y_true), lambda: res)

    def approx_function(x):
        masked = tf.boolean_mask(x, tf.less(x, 0.0))
        res = tf.reduce_mean(tf.pow(-masked, p))
        return tf.cond(tf.is_nan(res), lambda: tf.reduce_mean(y_true), lambda: res)

    difference = tf.zeros_like(tf.matmul(neg, pos)) + pos - neg
    print('DET')
    diff_vector = tf.constant(np.linspace(lb, ub, steps), dtype=difference.dtype, shape=(steps, 1, 1))
    #print 'RNG'
    #diff_vector = tf.concat([tf.random_uniform((steps, 1, 1), minval=lb, maxval=ub), diff_vector], axis=0)
    #steps *= 2
    difference = tf.expand_dims(difference, 0)
    difference = tf.tile(difference, [steps, 1, 1])
    difference -= diff_vector
    difference = tf.cond(
        tf.logical_or(tf.is_nan(tf.reduce_mean(pos)), tf.is_nan(tf.reduce_mean(neg))),
        lambda: y_true, lambda: tf.map_fn(approx_function, difference))
    return tf.reduce_mean(difference * (1 - diff_vector))


def page_rank_good_neg_loss(y_true, y_pred):
    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

    # Rebalance classes, so now equal probability is 1 - alpha
    alpha = tf.reduce_mean(y_true)
    gap = alpha / 10
    threshold_pos = 1 - alpha + gap
    threshold_neg = 1 - alpha - gap
    target_balance = 0.5

    neg_neg_normed = neg * target_balance / threshold_neg
    pos_neg_normed = pos * target_balance / threshold_pos
    neg_pos_normed = target_balance + (1 - target_balance) * ((neg - threshold_neg) / (1. - threshold_neg))
    pos_pos_normed = target_balance + (1 - target_balance) * ((pos - threshold_pos) / (1. - threshold_pos))

    neg_rebalanced = tf.where(neg < threshold_neg, neg_neg_normed, neg_pos_normed)
    pos_rebalanced = tf.where(pos < threshold_pos, pos_neg_normed, pos_pos_normed)

    pos_exp = tf.expand_dims(pos, 0)
    neg_exp = tf.expand_dims(neg, 1)

    pos_percentage = tf.reduce_mean(y_true)

    difference = tf.zeros_like(pos_exp * neg_exp) + pos_exp - neg_exp
    difference_errors = tf.where(difference <= 0, tf.ones_like(difference), tf.zeros_like(difference))

    difference_correct = tf.where(difference > 0, tf.ones_like(difference), tf.zeros_like(difference))

    normalize = lambda x: x / (K.epsilon() + tf.reduce_sum(x))

    error_pos = normalize(tf.ones_like(pos))
    error_neg = normalize(tf.ones_like(neg))

    correct_pos = normalize(tf.ones_like(pos))
    correct_neg = normalize(tf.ones_like(neg))

    iterations = 50

    for i in range(iterations):
        normed_neg_difference_errors = difference_errors / \
                                       (K.epsilon() + tf.reduce_sum(difference_errors, axis=0, keepdims=True))
        err_positive_ratings = tf.matmul(tf.transpose(normed_neg_difference_errors), tf.expand_dims(error_neg, 1))
        err_positive_ratings = tf.squeeze(err_positive_ratings)

        normed_pos_difference_errors = difference_errors / \
                                       (K.epsilon() + tf.reduce_sum(difference_errors, axis=1, keepdims=True))
        err_negative_ratings = tf.matmul(normed_pos_difference_errors, tf.expand_dims(error_pos, 1))
        err_negative_ratings = tf.squeeze(err_negative_ratings)

        error_pos = normalize(error_pos + err_positive_ratings)
        error_neg = normalize(error_neg + err_negative_ratings)

        normed_pos_difference_correct = difference_correct/ \
                                        (K.epsilon() + tf.reduce_sum(difference_correct, axis=0, keepdims=True))
        corr_positive_ratings = tf.matmul(tf.transpose(normed_pos_difference_correct),
                                          tf.expand_dims(correct_neg, 1))
        corr_positive_ratings = tf.squeeze(corr_positive_ratings)

        normed_neg_difference_correct = difference_correct / \
                                        (K.epsilon() + tf.reduce_sum(difference_correct, axis=1, keepdims=True))
        corr_negative_ratings = tf.matmul(normed_neg_difference_correct,
                                          tf.expand_dims(correct_pos, 1))
        corr_negative_ratings = tf.squeeze(corr_negative_ratings)

        correct_pos = normalize(correct_pos + corr_positive_ratings)
        correct_neg = normalize(correct_neg + corr_negative_ratings)

    correct_pos /= pos_percentage
    error_pos /= pos_percentage

    correct_neg /= 1 - pos_percentage
    error_neg /= 1 - pos_percentage

    pos_ratings = normalize(tf.sqrt(correct_pos * error_pos))
    neg_ratings = normalize(tf.sqrt(correct_neg * error_neg))

    pos_bce = -1 * tf.log(K.epsilon() + pos_rebalanced)
    neg_bce = -1 * tf.log(K.epsilon() + 1 - neg_rebalanced)

    loss = pos_percentage * tf.reduce_sum(pos_bce * pos_ratings) + \
           (1 - pos_percentage) * tf.reduce_sum(neg_bce * neg_ratings)

    return loss


def shifted_bce(y_true, y_pred):
    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

    # Rebalance classes, so now equal probability is 1 - alpha
    alpha = tf.reduce_mean(y_true)
    pos_percentage = tf.reduce_mean(y_true)
    gap = alpha / 20
    threshold_pos = 1 - alpha + gap # 1 - alpha + gap / 2
    threshold_neg = 1 - alpha - gap
    target_balance = 0.5
    #pos_percentage *= 2

    neg_neg_normed = neg * target_balance / threshold_neg
    pos_neg_normed = pos * target_balance / threshold_pos
    neg_pos_normed = target_balance + (1 - target_balance) * ((neg - threshold_neg) / (1. - threshold_neg))
    pos_pos_normed = target_balance + (1 - target_balance) * ((pos - threshold_pos) / (1. - threshold_pos))

    neg_rebalanced = tf.where(neg < threshold_neg, neg_neg_normed, neg_pos_normed)
    pos_rebalanced = tf.where(pos < threshold_pos, pos_neg_normed, pos_pos_normed)

    pos_bce = -1 * tf.log(K.epsilon() + pos_rebalanced)
    neg_bce = -1 * tf.log(K.epsilon() + 1 - neg_rebalanced)

    loss = pos_percentage * tf.reduce_mean(pos_bce) + (1 - pos_percentage) * tf.reduce_mean(neg_bce)

    return loss


class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0).reshape(-1)
            score = roc_auc_score(self.y_val, y_pred)
            print("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))


def dense_block(size, input, activation):
    next = l.Dense(size, activation=activation)(input)
    #next = l.Concatenate()([next, input])
    return next


def disc_block(size, input):#l.Dense(size, activation='elu')(input)#
    #next = l.Dense(10 * size, activation='elu')(input)
    next = DiscretizationLayer(size)(input)
    #next = l.Concatenate()([next, input])
    return next


def wide_disc_block(size, input, layer_config):#l.Dense(size, activation='elu')(input)#
    #next = l.Dense(10 * size, activation='elu')(input)
    next = DiscretizationLayerWide(size, layer_config)(input)
    #next = l.ELU()(next)
    return next

def log_block(size, input):
    next = LogarithmLayer(size / 2)(input)
    next = l.Concatenate()([next, input])
    return next


def ras(y_true, y_pred):
    try:
        return np.float32(roc_auc_score(y_true, y_pred))
    except:
        return np.float32(np.mean(y_true))


def auroc(y_true, y_pred):
    return tf.py_func(ras, (y_true, y_pred), tf.float32)


def auroc_tf(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def make_net(ld, lr, configs):
    input = l.Input((200,))
    next = input

    drate = 0.1

    #next = l.BatchNormalization()(next)
    next = wide_disc_block(ld, next, configs.get('disc_layer', {}))
    local_model = Model(input=input, output=next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(drate)(next)
    next = l.Dense(200, activation='elu')(next)
    next = l.Dropout(drate)(next)
    next = l.Dense(200, activation='elu')(next)
    #next = l.BatchNormalization()(next)
    #next = l.Dropout(drate)(next)
    #next = l.Dense(ld, activation='elu')(next)
    #next = l.Dropout(drate)(next)
    #next = l.Dense(ld, activation='elu')(next)
    #next = l.Dropout(drate)(next)
    #next = l.Dense(200, activation='elu')(next)
    #next = l.BatchNormalization()(next)
    #next = l.Dropout(drate)(next)
    #next = l.Dense(400, activation='elu')(next)
    #next = l.BatchNormalization()(next)
    #next = l.Dropout(drate)(next)
    #next = l.Dense(300, activation='elu')(next)
    #next = l.BatchNormalization()(next)
    #next = l.Dropout(drate)(next)

    #next = l.Dense(ld / 2, activation='elu')(next)
    #next = l.Dropout(drate)(next)
    #next = l.Dense(ld / 2, activation='elu')(next)
    #next = l.Dropout(drate)(next)
    #next = l.Dense(ld / 4, activation='elu')(next)
    #next = l.Dropout(drate)(next)
    #next = l.Dense(ld / 8, activation='elu')(next)
    #next = l.Dropout(drate)(next)
    #next = l.Dense(ld / 16, activation='elu')(next)

    next = l.Dense(1, activation='sigmoid')(next)

    model = Model(input=input, output=next)

    model.compile(optimizer=Adam(lr=lr), loss=shifted_bce, metrics=[auroc, 'accuracy'])
    #model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=[auroc, 'accuracy'])

    return model, local_model

def make_net3(ld, lr):
    input = l.Input((200,))
    next = input

    next = dense_block(4 * ld, next, 'elu')
    next = l.BatchNormalization()(next)
    next = l.Dropout(0.1)(next)
    next = l.Dense(2 * ld, activation='elu')(next)
    next = l.Dropout(0.1)(next)
    local_model = Model(input=input, output=next)

    next = l.Dense(1, activation='sigmoid')(next)

    model = Model(input=input, output=next)

    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=[auroc, 'accuracy'])

    return model, local_model

def make_net2(ld, lr):
    input = l.Input((200,))
    next = input

    next = l.BatchNormalization()(next)
    next_1 = LogarithmLayer(ld)(next)
    next_2 = l.Dense(ld, activation='elu')(next)
    next = l.Concatenate()([next_1, next_2])
    next = l.Dense(ld, activation='elu')(next)
    next = l.Dropout(0.2)(next)
    next_1 = LogarithmLayer(ld)(next)
    next_2 = l.Dense(ld, activation='elu')(next)
    next = l.Concatenate()([next_1, next_2])
    next = l.Dense(ld, activation='elu')(next)
    next = l.Dropout(0.2)(next)
    next_1 = LogarithmLayer(ld)(next)
    next_2 = l.Dense(ld, activation='elu')(next)
    next = l.Concatenate()([next_1, next_2])
    next = l.Dense(ld, activation='elu')(next)
    local_model = Model(input=input, output=next)

    next = l.Dense(1, activation='sigmoid')(next)

    model = Model(input=input, output=next)

    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model, local_model