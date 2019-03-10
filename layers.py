from keras.layers import Layer
from keras.initializers import RandomUniform, RandomNormal, Identity, Constant, TruncatedNormal, Zeros, Ones
from keras.constraints import NonNeg
import keras.backend as K
import keras.layers as l

import numpy as np
import tensorflow as tf

np.random.seed(27)
tf.random.set_random_seed(37)


class DiscretizationLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DiscretizationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        l = -3
        u = -l
        initer = [np.linspace(l, u, self.output_dim).reshape(1, -1) for _ in range(input_shape[1])]
        initer = np.concatenate(initer, axis=0)
        width_val = 4. * float(u - l) / input_shape[1]
        super(DiscretizationLayer, self).build(input_shape)
        self.bins = self.add_weight(name='bins',
                                    shape=(input_shape[1], self.output_dim),
                                    initializer=Constant(initer),
                                    trainable=True)

        self.widths = self.add_weight(name='widths',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=TruncatedNormal(width_val, width_val / 4),
                                      constraint=NonNeg(),
                                      trainable=True)

        self.biases = self.add_weight(name='biases',
                                      shape=(input_shape[1], self.output_dim,),
                                      initializer='glorot_uniform',#TruncatedNormal(0, 1),
                                      trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        x = tf.expand_dims(inputs, -1)
        bins = self.biases - tf.abs(x - self.bins) * self.widths
        bins2prob = tf.nn.softmax(bins)
        x = bins2prob * x
        x = tf.reduce_mean(x, axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim,)


class DiscretizationLayerWide(Layer):
    def __init__(self, output_dim, layer_config, **kwargs):
        self.output_dim = output_dim
        self.layer_config = layer_config
        super(DiscretizationLayerWide, self).__init__(**kwargs)

    def build(self, input_shape):
        u = self.layer_config['bins_init_range']
        l = -u
        bins_init = self.layer_config['bins_init']
        if bins_init == 'linspace':
            initer = [np.linspace(l, u, self.output_dim).reshape(1, -1) for _ in range(input_shape[1])]
            initer = np.concatenate(initer, axis=0)
            init = Constant(initer)
        elif bins_init == 'uniform':
            init = RandomUniform(l, u)
        else:
            raise Exception(bins_init)

        bias_initializer = Constant(self.layer_config['bias_init'])

        width_val = 3. * float(u - l) / input_shape[1]
        super(DiscretizationLayerWide, self).build(input_shape)
        self.bins = self.add_weight(name='bins',
                                    shape=(input_shape[1], self.output_dim),
                                    initializer=init,
                                    trainable=True)

        self.widths = self.add_weight(name='widths',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=TruncatedNormal(width_val, width_val / 4),
                                      constraint=NonNeg(),
                                      trainable=True)

        self.biases = self.add_weight(name='biases',
                                      shape=(input_shape[1], self.output_dim,),
                                      initializer=bias_initializer,
                                      trainable=True)

        self.dense_weight = self.add_weight(name='w',
                                            shape=(input_shape[1], self.output_dim),
                                            initializer='glorot_uniform', # RandomUniform(-1, 1),#
                                            trainable=True)

        self.dense_bias = self.add_weight(name='b',
                                          shape=(input_shape[1],),
                                          initializer=Zeros(),#RandomUniform(-0.1, 0.1),  # 'glorot_uniform',
                                          trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        input = tf.expand_dims(inputs, -1)
        bins = self.biases - tf.abs(input - self.bins) * self.widths
        if self.layer_config['pre_sm_dropout'] > 0.0:
            bins = tf.nn.dropout(bins, keep_prob=1.0 - self.layer_config['pre_sm_dropout'])
        if self.layer_config['softmax']:
                bins2prob = tf.nn.softmax(tf.nn.elu(bins))
                # bins2prob = tf.nn.softmax(bins)
        else:
            bins2prob = bins
        x = bins2prob * self.dense_weight# + self.dense_bias
        x = tf.reduce_sum(x, axis=2) + self.dense_bias
        x = tf.nn.tanh(x)
        #x = x * inputs
        return x


    def compute_output_shape(self, input_shape):
        return input_shape


class LaplaceLayerWide(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(LaplaceLayerWide, self).__init__(**kwargs)

    def build(self, input_shape):
        initer = [np.linspace(-3, 3, self.output_dim).reshape(1, -1) for _ in range(input_shape[1])]
        initer = np.concatenate(initer, axis=0)
        width_val = 4 * 6. /input_shape[1]
        widths = np.sqrt(width_val) * np.ones((input_shape[1], self.output_dim))
        super(LaplaceLayerWide, self).build(input_shape)
        self.bins = self.add_weight(name='bins',
                                    shape=(input_shape[1], self.output_dim),
                                    initializer=Constant(initer),
                                    trainable=True)

        self.widths = self.add_weight(name='widths',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=Constant(widths),#TruncatedNormal(width_val, width_val / 4),
                                      constraint=NonNeg(),
                                      trainable=True)

        self.dense_weight = self.add_weight(name='w',
                                            shape=(input_shape[1], self.output_dim),
                                            initializer='glorot_uniform',
                                            trainable=True)

        #self.dense_bias = self.add_weight(name='b',
        #                                  shape=(self.output_dim,),
        #                                  initializer=Zeros(),#TruncatedNormal(1. / width_val, .25 / width_val),
        #                                  trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        x = tf.expand_dims(inputs, -1)
        x = tf.exp(-1 * tf.abs(x - self.bins) / (K.epsilon() + self.widths)) / (K.epsilon() + 2 * self.widths)
        bins2prob = x / tf.reduce_mean(x, axis=2, keep_dims=True)
        x = bins2prob * self.dense_weight
        x = tf.reduce_sum(x, axis=2)
        #x = tf.nn.tanh(x)
        #x = x * inputs
        return x


    def compute_output_shape(self, input_shape):
        return input_shape


class LogarithmLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(LogarithmLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LogarithmLayer, self).build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=Identity(1. / input_shape[1]),
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer='glorot_uniform',
                                    trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        x = tf.log(1e-5 + tf.abs(inputs))
        x = tf.matmul(x, self.kernel) + self.bias
        x = tf.exp(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim,)



