#! -*- coding:utf-8 -*-
from keras.layers import *
import keras.backend as K


class AttentionPooling1D(Layer):
    def __init__(self, h_dim=None, **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.h_dim = h_dim

    def build(self, input_shape):
        super(AttentionPooling1D, self).build(input_shape)
        if not self.h_dim:
            self.h_dim = input_shape[-1]

        self.k_dense = Dense(self.h_dim, use_bias=False, activation='tanh')
        self.o_dense = Dense(1, use_bias=False)

    def call(self, x, mask=None):
        x0 = x
        x = self.k_dense(x0)
        x = self.o_dense(x)
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, 2)
            x = x - (1 - mask) * 1e12
        x = K.softmax(x, 1)
        x = K.sum(x0 * x, 1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        base_config = super(AttentionPooling1D, self).get_config()
        config = {'h_dim': self.h_dim}
        return dict(list(base_config.items()) + list(config.items()))


class DGCNN(Layer):
    """
    膨胀卷积网络，优势是本质是一个CNN，所以速度上比RNNs快，同时通过不同的膨胀系数，如【1，3，5，8】可以来整合全局信息，
    此外，与残差网络结合，解决梯度消失问题，并让信息在多个通道流通。所以在处理序列数据时可以抛弃RNNs而尝试使用该结构。
    ref: https://spaces.ac.cn/archives/5409
    """

    def __init__(self, o_dim=None, k_size=3, dilation_rate=1, skip_connection=True, dropout_rate=None, **kwargs):
        super(DGCNN, self).__init__(**kwargs)

        self.o_dim = o_dim
        self.k_size = k_size
        self.dilation_rate = dilation_rate
        self.skip_connection = skip_connection
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(DGCNN, self).build(input_shape)
        if self.o_dim is None:
            self.o_dim = input_shape[-1]
        self.conv1d = Conv1D(
            self.o_dim * 2,
            self.k_size,
            dilation_rate=self.dilation_rate,
            padding='same',
            name='dgcnn_conv1d'
        )
        if self.skip_connection and self.o_dim != input_shape[-1]:
            self.conv1d_1x1 = Conv1D(self.o_dim, 1)

    def call(self, x, mask=None):
        x0 = x
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, 2)
        #         x = x0 * mask if mask is not None else x0
        x0 = Lambda(lambda x_: x_, output_shape=lambda s: s)(x0)  # drop mask so do not put mask to conv1d
        x = self.conv1d(x0)
        x, g = x[:, :, :self.o_dim], x[:, :, self.o_dim:]
        if self.dropout_rate is not None:
            g = K.in_train_phase(K.dropout(g, self.dropout_rate), g)
        g = K.sigmoid(g)
        # mask is none
        mask = mask if mask is not None else K.ones_like(x)

        if self.skip_connection:
            if K.int_shape(x0)[-1] != self.o_dim:
                x0 = self.conv1d_1x1(x0)
            return (x0 * (1 - g) + x * g) * mask
        return x * g * mask

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.o_dim,)

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self):
        base_config = super(DGCNN, self).get_config()
        config = {
            'o_dim': self.o_dim,
            'k_size': self.k_size,
            'dilation_rate': self.dilation_rate,
            'skip_connection': self.skip_connection,
            'dropout_rate': self.dropout_rate
        }
        return dict(list(base_config.items()) + list(config.items()))
