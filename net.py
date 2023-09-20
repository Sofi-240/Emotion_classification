from typing import Union
import tensorflow as tf
from keras.layers import LSTM, Concatenate, Dropout, Dense, Input, Conv2D, MaxPool2D, Reshape
from keras.models import Model


def cnn(
        features: int = 128,
        sequence_size: int = 150,
        embedding_dim: int = 100,
        n_classes: int = 2,
        kernel_size: Union[None, list, tuple] = None,
        kernel_initializer=None,
        bias_initializer=None,
        dropout: float = 0.0,
        name: str = 'CnnModel'
):
    if kernel_size is None: kernel_size = [2, 3, 4]
    ki_ = tf.keras.initializers.TruncatedNormal
    bi_ = tf.keras.initializers.Constant

    inputs = Input(shape=(sequence_size, embedding_dim, 1), name='X')

    conv_pool = []

    for i, k in enumerate(kernel_size):
        x_conv = Conv2D(
            features,
            kernel_size=(k, embedding_dim),
            padding='VALID',
            ctivation='relu',
            kernel_initializer=ki_(stddev=0.1) if kernel_initializer is None else kernel_initializer,
            bias_initializer=bi_(value=0.1) if bias_initializer is None else bias_initializer,
            name=f'ConvX{i}'
        )(inputs)
        x_pool = MaxPool2D(
            pool_size=(sequence_size - k + 1, 1),
            name=f'MaxX{i}'
        )(x_conv)
        conv_pool.append(x_pool)

    h_pool = Concatenate(axis=3)(conv_pool)
    h_pool = Reshape((-1, len(kernel_size) * features, 1))(h_pool)

    if dropout > 0: h_pool = Dropout(dropout)(h_pool)

    outputs = Dense(
        n_classes,
        activation='softmax',
        kernel_initializer=ki_(stddev=0.1) if kernel_initializer is None else kernel_initializer,
        bias_initializer=bi_(value=0.1) if bias_initializer is None else bias_initializer,
    )(h_pool)

    return Model(inputs=inputs, outputs=outputs, name=name)
