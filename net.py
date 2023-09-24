from typing import Union
import tensorflow as tf
from data_utils import load_vectorizer
from keras.layers import LSTM, Concatenate, Dropout, Dense, Input, Embedding, Flatten, Conv1D, MaxPool1D, Bidirectional, TextVectorization


def cnn(
        vocabulary_size: int,
        sequence_size: int,
        features: int = 512,
        embedding_dim: int = 100,
        n_classes: int = 2,
        kernels_size: Union[None, list, tuple] = None,
        embedding_wights: Union[None, tf.Tensor] = None,
        kernel_initializer=None,
        bias_initializer=None,
        dropout: float = 0.0,
        name: str = 'CnnModel'
) -> tf.keras.Model:
    kernels_size = kernels_size if kernels_size is not None else [3, 4, 5]

    if embedding_wights is not None:
        embeddings_initializer = tf.keras.initializers.Constant(embedding_wights)
    else:
        embeddings_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05)

    embedding = Embedding(
        vocabulary_size,
        embedding_dim,
        embeddings_initializer=embeddings_initializer,
        input_length=sequence_size,
        trainable=embedding_wights is None,
        name='embedding'
    )

    ki_ = tf.keras.initializers.TruncatedNormal
    bi_ = tf.keras.initializers.Zeros

    inputs = Input(shape=[sequence_size, ], dtype=tf.int64)
    X = embedding(inputs)
    conv_pool = []

    for i, k in enumerate(kernels_size):
        x_conv = Conv1D(
            features,
            kernel_size=k,
            padding='VALID',
            activation='relu',
            kernel_initializer=ki_(stddev=0.1) if kernel_initializer is None else kernel_initializer,
            bias_initializer=bi_() if bias_initializer is None else bias_initializer,
            name=f'ConvX{i}'
        )(X)
        x_pool = MaxPool1D(
            pool_size=sequence_size - k + 1,
            name=f'MaxX{i}'
        )(x_conv)
        conv_pool.append(x_pool)

    h_pool = Concatenate(axis=1)(conv_pool)
    h_pool = Flatten()(h_pool)

    if dropout > 0: h_pool = Dropout(dropout)(h_pool)

    outputs = Dense(
        n_classes,
        activation='softmax'
    )(h_pool)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    if embedding_wights is not None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    else:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )
    return model


def rnn(
        vocabulary_size: int,
        sequence_size: int,
        units: Union[list, int] = 256,
        n_cells: int = 3,
        embedding_dim: int = 128,
        embedding_wights: Union[None, tf.Tensor] = None,
        bidirectional: bool = True,
        kernel_initializer=None,
        bias_initializer=None,
        recurrent_initializer=None,
        n_classes: int = 2,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        name: str = 'RnnModel'
) -> tf.keras.Model:
    if embedding_wights is not None:
        embeddings_initializer = tf.keras.initializers.Constant(embedding_wights)
    else:
        embeddings_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05)

    if isinstance(units, int): units = [units] * n_cells

    ki_ = tf.keras.initializers.GlorotUniform
    bi_ = tf.keras.initializers.Zeros
    ri_ = tf.keras.initializers.Orthogonal

    inputs = Input(shape=[sequence_size, ], dtype=tf.int64)

    X = Embedding(
        vocabulary_size,
        embedding_dim,
        embeddings_initializer=embeddings_initializer,
        input_length=sequence_size,
        trainable=embedding_wights is None,
        name='Embedding'
    )(inputs)

    for i, u in enumerate(units):
        cell = LSTM(
            u,
            return_sequences=i != (n_cells - 1),
            activation='tanh',
            recurrent_activation='sigmoid',
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_initializer=ki_() if kernel_initializer is None else kernel_initializer,
            bias_initializer=bi_() if bias_initializer is None else bias_initializer,
            recurrent_initializer=ri_(gain=1.0) if recurrent_initializer is None else recurrent_initializer,
            name=f'LTSM{i}'
        )
        if bidirectional: cell = Bidirectional(cell, name=f'Bi_LTSM{i}')
        X = cell(X)

    outputs = Dense(
        n_classes,
        activation='softmax',
        name='fc_softmax'
    )(X)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=0.01),
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )
    return model


def load_model(
        name: str
) -> tuple[tf.keras.Model, TextVectorization]:
    model_path = {
        'rnn_g': 'models_data\\rnn_glove\\rnn_glove',
        'cnn_g': 'models_data\\cnn_glove\\cnn_glove',
        'cnn': 'models_data\\cnn\\cnn',
    }
    if name not in model_path: raise ValueError(f'model name need to be one of {list(model_path.keys())}')
    return tf.keras.models.load_model(model_path[name] + '_model.keras'), load_vectorizer(model_path[name] + '_vectorizer')

