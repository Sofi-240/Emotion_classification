import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.layers import Embedding, TextVectorization
from data_utils import build_vocabulary, load_vocabulary, load_data, DATA_PATH, clean_str, LABELS_MAP
from viz import sns

BUFFER_SIZE = 1000
BATCH_SIZE = 64
EMBEDDING_DIM = 100
SEQUENCE_SIZE = 80
AUTOTUNE = tf.data.AUTOTUNE
VOCAB_SIZE = None


data = load_data(DATA_PATH + 'train.txt', label_split=';', label_first=False)
data = pd.DataFrame(data, columns=['token', 'label'])
data['token'] = data['token'].apply(clean_str)
# y = pd.get_dummies(ds['label'].map(LABELS_MAP), dtype='float32')

vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE, output_mode='int', split='whitespace', output_sequence_length=SEQUENCE_SIZE, ngrams=1
)
vectorize_layer.adapt(data['token'])

vocabulary = vectorize_layer.get_vocabulary()

# vocabulary, embedding_init = build_vocabulary(
#     list(data['token']), save=True, file_name='vocabulary', glove_wights=True, embedding_dim=EMBEDDING_DIM
# )


# vocabulary, _ = load_vocabulary(file_name='vocabulary')

# vectorize_layer = TextVectorization(
#     max_tokens=VOCAB_SIZE, output_mode='int', split='whitespace',
#     vocabulary=vocabulary[2:], output_sequence_length=SEQUENCE_SIZE
# )
#     inputs = Embedding(
#         VOCAB_SIZE, EMBEDDING_DIM,
#         embeddings_initializer=tf.keras.initializers.Constant(EMBEDDINGS_WIGHTS),
#         input_length=SEQUENCE_SIZE, trainable=False
#     )(inputs)