from typing import Union
import numpy as np
import tensorflow as tf
import pandas as pd
import re
from nltk import corpus, stem
from keras.layers import TextVectorization
import pickle

LABELS_MAP = {'sadness': 0, 'love': 1, 'anger': 2, 'joy': 3, 'fear': 4, 'surprise': 5}
WORD_LEMMA = stem.WordNetLemmatizer()
STOPWORDS = corpus.stopwords.words('english')
DATA_PATH = 'data\\'


def load_vocabulary_data(
        path: str = 'train',
        clean: bool = True
) -> list:
    voc_data = []
    with open(f'data\\{path}.txt') as f:
        for line in f:
            tok, _ = line[:-1].split(';')
            if clean: tok = standardization(tok)
            voc_data.append(tok)
        f.close()
    return voc_data


def standardization(
        token: str
) -> str:
    token = re.sub('[^0-9a-zA-Z]', ' ', token)
    token = [WORD_LEMMA.lemmatize(word.lower(), pos='a') for word in token.split() if word not in STOPWORDS]
    return ' '.join(token)


def loadDFdata(
        path: str = 'train',
        clean: bool = False
) -> pd.DataFrame:
    ds = []
    with open(f'{DATA_PATH}{path}.txt') as f:
        for line in f:
            tok, label = line[:-1].split(';', maxsplit=1)
            ds.append([tok, label])
        f.close()
    ds = pd.DataFrame(ds, columns=['token', 'label'])
    if not clean: return ds
    ds['token'] = ds['token'].apply(standardization)
    return ds


def loadTFdata(
        path: str = 'train',
        clean: bool = True,
        vectorizer: Union[TextVectorization, None] = None
) -> tf.data.Dataset:
    ds = loadDFdata(path, clean=clean)
    ds['label'] = ds['label'].map(LABELS_MAP)
    x = ds['token']
    y = tf.keras.utils.to_categorical(ds['label'])
    if vectorizer is not None: x = vectorizer(x)
    tf_ds = tf.data.Dataset.from_tensor_slices((x, y), path)
    return tf_ds


def build_glove_wights(
        vocabulary: Union[list, TextVectorization],
        embedding_dim: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(vocabulary, TextVectorization):
        vocabulary_ = vocabulary.get_vocabulary()[2:]
    elif isinstance(vocabulary, list):
        s_ = 0
        if vocabulary[0] == '': s_ += 1
        if vocabulary[1] == '[UNK]': s_ += 1
        vocabulary_ = vocabulary[s_:]
    else:
        raise ValueError(f'vocabulary need to type of list or TextVectorization')
    if embedding_dim not in [50, 100, 200, 300]:
        raise ValueError('embedding dim need to be 50/100/200 or 300')

    vocabulary_map = {v: i + 2 for i, v in enumerate(vocabulary_)}
    glove_wights = np.zeros((len(vocabulary_map) + 2, embedding_dim), dtype='float32')
    found_map = np.zeros((len(vocabulary_map) + 2,), dtype=bool)

    with open(f'glove.6B\\glove.6B.{embedding_dim}d.txt', encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            if not len(word) or word not in vocabulary_map: continue
            idx = vocabulary_map[word]
            glove_wights[idx] = np.fromstring(coefs, "f", sep=" ")
            del vocabulary_map[word]
            found_map[idx] = True
        f.close()
    print(f'Converted {found_map.sum()} words misses {len(vocabulary_map)}')
    return glove_wights, found_map


def save_vectorizer(
        vectorizer: TextVectorization,
        path: str
):
    pickle.dump(
        {
            'config': vectorizer.get_config(),
            'weights': vectorizer.get_weights()
        },
        open(f"{path}.pkl", "wb")
    )


def load_vectorizer(
        path: str
) -> TextVectorization:
    config = pickle.load(open(f"{path}.pkl", "rb"))
    vectorizer = TextVectorization.from_config(config['config'])
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    vectorizer.set_weights(config['weights'])
    return vectorizer


def get_vectorizer(
        max_tokens: Union[int, None],
        output_sequence_length: int,
        ngrams: Union[int, tuple],
        output_mode: str = 'int',
        split: str = 'whitespace',
        adapt: str = 'train'
) -> TextVectorization:
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode=output_mode,
        split=split,
        output_sequence_length=output_sequence_length,
        ngrams=ngrams
    )
    vectorizer.adapt(load_vocabulary_data(adapt, clean=True))
    return vectorizer
