from typing import Union
import numpy as np
from collections import Counter
import os
import re
from nltk import corpus, stem
import pickle

LABELS_MAP = {'sadness': 0, 'love': 1, 'anger': 2, 'joy': 3, 'fear': 4, 'surprise': 5}
WORD_LEMMA = stem.WordNetLemmatizer()
STOPWORDS = corpus.stopwords.words('english')
DATA_PATH = 'data\\'


def load_data(
        path: str,
        label_split: Union[str, None] = None,
        label_first: bool = False
) -> list:
    ds = []
    with open(path) as f:
        for line in f:
            if label_split is None:
                ds.append(line[:-1])
                continue
            tok, label = line[:-1].split(label_split, maxsplit=1)
            ds.append([label, tok] if label_first else [tok, label])
        f.close()
    return ds


def clean_str(
        token: str
) -> str:
    token = re.sub('[^0-9a-zA-Z]', ' ', token)
    token = [WORD_LEMMA.lemmatize(word.lower(), pos='a') for word in token.split() if word not in STOPWORDS]
    return ' '.join(token)


def build_vocabulary(
        tokens: list,
        save: bool = False,
        file_name: str = 'vocabulary',
        glove_wights: bool = False,
        embedding_dim: int = 100
) -> list[np.ndarray]:
    count = []
    for s in tokens: count += s.split()
    counter = Counter(count)
    counter = counter.most_common(len(counter))
    vectorizer_ = {'': 0, '[UNK]': 1}
    for i, (v, _) in enumerate(counter): vectorizer_[v] = i + 2
    wights = None
    if glove_wights:
        assert embedding_dim in [50, 100, 200, 300]
        wights = np.zeros((len(vectorizer_), embedding_dim), dtype='float32')
        found = 0
        with open(f'glove.6B\\glove.6B.{embedding_dim}d.txt', encoding="utf-8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                if not len(word) or word not in vectorizer_: continue
                wights[vectorizer_[word]] = np.fromstring(coefs, "f", sep=" ")
                found += 1
            f.close()
            print(f'Converted {found} words misses {len(vectorizer_) - found + 2}')
    vectorizer_ = np.array(list(vectorizer_.keys()))
    if not save: return [vectorizer_] if not glove_wights else [vectorizer_, wights]

    if os.path.exists(f"data//{file_name}.pkl"): os.remove(f"data//{file_name}.pkl")
    pickle.dump(
        {'weights': [vectorizer_] if not glove_wights else [vectorizer_, wights]},
        open(f"data//{file_name}.pkl", "wb")
    )
    print(f'Vocabulary size: {len(vectorizer_)}')
    return [vectorizer_] if not glove_wights else [vectorizer_, wights]


def load_vocabulary(
        file_name: str = 'vocabulary'
) -> list[np.ndarray]:
    return pickle.load(open(f"data//{file_name}.pkl", "rb"))['weights']
