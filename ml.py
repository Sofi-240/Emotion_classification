import pandas as pd
import pickle
from data_utils import standardization, LABELS_MAP, loadDFdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


def save_model():
    pipe = Pipeline([
        ("vect", TfidfVectorizer(norm='l1', max_df=0.4, min_df=5, ngram_range=(1, 2), preprocessor=standardization)),
        ("clf", ComplementNB(alpha=1.0))
    ])
    data = pd.concat(
        (loadDFdata('train', clean=False), loadDFdata('val', clean=False)), axis=0
    ).reset_index(drop=True)

    pipe.fit(data['token'], data['label'].map(LABELS_MAP))

    pickle.dump(pipe, open('models_data\\ML_model.pkl', 'wb'))


def load_model() -> Pipeline:
    return pickle.load(open('models_data\\ML_model.pkl', "rb"))
