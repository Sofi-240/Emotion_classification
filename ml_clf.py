import pandas as pd
import pickle
from data_utils import load_data, DATA_PATH, clean_str, LABELS_MAP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from viz import sns


def save_model():
    pipe = Pipeline([
        ("vect", TfidfVectorizer(norm='l1', max_df=0.4, min_df=5, ngram_range=(1, 2), preprocessor=clean_str)),
        ("clf", ComplementNB(alpha=1.0))
    ])

    data = load_data(DATA_PATH + 'train.txt', ';') + load_data(DATA_PATH + 'val.txt', ';')
    data = pd.DataFrame(data, columns=['token', 'label'])

    pipe.fit(data['token'], data['label'].map(LABELS_MAP))

    pickle.dump(pipe, open('models_data\\ML_model.pkl', 'wb'))


def load_model() -> Pipeline:
    return pickle.load(open('models_data\\ML_model.pkl', "rb"))


if __name__ == '__main__':
    test_ds = load_data(DATA_PATH + 'test.txt', ';')
    test_ds = pd.DataFrame(test_ds, columns=['token', 'label'])
    y_true = test_ds['label'].map(LABELS_MAP)

    pipeline = load_model()

    y_pred = pipeline.predict(test_ds['token'])

    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(
        conf_mat,
        annot=True,
        fmt='d',
        xticklabels=list(LABELS_MAP.keys()),
        yticklabels=list(LABELS_MAP.keys())
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')