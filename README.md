# Emotion Detection through NLP: ComplementNB vs. RNN and CNN

[**Refer to the Notebook**](notebook.ipynb)

## **Overview**

This project explores emotion detection using Natural Language Processing (NLP) techniques.
The goal is to compare and contrast three different machine learning models for emotion detection:
Complement Naive Bayes (ComplementNB), Convolutional Neural Networks (CNNs)
with and without GloVe word embeddings, and Recurrent Neural Networks (RNNs) with GloVe embeddings.

## **Dataset**

The dataset contain the flow labels:

<img height="300" src="data\dataset.png" width="400"/>

[You can find the dataset here](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp).

## **Notebook Structure**
The project notebook is structured as follows:

1) Data Processing and Analysis (EDA)
2) ComplementNB Model

<img height="200" src="models_data\ml.png" width="500"/>

3) CNN Model with GloVe Features

<img height="500" src="models_data\cnn_g.png" width="700"/>

4) CNN Model without GloVe Features

<img height="500" src="models_data\cnn.png" width="700"/>

5) RNN (LSTM) Model with GloVe Features

<img height="500" src="models_data\rnn_g.png" width="400"/>

## **Results**:

The mean accuracy results for the training data are as follows:

| Model        | Mean ACC |
|--------------|----------|
| ComplementNB | 0.91     |
| CNN - GloVe  | 0.85     |
| CNN          | 0.91     |
| RNN - GloVe  | 0.92     |



## **Trained Models**:
The trained models used in this project are saved. you can load them as follows:

| Model                          | Loading Function        | Returns                     | Make Predictions             |
|--------------------------------|-------------------------|-----------------------------|------------------------------|
| ComplementNB                   | ml.load_model()         | pipeline (sklearn)          | pipeline.predict(x)          |
| CNN with GloVe features        | net.load_model('cnn_g') | (model, vectorizer) (keras) | model.predict(vectorizer(x)) |
| CNN without GloVe features     | net.load_model('cnn')   | (model, vectorizer) (keras) | model.predict(vectorizer(x)) |
| RNN (LTSM) with GloVe features | net.load_model('rnn_g') | (model, vectorizer) (keras) | model.predict(vectorizer(x)) |







