from Fake_news.data import get_data

import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping

# Mlflow wagon server
MLFLOW_URI = "https://mlflow.lewagon.co/"

# class Trainer(object):

#     def __init__(self, data):
#         self.data = data


# def convert_sentences(X):
#     return [sentence.split(' ') for sentence in X]


def embed_sentence(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        if word in word2vec.wv:
            embedded_sentence.append(word2vec.wv[word])
    return np.array(embedded_sentence)


def embedding(word2vec, sentences):
    embed = []
    for sentence in sentences:
        embedded_sentence = embed_sentence(word2vec, sentence)
        embed.append(embedded_sentence)
    return embed


def embedding_pipeline(word2vec, X):
    X = embedding(word2vec, X)
    X = pad_sequences(X, dtype='float32', padding='post')
    return X


def word_2_vec(X_train, X_test):
    word2vec = Word2Vec(sentences=X_train, size=60, min_count=10, window=10)
    X_train_pad = embedding_pipeline(word2vec, X_train)
    X_test_pad = embedding_pipeline(word2vec, X_test)
    return X_train_pad, X_test_pad


def init_model():
    model = Sequential()
    model.add(layers.Masking())
    model.add(Bidirectional(LSTM(256)))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train(X_train_pad, y_train):
    model = init_model()
    es = EarlyStopping(patience=5, restore_best_weights=True)
    fitted_model = model.fit(X_train_pad, y_train,
                             batch_size=16,
                             epochs=5, validation_split=0.1,
                             callbacks=[es])

    return fitted_model
    # Attach ML flow


def evaluate(X_test_pad, y_test, model):
    evaluate = model.evaluate(X_test_pad, y_test)
    return evaluate
    # Attach ML flow

# def save():
