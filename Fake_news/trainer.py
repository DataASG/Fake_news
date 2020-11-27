from Fake_news.data import get_data

import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
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
class Trainer(object):
    ESTIMATOR = 'Bidirectional'
    EXPERIMENT_NAME = 'fake_news_model'

    def __init__(self, X, y, **kwargs):
        self.X_df = X
        self.y_df = y
        self.kwargs = kwargs
        self.batch_size = kwargs.get("batch_size", 16)
        self.epochs = kwargs.get('epochs', 5)
        self.validation_split = kwargs.get('validation_split', 0.1)
        self.patience= kwargs.get('patience', 10)
        self.verbose = kwargs.get('verbose', 0)
        self.test_size=kwargs.get('test_size', 0.3)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                 self.X_df, self.y_df, random_state=3, test_size=self.test_size)


    # def embed_sentence(self, word2vec, sentence):
    #     embedded_sentence = []
    #     for word in sentence:
    #         if word in word2vec.wv:
    #             embedded_sentence.append(word2vec.wv[word])
    #     return np.array(embedded_sentence)


    # def embedding(self, word2vec, sentences):
    #     embed = []
    #     for sentence in sentences:
    #         embedded_sentence = embed_sentence(word2vec, sentence)
    #         embed.append(embedded_sentence)
    #     return embed


    # def embedding_pipeline(self, word2vec, X):
    #     X = embedding(word2vec, X)
    #     X = pad_sequences(X, dtype='float32', padding='post')
    #     return X


    # def word_2_vec(self, X_train, X_test):


    #     return X_train_pad, X_test_pad



    # def init_model():
    #     model = Sequential()
    #     model.add(layers.Masking())
    #     model.add(Bidirectional(LSTM(256)))
    #     model.add(Dense(128, activation='tanh'))
    #     model.add(Dense(1, activation='sigmoid'))
    #     model.compile(optimizer='rmsprop',
    #                   loss='binary_crossentropy', metrics=['accuracy'])
    #     return model



    def train(self):
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

        def init_model():
            model = Sequential()
            model.add(layers.Masking())
            model.add(Bidirectional(LSTM(256)))
            model.add(Dense(128, activation='tanh'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='rmsprop',
                          loss='binary_crossentropy', metrics=['accuracy'])
            return model
        #Embedding and vectorising X_train and X_val
        word2vec = Word2Vec(sentences=self.X_train, size=60, min_count=10, window=10)
        self.X_train_pad = embedding_pipeline(word2vec, self.X_train)
        self.X_val_pad = embedding_pipeline(word2vec, self.X_val)
        # X_train_pad, self.X_val_pad = word_2_vec(self.X_train, self.X_val)
        #initialising the model and the EarlyStopping
        model = init_model()
        es = EarlyStopping(patience=self.patience, restore_best_weights=True)
        #fitting the model to X_train
        print('starting to train')
        self.history = model.fit(self.X_train_pad, self.y_train,
                                    batch_size=self.batch_size,
                                    epochs=self.epochs,
                                    validation_split=self.validation_split,
                                    verbose=self.verbose,
                                    callbacks=[es])



    def evaluate(self):
        # compute_score(self.X_val, self.y_val)
        # y_pred = fitted_model.predict(self.X_test_pad)
        # Returning the accuracy score of the model on the test sets
        accuracy_score = fitted_model.evaluate(self.X_val_pad, self.y_val)
        return accuracy_score[1]




    def save_model(self):
        joblib.dump(self.history, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))
    # def compute_score(self, X_val, y_val):
    #     y_pred = self.history.predict(X_val)#
    #     return y_pred



    # def train(X_train_pad, y_train):
    #     model = init_model()
    #     es = EarlyStopping(patience=5, restore_best_weights=True)
    #     fitted_model = model.fit(X_train_pad, y_train,
    #                              batch_size=16,
    #                              epochs=5, validation_split=0.1,
    #                              callbacks=[es])

    #     return fitted_model






    # Attach ML flow


# def evaluate(X_test_pad, y_test, model):
#     evaluate = model.evaluate(X_test_pad, y_test)
#     return evaluate
    # Attach ML flow


