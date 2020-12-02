from Fake_news.data import get_data
from Fake_news.encoders import TokenizerTransformer, PadSequencesTransformer
from Fake_news.gcp import storage_upload

import os
import pickle
import joblib
import numpy as np
import pandas as pd
import gensim.downloader as api

from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.pipeline import Pipeline
from gensim.models import KeyedVectors
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.preprocessing.sequence import pad_sequences
from memoized_property import memoized_property
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
        self.X = X
        self.y = y
        self.kwargs = kwargs
        self.batch_size = kwargs.get("batch_size", 32)
        self.epochs = kwargs.get('epochs', 20)
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.patience= kwargs.get('patience', 3)
        self.verbose = kwargs.get('verbose', 0)
        self.test_size=kwargs.get('test_size', 0.2)
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.upload = kwargs.get("upload", False)  # if True log info to nlflow
        self.experiment_name =  'fake_news_model-1.1'
        self.vocab_size = 400001
        self.embedding_dim = 50
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                 self.X, self.y, random_state=3, test_size=self.test_size)
        self.longest_text = self.X_train.str.split().str.len().max()


    def set_embedding(self):
        info = api.info()  # show info about available models/datasets
        self.model = api.load("glove-wiki-gigaword-50")

        self.embedding_weights = np.vstack([
           np.zeros(self.model.vectors.shape[1]),
           self.model.vectors
           ])


    def create_model(self, embedding_input_dim, embedding_output_dim, embedding_weights):
        model = Sequential([
            Embedding(input_dim=embedding_input_dim,
                      output_dim=embedding_output_dim,
                      weights=[embedding_weights],
                      trainable=False,
                      mask_zero=True),
            LSTM(256),
            Dense(128)
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def set_pipeline(self):
        my_tokenizer = TokenizerTransformer()
        my_padder = PadSequencesTransformer(maxlen=self.longest_text)
        es = EarlyStopping(monitor="val_loss", min_delta=0, patience=3, verbose=0, mode="auto", restore_best_weights=True)
        my_model = keras.wrappers.scikit_learn.KerasClassifier(
               build_fn=self.create_model,
               epochs=self.epochs,
               batch_size=self.batch_size,
               embedding_input_dim=self.vocab_size,
               embedding_output_dim=self.embedding_dim,
               embedding_weights=self.embedding_weights,
               validation_split=self.validation_split,
               callbacks=[es])

        self.pipeline = Pipeline([
              ('tokenizer', my_tokenizer),
              ('padder', my_padder),
              ('model', my_model)
              ])


    def train(self):
        self.set_embedding()
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y, )




    def evaluate(self):
        # compute_score(self.X_val, self.y_val)
        # y_pred = fitted_model.predict(self.X_test_pad)
        # Returning the accuracy score of the model on the test sets
        accuracy_score = self.pipeline.named_steps['model'].model.evaluate(self.X_val, self.y_val)
        self.mlflow_log_param('model' , 'Bidirectional-LSTM')
        self.mlflow_log_metric('accuracy' ,accuracy_score[1])
        return accuracy_score[1]

    def save_model(self):
        self.pipeline.named_steps['model'].model.save('lstm.h5')
        self.pipeline.named_steps['model'].model = None
        joblib.dump(self.pipeline, 'pipeline.pkl')
        storage_upload(filename='pipeline.pkl')
        storage_upload(filename='lstm.h5')
    # def compute_score(self, X_val, y_val):
    #     y_pred = self.history.predict(X_val)#
    #     return y_pred

    def predict(self, X_test):
        pipeline = joblib.load('pipeline.pkl')
        pipeline.named_steps['model'].model = load_model('lstm.h5')
        return pipeline.predict(X_test)


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    # def train(X_train_pad, y_train):
    #     model = init_model()
    #     es = EarlyStopping(patience=5, restore_best_weights=True)
    #     fitted_model = model.fit(X_train_pad, y_train,
    #                              batch_size=16,
    #                              epochs=5, validation_split=0.1,
    #                              callbacks=[es])

