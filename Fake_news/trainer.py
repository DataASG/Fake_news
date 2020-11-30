from Fake_news.data import get_data

import tensorflow as tf
import numpy as np
import joblib
import mlflow
from  mlflow.tracking import MlflowClient
from gensim.models import Word2Vec
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn import metrics
from tensorflow.keras.callbacks import EarlyStopping
from memoized_property import memoized_property
from xgboost import XGBRegressor
# Mlflow wagon server
MLFLOW_URI = "https://mlflow.lewagon.co/"
# class Trainer(object):

#     def __init__(self, data):
#         self.data = data


# def convert_sentences(X):
#     return [sentence.split(' ') for sentence in X]
class Trainer(object):
    EXPERIMENT_NAME = 'fake_news_model'

    def __init__(self, X, y, **kwargs):
        self.X_df = X
        self.y_df = y
        self.kwargs = kwargs
        self.batch_size = kwargs.get("batch_size", 16)
<<<<<<< HEAD
        self.epochs = kwargs.get('epochs', 5)
        self.validation_split = kwargs.get('validation_split', 0.1)
        self.patience = kwargs.get('patience', 10)
        self.verbose = kwargs.get('verbose', 0)
        self.test_size = kwargs.get('test_size', 0.3)
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.upload = kwargs.get("upload", False)  # if True log info to nlflow
        self.experiment_name = 'fake_news_model-1.1'
=======
        self.epochs = kwargs.get('epochs', 50)
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.upload = kwargs.get("upload", False)  # if True log info to nlflow
        self.max_iter = kwargs.get("max_iter", 50)
        self.test_size = kwargs.get('test_size', 0.3)
        self.experiment_name =  'fake_news_model_ML-1.1'
>>>>>>> 4c9944d5a6bff4d2a821307e3d44fff3b4f47bc1
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_df, self.y_df, random_state=3, test_size=self.test_size)

<<<<<<< HEAD
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
=======
    def get_estimator(self):
        estimator = self.kwargs.get("estimator", 'PassiveAggressive')
        if estimator == "Lasso":
            model = Lasso()
        elif estimator == "Ridge":
            model = Ridge()
        elif estimator == "Linear":
            model = LinearRegression()
        elif estimator == "GBM":
            model = GradientBoostingRegressor()
        elif estimator == "RandomForest":
            model = RandomForestRegressor()
            self.model_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                'max_features': ['auto', 'sqrt']}
            # 'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        elif estimator == 'PassiveAggressive':
            model = PassiveAggressiveClassifier(max_iter=self.max_iter)
        elif estimator == "xgboost":
            model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, max_depth=10, learning_rate=0.05,
                                 gamma=3)
        else:
            model = Lasso()
        estimator_params = self.kwargs.get("estimator_params", {})
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        return model


# Vectorising function


# instantiate + fit model

# save model
>>>>>>> 4c9944d5a6bff4d2a821307e3d44fff3b4f47bc1

    def train(self):
<<<<<<< HEAD
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
        # Embedding and vectorising X_train and X_val
        word2vec = Word2Vec(sentences=self.X_train,
                            size=60, min_count=10, window=10)
        self.X_train_pad = embedding_pipeline(word2vec, self.X_train)
        self.X_val_pad = embedding_pipeline(word2vec, self.X_val)
        # X_train_pad, self.X_val_pad = word_2_vec(self.X_train, self.X_val)
        # initialising the model and the EarlyStopping
        strategy = tf.distribute.MirroredStrategy()

        # Open a strategy scope.
        with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
            model = init_model()
        es = EarlyStopping(patience=self.patience, restore_best_weights=True)
        # fitting the model to X_train
        print('starting to train')
        self.history = model.fit(self.X_train_pad, self.y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_split=self.validation_split,
                                 verbose=self.verbose,
                                 callbacks=[es])
=======
        self.model = self.get_estimator()

        #fitting the model to X_train
        print('starting to train')
        self.history = self.model.fit(self.X_train, self.y_train
                                    )


>>>>>>> 4c9944d5a6bff4d2a821307e3d44fff3b4f47bc1

    def evaluate(self):
        # compute_score(self.X_val, self.y_val)
        # y_pred = fitted_model.predict(self.X_test_pad)
        # Returning the accuracy score of the model on the test sets
<<<<<<< HEAD
        accuracy_score = self.history.evaluate(self.X_val_pad, self.y_val)
        self.mlflow_log_param('model', 'Bidirectional-LSTM')
        self.mlflow_log_metric('accuracy', accuracy_score[1])
        return accuracy_score[1]
=======
        accuracy_score = self.model.score(self.X_val, self.y_val)
        self.mlflow_log_param('model' , 'Bidirectional-LSTM')
        self.mlflow_log_metric('accuracy' ,accuracy_score)
        print(accuracy_score)
        return accuracy_score
>>>>>>> 4c9944d5a6bff4d2a821307e3d44fff3b4f47bc1

    def save_model(self):
        joblib.dump(self.history, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))
    # def compute_score(self, X_val, y_val):
    #     y_pred = self.history.predict(X_val)#
    #     return y_pred

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


    # Attach ML flow


# def evaluate(X_test_pad, y_test, model):
#     evaluate = model.evaluate(X_test_pad, y_test)
#     return evaluate
    # Attach ML flow
