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
        self.epochs = kwargs.get('epochs', 50)
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.upload = kwargs.get("upload", False)  # if True log info to nlflow
        self.max_iter = kwargs.get("max_iter", 50)
        self.test_size = kwargs.get('test_size', 0.3)
        self.experiment_name =  'fake_news_model_ML-1.1'
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                 self.X_df, self.y_df, random_state=3, test_size=self.test_size)


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



    def train(self):
        self.model = self.get_estimator()

        #fitting the model to X_train
        print('starting to train')
        self.history = self.model.fit(self.X_train, self.y_train
                                    )



    def evaluate(self):
        # compute_score(self.X_val, self.y_val)
        # y_pred = fitted_model.predict(self.X_test_pad)
        # Returning the accuracy score of the model on the test sets
        accuracy_score = self.model.score(self.X_val, self.y_val)
        self.mlflow_log_param('model' , 'Bidirectional-LSTM')
        self.mlflow_log_metric('accuracy' ,accuracy_score)
        print(accuracy_score)
        return accuracy_score

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


