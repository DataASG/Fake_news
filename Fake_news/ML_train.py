from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np


class Trainer(object):
    def __init__(self, X, y, **kwargs):
        self.X_df = X
        self.y_df = y
        self.kwargs = kwargs
        self.max_iter = kwargs.get("max_iter", 50)
        self.max_features = kwargs.get("max_features", 10000)
        self.ngram_range = kwargs.get("ngram_range", (1, 3))
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_df, self.y_df, random_state=3, test_size=self.test_size)

    def train(self):
        linear_clf = PassiveAggressiveClassifier(max_iter=self.max_iter)
        self.model = linear_clf.fit(self.X_train, self.y_train)

    def evaluate(self):
        pred = self.model.predict(self.X_val)
        print(classification_report(y_val, pred))

    def save_model(self):
        joblib.dump(self.model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))
