from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report

from Fake_news.ML_train import Trainer
from Fake_news.ML_clean import cleaned_data_ml

import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Step 0 ---> Set params
    # Step 1 ---> Get Data
    print('getting data')
    df = get_data(local=True, sample_size=0.3, ML=True)
    y = df.pop('label')
    # Step 2 ---> Clean Data
    print('cleaning_data')
    X = cleaned_data_ml(df)
    del df
    # Step 2 1/2 ---> Split the model in X and y
    # Step 3 ---> Calling the trainer class
    print('calling trainer Class')
    t = Trainer(X=X, y=y)
    del X, y
    print('starting to train model')
    t.train()
    print('finished training, evaluating model')
    t.evaluate()
    print('Saving the model')
    # Save the model
