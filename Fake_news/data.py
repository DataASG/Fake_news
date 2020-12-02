import pandas as pd
from google.cloud import storage
# from Fake_news.params import #


def get_data(local=False, sample_size=1, nrows=None):
    if local:
        path = 'raw_data/data.csv'
    else:
        path = 'gs://fakenews475/data/data.csv'

    df = pd.read_csv(path, nrows=nrows)
    df = df.drop(columns=('Unnamed: 0'))
    df = df.drop(columns=('title'))
    # df_sample = df.sample(frac=sample_size, random_state=3)
    # df_sample.to_csv('df_sample.csv', index=False)
    # df = df_sample.reset_index(drop=True)
    X = df.text
    y = df.label
    return X, y
