import pandas as pd
from google.cloud import storage
# from Fake_news.params import #


def get_data(local=False, sample_size=0.005):
    if local:
        path = 'raw_data/data.csv'
    else:
        path = 'gs://fakenews475/data/data.csv'

    df = pd.read_csv(path)
    # df['totalwords'] = [len(x.split()) for x in df['text'].tolist()]
    # df = df.query(f'totalwords<2000')
    df = df.drop(columns=('Unnamed: 0'))
    df_sample = df.sample(frac=sample_size, random_state=3)
    # df_sample.to_csv('df_sample.csv', index=False)
    df_sample = df_sample.reset_index(drop=True)
    # df_sample.drop(columns=('totalwords'), inplace=True)
    return df_sample
