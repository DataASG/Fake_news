import pandas as pd
from sklearn.model_selection import train_test_split
from Fake_news.encoders import TokenizerTransformer, PadSequencesTransformer
import joblib
from tensorflow.keras.models import load_model

def get_data(local=False, sample_size=1, nrows=None):
    if local:
        path = '../raw_data/data.csv'
    else:
        path = 'gs://fakenews475/data/data.csv'

    df = pd.read_csv(path, nrows=nrows)
    mask = (df['text'].str.len() < 6000)
    df = df.loc[mask]
    df = df.drop(columns=('Unnamed: 0'))
    df = df.drop(columns=('title'))
    # df_sample = df.sample(frac=sample_size, random_state=3)
    # df_sample.to_csv('df_sample.csv', index=False)
    # df = df_sample.reset_index(drop=True)
    X = df.text
    y = df.label
    return X, y



def predict(X_test):
    pipeline = joblib.load('../pipeline.pkl')
    pipeline.named_steps['model'].model = load_model('../lstm.h5')
    return pipeline.predict(X_test)



X,y = get_data(local=True, sample_size=0.3, nrows=None)
X_train, X_val, y_train,y_val = train_test_split(
                 X, y, random_state=3, test_size= 0.3)
prediction = predict(X_val)
print(prediction)




