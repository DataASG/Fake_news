import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Data collection
# def get_data(sample_size):

#     df = pd.read_csv('../raw_data/wagon_data/data.csv')
#     df_sample = df.sample(frac=sample_size, random_state=3)
#     df_sample.to_csv('df_sample.csv', index=False)
#     df_sample = df_sample.reset_index(drop=True)
#     return df_sample


# Preprocessing
def clean_func(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    lowercased = text.lower()
    tokenized = word_tokenize(lowercased)
    words_only = [word for word in tokenized if word.isalpha()
                  ]
    stop_words = set(stopwords.words('english'))

    without_stopwords = [word for word in words_only if not word in stop_words]
    lemma = WordNetLemmatizer()
    lemmatized = [lemma.lemmatize(word)
                  for word in without_stopwords]
    return lemmatized


def cleaned_data(df_sample):
    df_sample_text = df_sample['text'].apply(lambda text: clean_func(text))
    df_sample_title = df_sample['title'].apply(lambda text: clean_func(text))

    df_sample_text_joined = df_sample_text.apply(lambda x: " ".join(x))
    df_sample_title_joined = df_sample_title.apply(lambda x: " ".join(x))

    # X = pd.concat([df_sample_title_joined, df_sample_text_joined], axis=1)
    # total_text = X['title'] + ' ' + X['text']

    # df_sample_text_list = total_text()

    return df_sample_text_joined, df_sample_title_joined

def vectoriser(df_text, df_title):
    tfidf_vec = TfidfVectorizer(max_features=10000, ngram_range=(1,3))
    df_text = tfidf_vec.fit_transform(df_text).toarray()
    df_title = tfidf_vec.fit_transform(df_title).toarray()
    X_vec = np.hstack((df_title, df_text))
    return X_vec
