import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd


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


def cleaned_data_ml(df_sample):
    df_sample_text = df_sample['text'].apply(lambda text: clean_func(text))
    df_sample_title = df_sample['title'].apply(lambda text: clean_func(text))

    df_sample_text_joined = df_sample_text.apply(lambda x: " ".join(x))
    df_sample_title_joined = df_sample_title.apply(lambda x: " ".join(x))

    tfidf_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))

    df_text = tfidf_vec.fit_transform(df_sample_text_joined).toarray()
    df_title = tfidf_vec.fit_transform(df_sample_title_joined).toarray()

    X_tfidf = np.hstack((df_title, df_text))

    return X_tfidf
