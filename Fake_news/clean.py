import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

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

    X = pd.concat([df_sample_title_joined, df_sample_text_joined], axis=1)
    total_text = X['title'] + ' ' + X['text']

    df_sample_text_list = total_text.to_list()

    return df_sample_text_list

# Model Ini


# def init_model():

#     model = Sequential()
#     model.add(layers.Masking())
#     model.add(Bidirectional(LSTM(256)))
#     model.add(Dense(128, activation='tanh'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer='rmsprop',
#                   loss='binary_crossentropy', metrics=['accuracy'])
#     return model


# df_sample = get_data(0.02)

# df_sample_text_list = cleaning_text(df_sample)

# y = df_sample['label']

# X_train, X_test, y_train, y_test = train_test_split(
#     df_sample_text_list, y, test_size=0.3, random_state=0)

# word2vec = Word2Vec(sentences=X_train, size=60, min_count=10, window=10)
# X_train_pad = embedding_pipeline(word2vec, X_train)
# X_test_pad = embedding_pipeline(word2vec, X_test)

# model = init_model()
# es = EarlyStopping(patience=5, restore_best_weights=True)
# fitted_model = model.fit(X_train_pad, y_train,
#                          batch_size=16,
#                          epochs=5, validation_split=0.1,
#                          callbacks=[es])

# model.evaluate(X_test_pad, y_test)
