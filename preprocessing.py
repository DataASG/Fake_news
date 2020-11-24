import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def clean(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')  # Remove Punctuation
    lowercased = text.lower()  # Lower Case
    tokenized = word_tokenize(lowercased)  # Tokenize
    words_only = [word for word in tokenized if word.isalpha()
                  ]  # Remove numbers
    stop_words = set(stopwords.words('english'))  # Make stopword list
    # Remove Stop Words
    without_stopwords = [word for word in words_only if not word in stop_words]
    lemma = WordNetLemmatizer()  # Initiate Lemmatizer
    lemmatized = [lemma.lemmatize(word)
                  for word in without_stopwords]  # Lemmatize
    return lemmatized


# data['clean_text'] = data.text.apply(clean)
# data['clean_text'] = data['clean_text'].astype('str')

