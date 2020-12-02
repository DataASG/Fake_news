from Fake_news.clean import cleaned_data

from gensim.models import Word2Vec

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from streamlit_texts import about, explanation, team, le_wagon, explanation_lewagon, alex, Jonathan, annso, felix
from bokeh.models.widgets import Div
from tensorflow import keras
import base64

model = keras.models.load_model('./LSTM')


def embed_sentence(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        if word in word2vec.wv:
            embedded_sentence.append(word2vec.wv[word])
    return np.array(embedded_sentence)


def embedding(word2vec, sentences):
    embed = []
    for sentence in sentences:
        embedded_sentence = embed_sentence(word2vec, sentence)
        embed.append(embedded_sentence)
    return embed


def embedding_pipeline(word2vec, X):
    X = embedding(word2vec, X)
    X = pad_sequences(X, dtype='float32', padding='post')
    return X


def predict(title, text):
    d = {'title': title, 'text': text, 'dummy': 'dumb'}
    X = pd.DataFrame(data=d, index=[0])
    X = clean_data(X)
    word2vec = Word2Vec(sentences=X, size=60, min_count=10, window=10)
    X = embedding_pipeline(word2vec, X)
    prediction = model.predict(X)
    return prediction


# def get_model():
#     # Create a simple model.
#     inputs = keras.Input(shape=(32,))
#     outputs = keras.layers.Dense(1)(inputs)
#     model = keras.Model(inputs, outputs)
#     model.compile(optimizer="adam", loss="mean_squared_error")
#     return model


# model = get_model()
# # Train the model.
# test_input = np.random.random((128, 32))
# test_target = np.random.random((128, 1))
# model.fit(test_input, test_target)
# # Calling `save('my_model')` creates a SavedModel folder `my_model`.
# model.save("my_model")
# # It can be used to reconstruct the model identically.
# reconstructed_model = keras.models.load_model("my_model")

###ideas ####
# st.markdown(
#"""
#<style>
#.reportview-container {
#background: url("url_goes_here")
#}
#.sidebar .sidebar-content {
#background: url("url_goes_here")
#}
#</style>
#"""#,
# unsafe_allow_html=True
#)

#########################
#title & image of page#

st.title('Fight against Fake News - The 2020 Fake News Detector')

image_fakenews = Image.open('Fake_news/data/fact.jpg')
# width=300)#use_column_width=True)
add_image_fake_news = st.image(image_fakenews, use_column_width=True)


st.markdown(":warning: **_ Let's use the deep learning Fake News detector_** :warning:    \n    \n:white_check_mark: 9/10 times it can correctly identify Fake News articles :white_check_mark:")
st.markdown(':arrow_double_down: Please input the title of the article you would like to check in the grey box below :arrow_double_down:')
############################################

###ideas to input background ####
import base64

main_bg = "Fake_news/data/e0ecf4.png"
main_bg_ext = "png"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)


############################################

#input text box#

# value='Please enter text here (empty textbox before entering text)', height=300
title = st.text_area('', max_chars=100)

st.markdown(':arrow_double_down: Please input the text of the article you would like to check in the grey box below :arrow_double_down:')
text = st.text_area('', max_chars=6000)

if st.button('Predict'):
    prediction = predict(title, text)
    st.write(prediction)
else:
    pass


if st.button('How does it work?'):
    st.write(explanation, height=200)
else:
    pass


#########################################
#sidebar#

add_selectbox = st.sidebar.header("Our Mission")

add_selectbox = st.sidebar.write(
    about)
add_selectbox = st.sidebar.header("The Journey")
add_selectbox = st.sidebar.write(
    team)

image_alex = Image.open('Fake_news/data/pic_alex.png')
image_annso = Image.open('Fake_news/data/pic_annso.png')
image_felix = Image.open('Fake_news/data/pic_felix.png')
image_jonathan = Image.open('Fake_news/data/pic_jonathan.png')


# add_image_side_annso = st.sidebar.image(image_annso, caption='Ann-Sophie', width=120) #use_column_width=True)
# add_image_side_felix = st.sidebar.image(image_felix, caption='Felix', width=120) #use_column_width=True)
# add_image_side_jonathan = st.sidebar.image(image_jonathan, caption='Jonathan', width=120) #use_column_width=True)

col1, col2, col3, col4 = st.sidebar.beta_columns(4)
with col1:
    st.sidebar.header("Team Members")
    st.sidebar.image(image_alex, width=120)
    add_selectbox = st.sidebar.write(
        alex)
with col2:
    st.sidebar.image(image_annso, width=120)
    add_selectbox = st.sidebar.write(
        annso)
with col3:
    st.sidebar.image(image_felix, width=120)
    add_selectbox = st.sidebar.write(
        felix)
with col4:
    # caption='Jonathan', width=120)
    st.sidebar.image(image_jonathan, width=120)
    add_selectbox = st.sidebar.write(
        Jonathan)
    #st.image(image_alex, caption='Team members', width=None, use_column_width=False, clamp=False, channels='RGB', output_format='auto')

if st.sidebar.button('Find out more about le Wagon'):
    js = "window.open('https://www.lewagon.com/')"  # New tab or window
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)
else:
    pass

image_lewagon = Image.open('Fake_news/data/le_wagon_logo.png')
add_image_side_logo = st.sidebar.image(
    image_lewagon, width=100)  # use_column_width=True)
