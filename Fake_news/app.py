from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
#import cv2
# header_team, header_mission
from streamlit_texts import about, explanation, team, le_wagon, explanation_lewagon
from bokeh.models.widgets import Div
from tensorflow import keras
model = keras.models.load_model('./LSTM')


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

image_fakenews = Image.open('Fake_news/data/fake_news.jpg')
# width=300)#use_column_width=True)
add_image_fake_news = st.image(image_fakenews, use_column_width=True)


st.markdown(
    ":warning: **_ Let's use the deep learning Fake News detector_** :warning:")
st.markdown(':arrow_double_down: Please input the title of the article you would like to check in the grey box below :arrow_double_down:')

############################################
#input text box#

# value='Please enter text here (empty textbox before entering text)', height=300
title = st.text_area('', max_chars=100)

st.markdown(':arrow_double_down: Please input the text of the article you would like to check in the grey box below :arrow_double_down:')
text = st.text_area('', max_chars=2000)

if st.button('Predict'):
    print(predict(title, text))
else:
    pass


if st.button('How does it work?'):
    st.write(explanation, height=200)
else:
    pass


def predict(title, text):
    d = {'title': title, 'text': text}
    X = pd.DataFrame(data=d)
    prediction = model.predict(X)
    return prediction


#########################################
#sidebar#


add_selectbox = st.sidebar.header("Our Mission")

add_selectbox = st.sidebar.write(
    about)


add_selectbox = st.sidebar.header("The Journey")


add_selectbox = st.sidebar.write(
    team)


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
    st.sidebar.image(image_alex, caption='Alexander', width=120)
with col2:
    st.sidebar.image(image_annso, caption='Ann-Sophie', width=120)
with col3:
    st.sidebar.image(image_felix, caption='Felix', width=120)
with col4:
    st.sidebar.image(image_jonathan, caption='Jonathan', width=120)

#st.image(image_alex, caption='Team members', width=None, use_column_width=False, clamp=False, channels='RGB', output_format='auto')


############################################
