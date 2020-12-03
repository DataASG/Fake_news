#from Fake_news.clean import cleaned_data
#from gensim.models import Word2Vec
import streamlit as st
#import numpy as np
from PIL import Image
#import pandas as pd
from streamlit_texts import about, explanation, team, le_wagon, explanation_lewagon, alex, Jonathan, annso, felix
from bokeh.models.widgets import Div
#from tensorflow import keras
import base64
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import gensim.downloader as api
from Fake_news.encoders import TokenizerTransformer, PadSequencesTransformer
from Fake_news.gcp import storage_upload
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.pipeline import Pipeline
from gensim.models import KeyedVectors
from Fake_news.clean import cleaned_data
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
@st.cache(allow_output_mutation=True)
def set_model():
    model = joblib.load('test_ben/models_LSTM_versions_10.2_pipeline.pkl')
    model.named_steps['model'].model = load_model('test_ben/models_LSTM_versions_10.2_lstm.h5')
    # model._make_predict_function()
    # model.summary()  # included to make it visible when model is reloaded
    return model
pipeline = set_model()
st.title('Fight against Fake News - The 2020 Fake News Detector')
image_fakenews = Image.open('Fake_news/data/fact.jpg')
image_predict_fake_news = Image.open('Fake_news/data/fake_news_2.jpg')
image_predict_real_news = Image.open('Fake_news/data/good_news.jpg')
# width=300)#use_column_width=True)
add_image_fake_news = st.image(image_fakenews, use_column_width=True)
st.markdown(":warning: **_ Let's use the deep learning Fake News detector_** :warning:    \n    \n:white_check_mark: We will correctly identify real from fake news articles 9/10 times :white_check_mark:")
#st.markdown(':arrow_double_down: Please input the title of the article you would like to check in the grey box below :arrow_double_down:')
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
#title = st.text_area('', max_chars=100)
st.markdown(':arrow_double_down: Please input the text of the article you would like to check in the grey box below :arrow_double_down:')
text = st.text_area('', max_chars=6000)
if st.button('Predict'):
    test = pipeline.predict([text])
    if test[0] == 0:
        response = 'Great, all clues seems to indicate this is real news. Your sources must to be good. Carry on!'
        add_image_predict_real_news = st.image(image_predict_real_news, use_column_width=True) # image after predicitng real news
    else:
        response = 'Be careful, there is a high chance this is fake news. This article seems to have been made to decieve you. Check your sources again!'
        add_image_predict_fake_news = st.image(image_predict_fake_news, use_column_width=True) # image after predicting fake news
    st.write(response)
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
