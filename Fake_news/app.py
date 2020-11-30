

import streamlit as st
from PIL import Image
#import cv2
from streamlit_texts import about, explanation, team, le_wagon, explanation_lewagon #header_team, header_mission
from bokeh.models.widgets import Div

###ideas ####
#st.markdown(
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
    #unsafe_allow_html=True
#)

#########################
#title & image of page#

st.title('Fight against Fake News - The 2020 Fake News Detector')

image_fakenews = Image.open('data/fake_news.jpg')
add_image_fake_news = st.image(image_fakenews, use_column_width=True) #width=300)#use_column_width=True)


st.markdown(":warning: **_ Let's use the deep learning Fake News detector_** :warning:")
st.markdown(':arrow_double_down: Please input the text of the article you would like to check in the grey box below :arrow_double_down:')

############################################
#input text box#

text = st.text_area('', max_chars=2000) #value='Please enter text here (empty textbox before entering text)', height=300


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


if st.sidebar.button('Find out more about le Wagon'):
    js = "window.open('https://www.lewagon.com/')"  # New tab or window
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)
else:
    pass


image_lewagon = Image.open('data/le_wagon_logo.png')
add_image_side_logo = st.sidebar.image(image_lewagon, width=100)#use_column_width=True)



image_alex = Image.open('data/pic_alex.png')
image_annso = Image.open('data/pic_annso.png')
image_felix = Image.open('data/pic_felix.png')
image_jonathan = Image.open('data/pic_jonathan.png')


#add_image_side_annso = st.sidebar.image(image_annso, caption='Ann-Sophie', width=120) #use_column_width=True)
#add_image_side_felix = st.sidebar.image(image_felix, caption='Felix', width=120) #use_column_width=True)
#add_image_side_jonathan = st.sidebar.image(image_jonathan, caption='Jonathan', width=120) #use_column_width=True)

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


