import streamlit as st
from streamlit_texts import about, explanation

st.title('Fake News Detector')

st.markdown('This is our **_Fake News Detector_**')


add_selectbox = st.sidebar.write(
    about)

text = st.text_area('Please input the text of the article you would like to check here',
    value='Please enter text here', height=400, max_chars=1000)


if st.button('How does it work?'):
    st.write(explanation)
else:
    pass

