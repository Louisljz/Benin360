import requests
import streamlit as st

# Define language codes
language_codes = {
    'English': 'en',
    'Spanish': 'es',
    'Italian': 'it',
    'German': 'de'
}

# Streamlit interface
st.subheader('Introductory Tour Video 📸')
st.write('Makes video content more accessible to tourists in multiple languages.')

with st.container():
    uploaded_file = st.file_uploader("📂 Choose a video file", type=['mp4', 'mov', 'avi'])

    if uploaded_file is not None:
        st.video(uploaded_file)
    
    target_language = st.selectbox("🚩 What's your language?", ("English", "Spanish", "Italian", "German"))

    dub = st.checkbox('🗣️ Dub video?', value=False)


if st.button('Process Video') and uploaded_file is not None:
    with open("upload.mp4", "wb") as out_file:
        out_file.write(uploaded_file.getvalue())
    with open("upload.mp4", "rb") as in_file:
        data = {'target_language': language_codes[target_language], 'dub': dub}
        response = requests.post(st.secrets['API_URL'], files={'file': in_file}, data=data)

    if response.status_code == 200:
        st.success('Video has been processed successfully!')
        st.video(response.content)
    else:
        st.error('There was an error processing the video.')
