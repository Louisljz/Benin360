import requests
import streamlit as st
import zipfile
import os
from langchain_openai import ChatOpenAI


def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def read_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Define language codes
language_codes = {
    'French': 'fr',
    'Yoruba': 'yo',
    'English': 'en',
    'Spanish': 'es',
    'Italian': 'it',
    'German': 'de'
}

# Streamlit interface
st.set_page_config(page_title="Introductory Tour Videos", page_icon="📼", layout="wide")
st.subheader('Introductory Tour Video 📸')
st.write('Makes video content more accessible to tourists in multiple languages.')

with st.container():
    uploaded_file = st.file_uploader("📂 Choose a video file", type=['mp4', 'mov', 'avi'])

    if uploaded_file is not None:
        st.video(uploaded_file)
    
    source_language = st.selectbox("🌐 What's the video's language?", ("French", "Yoruba"))
    lang_options = ["English", "Spanish", "Italian", "German", "French"]
    if source_language == "French":
        lang_options.remove("French")
    target_language = st.selectbox("🚩 What's your language?", lang_options)

    dub = st.checkbox('🗣️ Dub video?', value=False)


if st.button('Process Video') and uploaded_file is not None:
    with open("upload.mp4", "wb") as out_file:
        out_file.write(uploaded_file.getvalue())
    with open("upload.mp4", "rb") as in_file:
        data = {'source_language': language_codes[source_language], 
                'target_language': language_codes[target_language], 
                'dub': dub}
        with st.spinner('Processing video...'):
            response = requests.post('http://127.0.0.1:8000/process_video', files={'file': in_file}, data=data)

    if response.status_code == 200:
        # Save the received ZIP file
        zip_path = 'output.zip'
        with open(zip_path, "wb") as f:
            f.write(response.content)

        # Unzip the file
        if not os.path.exists("temp"):
            os.makedirs("temp")
        unzip_file(zip_path, "temp")

        # Read and display the transcript
        transcript_path = os.path.join("temp", "transcript.txt")
        st.session_state.transcript = read_transcript(transcript_path)

        # Read and display the video
        st.session_state.video_path = os.path.join("temp", "download.mp4")
        st.success('Video has been processed successfully!')
        os.remove(zip_path)
        os.remove("upload.mp4")
    else:
        st.error('There was an error processing the video.')

if 'transcript' in st.session_state:
    st.video(st.session_state.video_path)
    query = st.text_input('Ask about the tour video..')
    if query:
        llm = ChatOpenAI(model="gpt-4-turbo")
        messages = [
            ("system", '''You will be given video transcript and your task is to answer user queries based on the given context. 
            Answer in the same language and keep your response concise under 2 sentences.'''),
            ("human", f"Video Transcript: {st.session_state.transcript} \n Question: {query}"),
        ]
        response = llm.invoke(messages).content
        st.write('Answer: ' + response)
