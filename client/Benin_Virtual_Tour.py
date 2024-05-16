import streamlit as st
from audio_recorder_streamlit import audio_recorder

import os
import requests
from openai import OpenAI

import langchain
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain_community.retrievers import WikipediaRetriever
from langchain.agents import AgentExecutor, create_openai_tools_agent


langchain.debug = True
client = OpenAI()
st.set_page_config(page_title="Benin Tourism", page_icon="🌍", layout="wide")

@st.cache_data
def download_prompt():
    return hub.pull("hwchase17/openai-tools-agent")

def translate_text(text, source_lang):
    url = 'https://translation.googleapis.com/language/translate/v2'
    params = {
        'q': text,
        'source': source_lang,
        'target': 'en',
        'key': st.secrets['GCLOUD_API_KEY']
    }
    response = requests.get(url, params=params)
    return response.json()['data']['translations'][0]['translatedText']

def transcribe(audio_file):
    transcription = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )
    return transcription.text

def text_to_speech(text):
    response = client.audio.speech.create(model="tts-1", voice="shimmer", input=text)
    return response.content

def setup_retriever(rag_retriever):
    wiki_retriever = WikipediaRetriever()
    wiki_tool = create_retriever_tool(
        wiki_retriever,
        "explore_benin",
        "search for information about Benin. include 'benin' in your wikipedia search query",
    )
    site = st.session_state.tourist_site.replace(' ', '_')
    rag_tool = create_retriever_tool(
        rag_retriever,
        f"explore_{site}",
        f"search for information about {site}.",
    )

    tools = [wiki_tool, rag_tool]

    llm = ChatOpenAI(model='gpt-4-turbo')
    prompt = download_prompt()
    system_messages = '''
    You are a tour guide assistant, designed to provide tourists with info about Benin and its popular tourist sites.
    Please respond concisely, and write in 2 sentences maximum. answer in the same language as the user question.
    If the information retrieved is misleading, you may use your own knowledge to answer, or say "I don't know".
    '''
    prompt.messages[0].prompt.template = system_messages

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    return agent_executor


tourist_sites_benin = [
    "Pendjari National Park",
    "Porto-Novo Museum",
    "Lake Nokoue",
    "Fidjrosse Beach",
    "The Cathedral of Notre Dame de Misericorde",
]


if 'tourist_site' not in st.session_state:
    st.title('Welcome to Benin!')
    tourist_site = st.selectbox('Select a tourist site to learn more about it.', tourist_sites_benin)
    if st.button('Explore'):
        st.session_state.tourist_site = tourist_site
        st.rerun()

else:
    if 'vector_store' not in st.session_state:
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        st.session_state.vector_store = PineconeVectorStore(index_name="benin-tourism", embedding=embeddings)

    st.header(f'Welcome to {st.session_state.tourist_site}!')
    st.subheader('Virtual Tour Guide 🤖')
    tabs = st.tabs(['Inquiries', 'Add Info'])

    with tabs[0]:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        st.write('Have any questions about the tourist site? Ask the virtual tour guide!')
        tts = st.checkbox("Enable Text-to-Speech")
        chatbox = st.container(height=300)

        with chatbox:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            query = st.empty()
            answer = st.empty()

        cols = st.columns([0.9, 0.1])
        error = st.empty()

        with cols[0]:
            prompt = st.chat_input("What's Up?")

        with cols[1]:
            audio_bytes = audio_recorder(text="", icon_size="2x")
            if audio_bytes and not prompt:
                file_name = "speech.mp3"
                try:
                    with open(file_name, "wb+") as audio_file:
                        audio_file.write(audio_bytes)
                        audio_file.seek(0)
                        transcript = transcribe(audio_file)

                    os.remove(file_name)
                    if transcript:
                        prompt = transcript
                except:
                    error.warning(
                        "The recorded audio is too short. Please record your inquiry again!",
                        icon="🚨",
                    )
        
        agent = setup_retriever(st.session_state.vector_store.as_retriever(search_kwargs={"namespace": st.session_state.tourist_site}))

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with query.chat_message("user"):
                st.write(prompt)
            with answer.chat_message("assistant"):
                with st.spinner('thinking.. 💭'):
                    response = agent.invoke({'input': prompt})['output']
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                if tts:
                    with st.spinner("generating audio.. 🎧"):
                        audio = text_to_speech(response)
                    st.audio(audio)

    with tabs[1]:
        lang = st.selectbox('Select the language of the information you want to add.', ['English', 'French', 'Yoruba', 'Fon'])
        content = st.text_area('Add more information about the tourist site to help other visitors.', height=200)
        if st.button('Submit'):
            content = translate_text(content, lang)
            st.session_state.vector_store.add_texts([content], namespace=st.session_state.tourist_site)
            with st.expander('Translation'):
                st.write(content)
            st.info('Information added to Database! ')
