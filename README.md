# Benin 360
Explore Benin tourist sites with a multilingual virtual tour guide, and introductory tour videos, adapted to your foreign language.

## FrontEnd
Provider: Streamlit

Tech Stack:
- Langchain for Agents & RAG Wrapper
- Pinecone for Vectorstore
- modernmt for african dialect translation
- openai for llm, speech & text services

## BackEnd
Provider: Fast API

Tech Stack:
- torch for GPU support
- transformers to run hugging face models
    1. https://huggingface.co/neoform-ai/whisper-medium-yoruba
    2. https://huggingface.co/chrisjay/fonxlsr
- moviepy, pydub, pysrt for subtitles & dubbing

## Test App Locally
1. `cd ./api` -> `uvicorn api:app --reload`
2. `cd ./client` -> `streamlit run Benin_Virtual_Tour.py`

## Docker API Setup

1. `docker build -t benin360-raw .`
2. run container through docker hub, open file editor.
3. go to `etc/ImageMagick-6/policy.xml`, and comment out a security policy.
4. `docker commit <container-name> gcr.io/benin360/api:starter`
5. `docker push gcr.io/benin360/api:starter`
6. Navigate to GCloud Console -> Cloud Run -> Create Service -> Select image from artifact registry -> Configure VM specs --> Deploy!