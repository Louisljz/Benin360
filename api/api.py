from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
import shutil
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, AudioFileClip, CompositeAudioClip

import pysrt
import textwrap
import requests
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
app = FastAPI()
client = OpenAI()


def clean_temp_folder():
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    os.makedirs('temp')

def text_to_speech(text, output_filename):
    response = client.audio.speech.create(model="tts-1", voice="shimmer", input=text)
    with open(output_filename, "wb") as out:
        out.write(response.content)

def decode_audio(inFile, outFile):
    if not outFile.endswith(".mp3"):
        outFile += ".mp3"
    AudioSegment.from_file(inFile).set_channels(1).export(outFile, format="mp3")

def transcribe(file_path: str):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="srt"
        )
    with open("temp/transcript.srt", "w") as transcript_file:
        transcript_file.write(transcription)

def caption(video_file):
    subs = pysrt.open('temp/translated_output.srt', encoding='utf-8')
    subtitles_list = [((s.start.ordinal / 1000, s.end.ordinal / 1000), s.text) for s in subs]
    video = VideoFileClip(video_file)

    def generator(txt, start, end):
        wrapped_txt = textwrap.fill(txt, width=30)  # Wrap the text if it's too long
        return (TextClip(wrapped_txt, font='DejaVu-Sans', fontsize=80, color='white', stroke_color='black', stroke_width=1.5)
                .set_start(start)
                .set_duration(end - start)
                .set_position(('center', 'bottom')))

    subtitles = [generator(txt, start, end) for ((start, end), txt) in subtitles_list]
    video = CompositeVideoClip([video] + subtitles)
    video.write_videofile("temp/download.mp4", codec='libx264', audio_codec='aac')

def translate_srt(input_srt, output_srt, target_language='en'):
    subs = pysrt.open(input_srt, encoding='latin-1')

    url = 'https://translation.googleapis.com/language/translate/v2'

    for sub in subs:
        params = {
            'q': sub.text,
            'target': target_language,
            'key': os.getenv('GCLOUD_API_KEY')
        }
        response = requests.get(url, params=params)
        sub.text = response.json()['data']['translations'][0]['translatedText']

    subs.save(output_srt, encoding='utf-8')

def add_tts_to_video(video_path, srt_path):
    subs = pysrt.open(srt_path, encoding='utf-8')
    tts_audios = []
    for i, sub in enumerate(subs):
        tts_audio_path = f"temp/tts_{i}.wav"
        text_to_speech(sub.text, tts_audio_path)
        audio_clip = AudioFileClip(tts_audio_path)
        duration = audio_clip.duration  # Get the actual duration of the TTS audio clip
        audio_clip = audio_clip.set_start(sub.start.ordinal / 1000).set_duration(duration)
        tts_audios.append(audio_clip)

    video = VideoFileClip(video_path)
    final_audio = CompositeAudioClip(tts_audios).set_duration(video.duration)
    final_video = video.set_audio(final_audio)
    final_video.write_videofile("temp/video_with_tts.mp4", codec='libx264', audio_codec='aac')

def cap(video_file, target_language, dubbing):
    # Step 1: Decode audio from video
    audio_file_path = "temp/audio.mp3"
    decode_audio(video_file, audio_file_path)
    
    # Step 2: Transcribe audio to SRT
    transcribe(audio_file_path)
    
    # Step 3: Translate the SRT
    translate_srt("temp/transcript.srt", "temp/translated_output.srt", target_language=target_language)
    
    if dubbing:
        # Step 4: Add TTS to video
        add_tts_to_video(video_file, "temp/translated_output.srt")
        # Step 5: Add captions to the video with TTS audio
        caption("temp/video_with_tts.mp4")
    else:
        # Step 4: Add captions to the original video
        caption(video_file)


@app.post("/process_video")
def process_video(file: UploadFile = File(...), target_language: str = Form(...), dub: bool = Form(...)):
    clean_temp_folder()

    contents = file.file.read()
    file_path = os.path.join("temp", file.filename)
    with open(file_path, 'wb') as f:
        f.write(contents)

    cap(file_path, target_language, dub)

    return FileResponse("temp/download.mp4", media_type='video/mp4', filename="download.mp4")
